"""
Unit tests for AutoMode core switching logic.

The most important test is test_lora_fft_roundtrip_changes_param_count — this
is the test that would have caught the silent prefix-matching bug in the 2B
notebook (where AutoMode appeared to run but every switch was a no-op).

How to run
----------
    cd /path/to/automode_pkg
    pytest tests/test_switching.py -v

Or without pytest:
    python tests/test_switching.py
"""

import os
import sys
import math
import unittest

# Allow running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, TaskType

from automode.core import (
    identify_layer_for_param,
    identify_layer_for_module,
    get_all_layer_ids,
    get_transformer_layers,
    group_lora_modules_by_layer,
    freeze_non_layer_params,
    count_trainable_params,
    promote_to_fft,
    demote_to_lora,
    current_layer_mode,
    AutoModeController,
    GradNormCollector,
    EMAGradCollector,
    RandomCollector,
)
from automode.config import RunConfig


# A tiny stand-in for a transformer decoder — two "layers" each with a q_proj
# and v_proj. This lets us test the switching logic without downloading a real
# LLM (which would be slow and fragile on CI).
class FakeDecoderLayer(nn.Module):
    def __init__(self, d=16):
        super().__init__()
        self.self_attn = nn.ModuleDict({
            "q_proj": nn.Linear(d, d, bias=False),
            "v_proj": nn.Linear(d, d, bias=False),
            "k_proj": nn.Linear(d, d, bias=False),
            "o_proj": nn.Linear(d, d, bias=False),
        })

    def forward(self, x):
        return self.self_attn.o_proj(self.self_attn.v_proj(x))


class FakeInnerModel(nn.Module):
    def __init__(self, n_layers=3, d=16, vocab=100):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList([FakeDecoderLayer(d) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d)

    def forward(self, input_ids):
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x) + x
        return self.norm(x)


class FakeCausalLM(nn.Module):
    """Minimal causal LM with the `.model.layers` hierarchy PEFT expects."""
    def __init__(self, n_layers=3, d=16, vocab=100):
        super().__init__()
        self.model = FakeInnerModel(n_layers=n_layers, d=d, vocab=vocab)
        self.lm_head = nn.Linear(d, vocab, bias=False)

    def forward(self, input_ids=None, labels=None, attention_mask=None,
            position_ids=None, past_key_values=None, inputs_embeds=None,
            use_cache=None, output_attentions=None, output_hidden_states=None,
            return_dict=None, **kwargs):
        h = self.model(input_ids)
        logits = self.lm_head(h)
        out = {"logits": logits}
        if labels is not None:
            out["loss"] = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1),
            )
        return out


def wrap_with_lora(model, r=4, alpha=8, targets=("q_proj", "v_proj")):
    """Apply PEFT LoRA to our fake model."""
    cfg = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=0.0,
        target_modules=list(targets),
        bias="none",
        # PEFT needs task_type to build its wrapper. CAUSAL_LM works but we
        # bypass the internal forward expectations since we control .forward.
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    return get_peft_model(model, cfg)


# ──────────────────────────────────────────────────────────────────────────────
# Tests: layer identification
# ──────────────────────────────────────────────────────────────────────────────

class TestLayerIdentification(unittest.TestCase):

    def test_plain_model_path(self):
        self.assertEqual(
            identify_layer_for_param("model.layers.7.self_attn.q_proj.weight"),
            "layer_7",
        )

    def test_peft_wrapped_path(self):
        self.assertEqual(
            identify_layer_for_param(
                "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight"
            ),
            "layer_0",
        )

    def test_double_peft_wrap(self):
        self.assertEqual(
            identify_layer_for_param(
                "base_model.model.base_model.model.model.layers.12.self_attn.v_proj.lora_B.default.weight"
            ),
            "layer_12",
        )

    def test_non_layer_returns_none(self):
        self.assertIsNone(identify_layer_for_param("lm_head.weight"))
        self.assertIsNone(identify_layer_for_param("base_model.model.model.embed_tokens.weight"))
        self.assertIsNone(identify_layer_for_param("model.norm.weight"))

    def test_gets_all_layer_ids(self):
        m = FakeCausalLM(n_layers=5)
        ids = get_all_layer_ids(m)
        self.assertEqual(ids, ["layer_0", "layer_1", "layer_2", "layer_3", "layer_4"])

    def test_layer_ids_sorted_numerically_not_alphabetically(self):
        """This catches a classic bug: sorted() gives [layer_0, layer_1, layer_10, layer_2, ...]"""
        m = FakeCausalLM(n_layers=12)
        ids = get_all_layer_ids(m)
        # Not ["layer_0", "layer_1", "layer_10", "layer_11", "layer_2", ...]
        expected = [f"layer_{i}" for i in range(12)]
        self.assertEqual(ids, expected)

    def test_get_transformer_layers_plain(self):
        m = FakeCausalLM(n_layers=4)
        layers = get_transformer_layers(m)
        self.assertEqual(len(layers), 4)

    def test_get_transformer_layers_peft_wrapped(self):
        m = FakeCausalLM(n_layers=4)
        wrapped = wrap_with_lora(m)
        layers = get_transformer_layers(wrapped)
        self.assertEqual(len(layers), 4)


# ──────────────────────────────────────────────────────────────────────────────
# Tests: LoraLayer grouping (the bit that the prefix bug broke)
# ──────────────────────────────────────────────────────────────────────────────

class TestLoraGrouping(unittest.TestCase):

    def test_group_by_layer_finds_all_targets(self):
        m = FakeCausalLM(n_layers=3)
        m = wrap_with_lora(m, targets=("q_proj", "v_proj"))
        groups = group_lora_modules_by_layer(m)
        # 3 layers × 2 target modules = 3 groups of 2
        self.assertEqual(len(groups), 3)
        for lid in ["layer_0", "layer_1", "layer_2"]:
            self.assertIn(lid, groups)
            self.assertEqual(len(groups[lid]), 2)  # q_proj + v_proj

    def test_group_handles_all_seven_targets(self):
        m = FakeCausalLM(n_layers=2)
        m = wrap_with_lora(m, targets=("q_proj", "k_proj", "v_proj", "o_proj"))
        groups = group_lora_modules_by_layer(m)
        self.assertEqual(len(groups), 2)
        for lid in ["layer_0", "layer_1"]:
            self.assertEqual(len(groups[lid]), 4)


# ──────────────────────────────────────────────────────────────────────────────
# THE CRITICAL TEST: the one that would have caught the prefix bug.
# ──────────────────────────────────────────────────────────────────────────────

class TestSwitchingRoundtrip(unittest.TestCase):
    """
    If promote_to_fft() silently does nothing (the original bug), then
    switching a layer to FFT should NOT change the trainable parameter count.

    These tests assert that the count DOES change — which is what the original
    sanity check failed to check. The 2B notebook's "passed" sanity check was
    reporting 3,194,880 trainable params before AND after "switching 23 layers
    to FFT", when the actual expected count was ~2.3 billion.
    """

    def setUp(self):
        self.model = FakeCausalLM(n_layers=4, d=32)
        self.model = wrap_with_lora(self.model, r=4, alpha=8)
        freeze_non_layer_params(self.model)

    def test_initial_state_has_lora_only_trainable(self):
        trainable, total = count_trainable_params(self.model)
        # Only lora_A, lora_B adapters should be trainable
        self.assertGreater(trainable, 0)
        self.assertLess(trainable, total)
        # lm_head (32 * 100 = 3200) and embeddings should be frozen
        for name, p in self.model.named_parameters():
            if "lm_head" in name or "embed_tokens" in name:
                self.assertFalse(p.requires_grad, f"{name} should be frozen")

    def test_promote_increases_trainable_count(self):
        """THE critical test. Catches the prefix bug."""
        trainable_before, _ = count_trainable_params(self.model)

        groups = group_lora_modules_by_layer(self.model)
        promote_to_fft(groups["layer_0"])

        trainable_after, _ = count_trainable_params(self.model)

        # Layer 0's q_proj + v_proj base weights should now be trainable.
        # Each is a 32x32 Linear = 1024 params. Total new trainable = 2048.
        # LoRA adapters for layer 0 are now frozen (lost: 2 * 2 * (32*4) = 512).
        # Net: should grow significantly.
        self.assertGreater(
            trainable_after, trainable_before,
            f"After promote_to_fft, trainable params MUST increase. "
            f"Before: {trainable_before}, After: {trainable_after}. "
            f"This test failing means switching is a silent no-op (the prefix bug)."
        )

    def test_demote_restores_trainable_count(self):
        """Round-trip: LoRA → FFT → LoRA returns to original count."""
        trainable_init, _ = count_trainable_params(self.model)

        groups = group_lora_modules_by_layer(self.model)
        promote_to_fft(groups["layer_0"])
        trainable_fft, _ = count_trainable_params(self.model)
        self.assertGreater(trainable_fft, trainable_init)

        demote_to_lora(groups["layer_0"])
        trainable_back, _ = count_trainable_params(self.model)

        # Should be back to the original count (within tolerance)
        self.assertEqual(
            trainable_back, trainable_init,
            f"Round-trip failed: {trainable_init} → {trainable_fft} → {trainable_back}"
        )

    def test_demote_zeros_lora_B(self):
        """After demotion, B should be zeroed so BA=0 (fresh adapter)."""
        groups = group_lora_modules_by_layer(self.model)
        promote_to_fft(groups["layer_0"])
        demote_to_lora(groups["layer_0"])

        for mod in groups["layer_0"]:
            for name, p in mod.lora_B.named_parameters():
                self.assertTrue(
                    torch.allclose(p, torch.zeros_like(p)),
                    f"lora_B for {name} was not zeroed after demotion"
                )

    def test_current_layer_mode_reports_correctly(self):
        groups = group_lora_modules_by_layer(self.model)
        mods = groups["layer_0"]

        self.assertEqual(current_layer_mode(mods), "lora")
        promote_to_fft(mods)
        self.assertEqual(current_layer_mode(mods), "full_ft")
        demote_to_lora(mods)
        self.assertEqual(current_layer_mode(mods), "lora")


# ──────────────────────────────────────────────────────────────────────────────
# Tests: importance signal collectors produce sane outputs
# ──────────────────────────────────────────────────────────────────────────────

class TestImportanceSignals(unittest.TestCase):

    def setUp(self):
        self.model = FakeCausalLM(n_layers=3, d=16)
        self.model = wrap_with_lora(self.model, r=4)
        freeze_non_layer_params(self.model)
        # Run one forward+backward so params have .grad populated
        input_ids = torch.randint(0, 100, (2, 8))
        labels = input_ids.clone()
        out = self.model(input_ids, labels=labels)
        out["loss"].backward()

    def test_grad_norm_collector(self):
        c = GradNormCollector()
        c.accumulate(self.model)
        scores = c.compute_scores()
        # Three layers, each should have a positive score
        self.assertEqual(len(scores), 3)
        for lid, s in scores.items():
            self.assertIn(lid, {"layer_0", "layer_1", "layer_2"})
            self.assertGreater(s, 0, f"{lid} got zero score — did grads propagate?")

    def test_ema_grad_collector_smooths_across_calls(self):
        c = EMAGradCollector(decay=0.5)
        c.accumulate(self.model)
        s1 = c.compute_scores()

        # Second backward pass with different input
        for p in self.model.parameters():
            if p.grad is not None:
                p.grad.zero_()
        input_ids = torch.randint(0, 100, (2, 8))
        out = self.model(input_ids, labels=input_ids)
        out["loss"].backward()
        c.accumulate(self.model)
        s2 = c.compute_scores()

        # EMA values should be between the two raw observations
        self.assertEqual(set(s1.keys()), set(s2.keys()))
        # s2 should not equal s1 (EMA updated with new observation)
        # but shouldn't be wildly different either
        for lid in s1:
            self.assertNotAlmostEqual(s1[lid], s2[lid], places=8)

    def test_random_collector(self):
        c = RandomCollector()
        c.accumulate(self.model)
        scores = c.compute_scores()
        self.assertEqual(len(scores), 3)
        # Each in [0, 1)
        for s in scores.values():
            self.assertGreaterEqual(s, 0.0)
            self.assertLess(s, 1.0)


# ──────────────────────────────────────────────────────────────────────────────
# Tests: AutoModeController integration
# ──────────────────────────────────────────────────────────────────────────────

class TestControllerIntegration(unittest.TestCase):

    def setUp(self):
        self.model = FakeCausalLM(n_layers=4, d=32)
        self.model = wrap_with_lora(self.model, r=4, alpha=8)
        freeze_non_layer_params(self.model)

        self.cfg = RunConfig(
            method="automode",
            epochs=1,
            dynamic_updates=2,
            dynamic_threshold=50,  # 50th percentile → top half goes FFT
            importance_signal="grad_norm",
        )

    def test_controller_attaches_cleanly(self):
        ctrl = AutoModeController(self.cfg, total_optimizer_steps=20)
        ctrl.attach(self.model)
        self.assertEqual(len(ctrl._lora_groups), 4)
        self.assertEqual(len(ctrl._layer_mode), 4)
        for lid, mode in ctrl._layer_mode.items():
            self.assertEqual(mode, "lora")

    def test_controller_switches_at_correct_interval(self):
        """With total_steps=20 and u=2, update_interval should be ~10."""
        ctrl = AutoModeController(self.cfg, total_optimizer_steps=20)
        self.assertEqual(ctrl.update_interval, 10)

    def test_controller_end_to_end_switch(self):
        ctrl = AutoModeController(self.cfg, total_optimizer_steps=4)
        # With total=4 and u=2, interval=2
        ctrl.attach(self.model)

        # Simulate training: forward -> backward -> accumulate -> switch
        input_ids = torch.randint(0, 100, (2, 8))
        labels = input_ids.clone()

        trainable_snapshots = []
        for opt_step in range(1, 5):
            out = self.model(input_ids, labels=labels)
            out["loss"].backward()
            ctrl.on_micro_step(self.model)
            for p in self.model.parameters():
                if p.grad is not None:
                    p.grad = None
            changed = ctrl.maybe_switch(self.model, opt_step)
            trainable, _ = count_trainable_params(self.model)
            trainable_snapshots.append((opt_step, changed, trainable))

        # Switches should have fired at steps 2 and 4 (multiples of interval=2)
        # NEW:
        decision_steps = [h["step"] for h in ctrl.switch_history]
        self.assertEqual(
            decision_steps, [2, 4],
            f"Expected switching decisions at [2, 4], got {decision_steps}"
        )

        trainable_counts = [t for _, _, t in trainable_snapshots]
        self.assertGreater(
            max(trainable_counts), min(trainable_counts),
            "No capacity reallocation occurred across any decision point."
        )


# ──────────────────────────────────────────────────────────────────────────────
# Tests: freeze_non_layer_params doesn't accidentally freeze LoRA
# ──────────────────────────────────────────────────────────────────────────────

class TestFreezeNonLayerParams(unittest.TestCase):

    def test_freezes_lm_head_and_embeddings(self):
        m = FakeCausalLM(n_layers=2)
        m = wrap_with_lora(m)
        freeze_non_layer_params(m)
        for name, p in m.named_parameters():
            if "lm_head" in name or "embed_tokens" in name:
                self.assertFalse(p.requires_grad, f"{name} should be frozen")

    def test_preserves_lora_adapter_trainability(self):
        m = FakeCausalLM(n_layers=2)
        m = wrap_with_lora(m)
        freeze_non_layer_params(m)
        lora_trainable = [
            name for name, p in m.named_parameters()
            if "lora_" in name and p.requires_grad
        ]
        self.assertGreater(
            len(lora_trainable), 0,
            "LoRA adapters should still be trainable after freeze_non_layer_params"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
