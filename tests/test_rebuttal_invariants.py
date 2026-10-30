"""
Regression tests for the rebuttal-campaign corrections.

Every model here is a tiny, locally constructed Gemma-2 (real q_proj/v_proj
naming, real PEFT LoRA) with a character-level tokenizer: no downloads, no
gated access. Set AUTOMODE_TEST_DEVICE=cuda (remote preflight does) to run the
model tests on the GPU; that run FAILS rather than skips if CUDA is missing.

    pytest -q tests/test_rebuttal_invariants.py
"""

import copy
import dataclasses
import json
import math
import os
import random
import signal
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from automode import rebuttal  # noqa: E402
from automode.analysis import switch_diagnostics, summarize_runs, paired_seed_differences, paired_item_table  # noqa: E402
from automode.config import RunConfig, CORE_METHODS, UNAVAILABLE_METHODS, NON_SCIENTIFIC_FIELDS  # noqa: E402
from automode.core import (  # noqa: E402
    AutoModeController, GradNormCollector, LORA_INIT_DESCRIPTION, assert_trainable_scope, demote_to_lora,
    fixed_layer_allocation, group_lora_modules_by_layer, promote_to_fft, rebuild_optimizer,
)
from automode.data import (  # noqa: E402
    CausalLMCollator, build_sft_example, make_train_dataloader, tokenize_examples,
)
from automode.eval import (  # noqa: E402
    EVALUATORS, evaluate_gsm8k, extract_gsm8k_answer, normalize_number, plurality_vote,
)
from automode.models import apply_method  # noqa: E402
from automode.train import train_one_run  # noqa: E402
from peft.tuners.lora import LoraLayer  # noqa: E402

DEVICE = torch.device(os.environ.get("AUTOMODE_TEST_DEVICE", "cpu"))
MANIFEST = REPO / "configs" / "rebuttal_core.json"


# ──────────────────────────────────────────────────────────────────────────────
# Tiny local fixtures
# ──────────────────────────────────────────────────────────────────────────────

def tiny_tokenizer():
    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast
    specials = ["<pad>", "<eos>", "<bos>", "<unk>"]
    vocab = {t: i for i, t in enumerate(specials + [chr(i) for i in range(32, 127)] + ["\n"])}
    tk = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Split(Regex(r"[\s\S]"), behavior="isolated")
    tk.decoder = decoders.Fuse()
    tk.post_processor = processors.TemplateProcessing(single="<bos> $A", special_tokens=[("<bos>", 2)])
    return PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="<pad>", eos_token="<eos>",
                                   bos_token="<bos>", unk_token="<unk>")


def tiny_gemma(n_layers=4, seed=0, dtype=torch.float32):
    from transformers import Gemma2Config, Gemma2ForCausalLM
    torch.manual_seed(seed)
    cfg = Gemma2Config(vocab_size=100, hidden_size=32, intermediate_size=64, num_hidden_layers=n_layers,
                       num_attention_heads=2, num_key_value_heads=1, head_dim=16, max_position_embeddings=512,
                       sliding_window=256, pad_token_id=0, eos_token_id=1, bos_token_id=2,
                       attn_implementation="eager")
    # Parameters only, as from_pretrained(torch_dtype=...) leaves them: buffers keep their precision.
    model = Gemma2ForCausalLM(cfg).to(device=DEVICE)
    for p in model.parameters():
        p.data = p.data.to(dtype)
    model.config.use_cache = False
    return model


def make_cfg(method="automode", **kw):
    base = dict(method=method, model_name="tiny-local", device=str(DEVICE), dtype="float32",
                gradient_checkpointing=False, epochs=1, learning_rate=1e-2, lr_scheduler="linear",
                warmup_ratio=0.1, train_batch_size=2, grad_accum_steps=2, lora_r=4, lora_alpha=8,
                lora_dropout=0.05, dynamic_updates=2, dynamic_threshold=50, max_source_len=64,
                max_target_len=16, max_new_tokens=8, eval_batch_size=2, sampling_k=3,
                protocol_version="test", evaluation_seed=123, seed=8, data_seed=8)
    if method == "fixed_mixed":
        base.update(fixed_base_layers=2, allocation_seed=3)
    base.update(kw)
    return RunConfig(**base)


TRAIN_EXAMPLES = [{"prompt": f"Question: what is {i} plus {i + 1}?\n\nAnswer:",
                   "target": f"{i} + {i + 1} = {2 * i + 1}\n#### {2 * i + 1}"} for i in range(13)]


def tiny_loader(cfg, tok, examples=TRAIN_EXAMPLES):
    feats, stats = tokenize_examples(examples, tok, cfg)
    return make_train_dataloader(feats, stats, tok, cfg)


def tiny_gsm8k_test(n=3):
    from datasets import Dataset
    return Dataset.from_dict({"question": [f"what is {i} plus {i}?" for i in range(n)],
                              "answer": [f"{i}+{i}={2 * i}\n#### {2 * i}" for i in range(n)]})


def run_paths(root: Path):
    paths = {k: root / k for k in ("configs", "logs", "dynamic", "evals")}
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    paths["run"] = root
    return paths


@torch.no_grad()
def logits(model, ids):
    model.eval()
    return model(input_ids=ids)["logits"].float()


@pytest.fixture
def restore_signals():
    old = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
    yield
    for s, h in old.items():
        signal.signal(s, h)


def test_requested_test_device_is_available():
    if DEVICE.type == "cuda":
        assert torch.cuda.is_available(), "AUTOMODE_TEST_DEVICE=cuda but CUDA is unavailable"


# ──────────────────────────────────────────────────────────────────────────────
# 1. Optimizer / scheduler continuity
# ──────────────────────────────────────────────────────────────────────────────

def _lr_trace(rebuild_at, buggy=False, total=30, warmup=5):
    from transformers import get_scheduler
    torch.manual_seed(0)
    model = nn.ParameterList([nn.Parameter(torch.randn(4)), nn.Parameter(torch.randn(4))])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    sched = get_scheduler("linear", opt, num_warmup_steps=warmup, num_training_steps=total)

    def rebuild(o):
        if not buggy:
            return rebuild_optimizer(model, o, sched)
        # The submitted attach_optimizer_to_scheduler, kept here only to show the
        # test detects it: base_lrs overwritten with the current decayed lr.
        cur = o.param_groups[0]["lr"]
        new = torch.optim.AdamW(model.parameters(), lr=cur, weight_decay=0.01)
        sched.optimizer = new
        sched.base_lrs = [cur for _ in sched.base_lrs]
        return new

    if 0 in rebuild_at:
        opt = rebuild(opt)
    lrs = []
    for step in range(1, total + 1):
        lrs.append(opt.param_groups[0]["lr"])
        sum(p.sum() for p in model).backward()
        opt.step()
        sched.step()
        opt.zero_grad()
        if step in rebuild_at:
            opt = rebuild(opt)
    return lrs, sched, opt


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_lr_sequence_identical_with_forced_rebuilds_including_warmup():
    ref, _, _ = _lr_trace(set())
    got, sched, opt = _lr_trace({0, 2, 4, 9, 20})  # 0, 2, 4 are inside warmup
    assert got == ref  # exact, every step
    assert sched.base_lrs == [1e-3] and opt.param_groups[0]["initial_lr"] == 1e-3
    assert sched.last_epoch == 30
    bug, _, _ = _lr_trace({0, 2, 4, 9, 20}, buggy=True)
    assert bug != ref and max(bug[1:]) == 0.0  # the old path froze lr at warmup's 0


def test_rebuild_preserves_every_param_group_hyperparameter():
    from transformers import get_scheduler
    p = nn.Parameter(torch.randn(3))
    model = nn.ParameterList([p])
    opt = torch.optim.AdamW([p], lr=3e-4, betas=(0.8, 0.95), eps=1e-6, weight_decay=0.05,
                            amsgrad=True, foreach=False)
    sched = get_scheduler("linear", opt, num_warmup_steps=2, num_training_steps=10)
    p.sum().backward(); opt.step(); sched.step()
    new = rebuild_optimizer(model, opt, sched)
    strip = lambda g: {k: v for k, v in g.items() if k != "params"}  # noqa: E731
    assert strip(new.param_groups[0]) == strip(opt.param_groups[0])
    assert new.defaults == opt.defaults and sched.optimizer is new and type(new) is type(opt)
    assert p.grad is None


def test_adam_state_carried_by_identity_and_cleared_on_leave_reset_and_activation():
    torch.manual_seed(0)
    keep, leave, new, reset = (nn.Parameter(torch.randn(5)) for _ in range(4))
    model = nn.ParameterList([keep, leave, new, reset])
    new.requires_grad = False
    opt = torch.optim.AdamW([keep, leave, reset], lr=1e-2)
    for _ in range(3):
        (keep.pow(2).sum() + leave.sum() + reset.pow(3).sum()).backward()
        opt.step(); opt.zero_grad()
    snap = {k: v.clone() for k, v in opt.state[keep].items()}
    leave.requires_grad = False
    new.requires_grad = True
    with torch.no_grad():
        reset.zero_()  # in-place reset: same tensor identity
    opt2 = rebuild_optimizer(model, opt, None, reset_params=[reset])
    for k, v in snap.items():
        assert torch.equal(opt2.state[keep][k], v)
    assert leave not in opt2.state
    assert len(opt2.state[new]) == 0 and len(opt2.state[reset]) == 0
    (keep.sum() + new.sum() + reset.sum()).backward()
    opt2.step()
    assert float(opt2.state[keep]["step"]) == 4.0 and float(opt2.state[reset]["step"]) == 1.0


def test_noop_rebuild_leaves_training_trajectory_bitwise_identical():
    def train(rebuild_at):
        torch.manual_seed(1)
        model = nn.Linear(6, 3)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.01)
        x = torch.randn(16, 6)
        for step in range(8):
            model(x).pow(2).mean().backward(); opt.step(); opt.zero_grad()
            if step in rebuild_at:
                opt = rebuild_optimizer(model, opt)
        return torch.cat([p.detach().flatten() for p in model.parameters()])
    assert torch.equal(train(set()), train({1, 4, 6}))


# ──────────────────────────────────────────────────────────────────────────────
# 2. Promotion, demotion, initialisation, export
# ──────────────────────────────────────────────────────────────────────────────

def _automode_model(**kw):
    cfg = make_cfg("automode", **kw)
    torch.manual_seed(0)
    model = apply_method(tiny_gemma(), cfg)
    with torch.no_grad():  # non-zero adapters so merges actually move base weights
        for n, p in model.named_parameters():
            if "lora_B" in n:
                p.normal_(0, 0.5)
    return cfg, model


def test_promote_then_train_base_then_demote_preserves_eval_function():
    cfg, model = _automode_model()
    groups = group_lora_modules_by_layer(model)
    mods = groups["layer_1"]
    x = torch.randint(3, 100, (2, 12), device=DEVICE)
    q_base = mods[0].get_base_layer().weight
    w0 = q_base.detach().clone()

    before = logits(model, x)
    promote_to_fft(mods)
    assert torch.allclose(logits(model, x), before, atol=1e-5)
    assert not torch.equal(q_base, w0)  # the delta really went into the base
    assert all(p.requires_grad for m in mods for p in m.get_base_layer().parameters())
    assert not any(p.requires_grad for m in mods for p in list(m.lora_A.parameters()) + list(m.lora_B.parameters()))
    with pytest.raises(RuntimeError):
        promote_to_fft(mods)  # already merged: no silent double merge

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    model.train()
    model(input_ids=x, labels=x)["loss"].backward()
    opt.step(); opt.zero_grad(set_to_none=True)
    trained = q_base.detach().clone()

    before_demote = logits(model, x)
    reset = demote_to_lora(mods, torch.Generator().manual_seed(5))
    assert torch.allclose(logits(model, x), before_demote, atol=1e-5)
    assert torch.equal(q_base, trained)  # no unmerge of a directly-updated base
    for m in mods:
        A, B = m.lora_A["default"].weight, m.lora_B["default"].weight
        assert torch.count_nonzero(B) == 0 and A.abs().max() <= 1 / math.sqrt(A.shape[1]) + 1e-7
        assert any(r is A for r in reset) and any(r is B for r in reset)
        assert m.merged_adapters == []
    model.train()
    model(input_ids=x, labels=x)["loss"].backward()
    for m in mods:
        assert m.lora_B["default"].weight.grad.abs().sum() > 0  # adapter path active again
        assert m.get_base_layer().weight.grad is None
    with pytest.raises(RuntimeError):
        demote_to_lora(mods)  # not merged: unexpected state must fail


def test_lora_init_is_uniform_kaiming_in_peft_and_on_reset():
    assert "Uniform" in LORA_INIT_DESCRIPTION and "NOT Gaussian" not in LORA_INIT_DESCRIPTION
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(0)
    base = nn.Sequential()
    base.add_module("q_proj", nn.Linear(512, 512, bias=False))
    model = get_peft_model(base, LoraConfig(r=64, lora_alpha=64, target_modules=["q_proj"]))
    mod = model.base_model.model.q_proj
    bound = 1 / math.sqrt(512)
    for A, B in [(mod.lora_A["default"].weight, mod.lora_B["default"].weight)]:
        assert torch.count_nonzero(B) == 0
        assert A.abs().max() <= bound and A.abs().max() > 0.99 * bound  # bounded, fills the support
        assert abs(A.std().item() - bound / math.sqrt(3)) < 0.03 * bound
    peft_A = mod.lora_A["default"].weight.detach().clone()
    mod.merge(adapter_names=["default"])
    demote_to_lora([mod], torch.Generator().manual_seed(1))
    A = mod.lora_A["default"].weight
    assert A.abs().max() <= bound and abs(A.std().item() - peft_A.std().item()) < 0.03 * bound


def test_full_export_round_trip_keeps_base_updates_and_adapter_only_save_loses_them(tmp_path):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    from automode.export import export_full_inference_model, load_unmerged_export
    tok = tiny_tokenizer()
    tiny_gemma().save_pretrained(tmp_path / "orig_base")  # same seed as _automode_model's base
    cfg, model = _automode_model()
    groups = group_lora_modules_by_layer(model)
    promote_to_fft(groups["layer_0"]); promote_to_fft(groups["layer_2"])
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    x = torch.randint(3, 100, (2, 12), device=DEVICE)
    model.train()
    for _ in range(3):
        model(input_ids=x, labels=x)["loss"].backward(); opt.step(); opt.zero_grad()
    live = logits(model, x)

    model.save_pretrained(tmp_path / "adapter_only")  # what the submitted code saved
    lossy = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(tmp_path / "orig_base").to(DEVICE), tmp_path / "adapter_only")
    assert (logits(lossy, x) - live).abs().max() > 1e-3

    info = export_full_inference_model(model, tok, tmp_path / "export", cfg, {"run": "test"})
    assert info["status"] == "ok" and info["resumable_training_checkpoint"] is False
    assert info["n_lora_layers_already_merged"] == 4 and info["n_lora_layers_unmerged_in_export"] == 4
    reloaded = load_unmerged_export(tmp_path / "export", DEVICE)
    assert torch.allclose(logits(reloaded, x), live, atol=1e-4)
    meta = json.loads((tmp_path / "export" / "export_metadata.json").read_text())
    assert meta["kind"] == "full_unmerged_inference_export"


def automode_autocast(cfg):
    from automode.train import autocast_ctx
    return autocast_ctx(cfg, DEVICE)


def _mixed_automode_model(dtype, adapter_std=2e-4):
    """Every state an AutoMode export can meet, with adapters small next to the base
    weights like the campaign's: layer_0 merged and base-trained; layer_1 merged,
    base-trained, demoted, adapter retrained; layer_2 LoRA-trained; layer_3 zero delta."""
    cfg = make_cfg("automode", dtype={torch.float32: "float32", torch.bfloat16: "bfloat16"}[dtype], lora_dropout=0.0)
    torch.manual_seed(0)
    model = apply_method(tiny_gemma(dtype=dtype), cfg)
    groups = group_lora_modules_by_layer(model)

    def set_adapters(mods):
        with torch.no_grad():
            for m in mods:
                m.lora_B["default"].weight.normal_(0, adapter_std)
    set_adapters(groups["layer_0"] + groups["layer_1"] + groups["layer_2"])
    promote_to_fft(groups["layer_0"]); promote_to_fft(groups["layer_1"])

    x = torch.randint(3, 100, (2, 12), device=DEVICE)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    model.train()
    model(input_ids=x, labels=x)["loss"].backward(); opt.step(); opt.zero_grad(set_to_none=True)
    demote_to_lora(groups["layer_1"], torch.Generator().manual_seed(1))
    set_adapters(groups["layer_1"])
    with torch.no_grad():
        for m in groups["layer_3"]:
            m.lora_B["default"].weight.zero_()
    model.eval()
    return cfg, model, x


def test_unmerged_bf16_export_rebuilds_the_evaluated_model_exactly(tmp_path):
    from automode.export import export_full_inference_model, load_unmerged_export
    cfg, model, x = _mixed_automode_model(torch.bfloat16)
    live = logits(model, x)
    snapshot = {k: v.clone() for k, v in model.state_dict().items()}
    state_before = {n: list(m.merged_adapters) for n, m in model.named_modules() if isinstance(m, LoraLayer)}

    info = export_full_inference_model(model, tiny_tokenizer(), tmp_path / "export", cfg, {})
    assert info["status"] == "ok" and info["checks"]["reload_vs_live"]["max_abs_logit_diff"] <= 1e-4

    reloaded = load_unmerged_export(tmp_path / "export", DEVICE)
    assert torch.allclose(logits(reloaded, x), live, atol=1e-4)
    rebuilt = reloaded.state_dict()
    assert set(rebuilt) == set(snapshot)
    for k, v in snapshot.items():
        assert rebuilt[k].dtype == v.dtype and torch.equal(rebuilt[k], v), k  # learned base AND adapters
    assert {n: list(m.merged_adapters) for n, m in reloaded.named_modules()
            if isinstance(m, LoraLayer)} == state_before
    assert info["n_lora_layers_already_merged"] == 2 and info["n_lora_layers_unmerged_in_export"] == 6

    # The formula holds (no raise), yet a bf16 merge cannot keep these deltas.
    merged_now = [r for r in info["merge_audit"] if r["state"] == "merged_now"]
    with_delta = [r for r in merged_now if r["delta_error_ratio"] is not None]
    assert len(merged_now) == 6 and len(with_delta) == 4
    assert all(r["delta_error_ratio"] > 0.05 and r["unchanged_fraction"] > 0.05 for r in with_delta)
    assert {"max_total_variation", "max_kl", "mean_kl"} <= set(info["checks"]["merge_vs_live"])


@pytest.mark.parametrize("method,kw", [(m, {}) for m in CORE_METHODS] + [("automode", {"base_weight_dtype": "float32"})])
def test_bf16_export_verifies_for_every_core_method(method, kw, tmp_path):
    from automode.export import export_full_inference_model
    cfg = make_cfg(method, dtype="bfloat16", **kw)
    torch.manual_seed(0)
    model = apply_method(tiny_gemma(dtype=torch.bfloat16), cfg)
    x = torch.randint(3, 100, (2, 12), device=DEVICE)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    model.train()
    with automode_autocast(cfg):
        model(input_ids=x, labels=x)["loss"].backward()
    opt.step()
    info = export_full_inference_model(model, tiny_tokenizer(), tmp_path / "export", cfg, {})
    assert info["status"] == "ok" and info["checks"]["reload_vs_live"]["max_abs_logit_diff"] <= 1e-4
    assert info["kind"] == ("full_inference_export" if method == "static_qv" else "full_unmerged_inference_export")


def test_same_mixed_states_merge_faithfully_in_fp32(tmp_path):
    from automode.export import export_full_inference_model
    cfg, model, x = _mixed_automode_model(torch.float32)
    info = export_full_inference_model(model, tiny_tokenizer(), tmp_path / "export", cfg, {})
    assert info["status"] == "ok" and info["merged_model_within_tolerance"] is True
    assert info["checks"]["merge_vs_live"]["max_abs_logit_diff"] <= 1e-3
    assert info["checks"]["merge_vs_live"]["top1_agreement"] == 1.0
    assert all(r["delta_error_ratio"] < 1e-3 for r in info["merge_audit"]
               if r["state"] == "merged_now" and r["delta_error_ratio"] is not None)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_merge_audit_rejects_a_wrong_merge(dtype, monkeypatch):
    from peft.tuners.lora import Linear
    from automode.export import merge_and_audit
    cfg, model, _ = _mixed_automode_model(dtype, adapter_std=1e-2)
    real_merge = Linear.merge

    def double_merge(self, *a, **k):
        delta = self.get_delta_weight("default").detach()
        real_merge(self, *a, **k)
        self.get_base_layer().weight.data += delta.to(self.get_base_layer().weight.dtype)
    monkeypatch.setattr(Linear, "merge", double_merge)
    with pytest.raises(RuntimeError, match="does not match"):
        merge_and_audit(model)
    monkeypatch.setattr(Linear, "merge", real_merge)

    cfg, model, _ = _mixed_automode_model(dtype, adapter_std=1e-2)
    already = group_lora_modules_by_layer(model)["layer_0"][0].get_base_layer().weight

    def touching_merge(self, *a, **k):
        real_merge(self, *a, **k)
        already.data[0, 0] += 1.0
    monkeypatch.setattr(Linear, "merge", touching_merge)
    with pytest.raises(RuntimeError, match="Already-merged"):
        merge_and_audit(model)


def test_reload_mismatch_still_fails_and_is_recorded(tmp_path, monkeypatch):
    import automode.export as export
    cfg, model, _ = _mixed_automode_model(torch.bfloat16)
    real_load = export.load_unmerged_export

    def corrupted(out_dir, device):
        m = real_load(out_dir, device)
        with torch.no_grad():
            group_lora_modules_by_layer(m)["layer_3"][0].get_base_layer().weight.mul_(2)
        return m
    monkeypatch.setattr(export, "load_unmerged_export", corrupted)
    with pytest.raises(RuntimeError, match="Export verification failed"):
        export.export_full_inference_model(model, tiny_tokenizer(), tmp_path / "export", cfg, {})
    assert json.loads((tmp_path / "export" / "export_metadata.json").read_text())["status"] == "failed"


def test_export_diagnostic_measures_counterfactuals_without_test_data(tmp_path, tiny_loaders):
    from automode.export import base_update_assessment, export_diagnostic
    cfg, model, _ = _mixed_automode_model(torch.bfloat16)
    tok = tiny_tokenizer()
    cfg.max_new_tokens = 8
    batch = next(iter(tiny_loader(cfg, tok)))
    assessment = base_update_assessment(model, batch, [{"step": 1, "lr": 2e-5, "changed_fraction": 0.1}], cfg)
    assert 0.0 <= assessment["per_update"][0]["predicted_if_step_equals_lr"] <= 1.0
    assert assessment["n_trainable_base_elements"] > 0 and assessment["weight_dtypes"] == ["torch.bfloat16"]
    assert not any(p.grad is not None for p in model.parameters())

    prompts = [tok(p["prompt"])["input_ids"] for p in TRAIN_EXAMPLES[:3]]
    report = export_diagnostic(model, tok, prompts, tmp_path / "export", cfg, {"job_id": "t"})
    assert report["export"]["status"] == "ok"
    assert report["n_unmerged_projections"] == 6 and len(report["single_projection_merge_vs_live"]) == 6
    for key in ("exact_merge_vs_live", "adapter_removed_vs_live", "merged_vs_live", "merged_vs_adapter_removed"):
        assert report[key]["max_abs_logit_diff"] >= 0.0
    assert 0.0 <= report["greedy_generation"]["identical_text_fraction"] <= 1.0


def test_export_diagnostic_command_needs_preflight_then_reports_on_unseen_train_rows(tmp_path, tiny_loaders, monkeypatch):
    m = json.loads(MANIFEST.read_text())
    m["common"].update(train_batch_size=2, grad_accum_steps=2, max_source_len=64, max_target_len=16, device=str(DEVICE))
    monkeypatch.setattr(rebuttal, "_common_context", lambda a: (m, "e" * 64, SRC, tmp_path))
    monkeypatch.setattr(rebuttal, "_load_train_dataset",
                        lambda cfg: TRAIN_EXAMPLES[:8] if cfg.max_train_samples else TRAIN_EXAMPLES)
    monkeypatch.setattr(rebuttal, "_load_gsm8k_test", lambda cfg: pytest.fail("the diagnostic must not read test data"))
    args = rebuttal.argparse.Namespace(worker_id=0, output=tmp_path)
    assert rebuttal.cmd_export_diagnostic(args) == 1
    report_path = tmp_path / "export_diagnostic_report.json"
    assert "preflight_report.json missing" in json.loads(report_path.read_text())["error"]

    (tmp_path / "preflight_report.json").write_text(json.dumps({"status": "ok", "worker_id": 0, **SRC}))
    (tmp_path / "resolved_manifest.json").write_text(json.dumps(
        {"manifest_sha256": "e" * 64, "revisions": {"model_revision": "r", "dataset_revision": "d"}}))
    assert rebuttal.cmd_export_diagnostic(args) == 0
    report = json.loads(report_path.read_text())
    assert report["status"] == "ok" and report["export"]["status"] == "ok"
    assert report["n_diagnostic_prompts"] == 5  # the rows the pilot config did not train on
    assert "base_update_assessment" in report and "greedy_generation" in report
    assert not list(Path(report["attempt"]).rglob("*.safetensors"))  # weights are not kept


# ──────────────────────────────────────────────────────────────────────────────
# 3. Data: masks, padding, EOS, truncation, seeded order
# ──────────────────────────────────────────────────────────────────────────────

def test_prompt_mask_is_exact_with_eos_and_left_truncated_long_prompt():
    tok = tiny_tokenizer()
    prompt, target = "Question: 2+2?\n\nAnswer:", "#### 4"
    ex = build_sft_example(tok, prompt, target, 64, 16)
    n_prompt = len(tok(prompt)["input_ids"])
    assert ex["labels"][:n_prompt] == [-100] * n_prompt
    sup = ex["input_ids"][n_prompt:]
    assert ex["labels"][n_prompt:] == sup and sup[-1] == tok.eos_token_id
    assert tok.decode(sup, skip_special_tokens=True) == " #### 4"

    long_prompt = "x" * 200 + "\n\nAnswer:"
    ex = build_sft_example(tok, long_prompt, target, 32, 16)
    assert ex["prompt_truncated"] and ex["input_ids"][0] == tok.bos_token_id
    supervised = [l for l in ex["labels"] if l != -100]
    assert len(supervised) == ex["n_target_tokens"] == len(" #### 4") + 1  # no prompt token leaks into loss
    assert tok.decode(ex["input_ids"][:32], skip_special_tokens=True).endswith("Answer:")
    assert ex["labels"][:32] == [-100] * 32

    cut = build_sft_example(tok, prompt, "y" * 40, 64, 5)
    assert cut["target_truncated"] and cut["n_target_tokens"] == 5 and tok.eos_token_id not in cut["input_ids"]


def test_zero_target_examples_are_kept_counted_and_padding_never_supervised():
    tok = tiny_tokenizer()
    cfg = make_cfg(max_target_len=0)
    feats, stats = tokenize_examples(TRAIN_EXAMPLES[:3], tok, cfg)
    assert stats["n_examples"] == 3 and stats["n_empty_target"] == 3 and stats["rate_empty_target"] == 1.0
    assert all(set(f["labels"]) == {-100} for f in feats)

    a = build_sft_example(tok, "ab", "c", 64, 16)
    b = build_sft_example(tok, "abcdefgh", "ijk", 64, 16)
    batch = CausalLMCollator(tok)([a, b])
    L = len(b["input_ids"])
    assert batch["input_ids"].shape == (2, L)
    pad = L - len(a["input_ids"])
    assert batch["attention_mask"][0, -pad:].sum() == 0 and batch["attention_mask"][0, :-pad].all()
    assert (batch["labels"][0, -pad:] == -100).all() and (batch["input_ids"][0, -pad:] == tok.pad_token_id).all()


def test_data_order_depends_only_on_data_seed():
    tok = tiny_tokenizer()

    def order(cfg, burn):
        torch.manual_seed(burn)
        torch.randn(1000)  # different global RNG consumption (e.g. different param init)
        loader = tiny_loader(cfg, tok)
        return [b["input_ids"].tolist() for _ in range(cfg.epochs) for b in loader], loader.data_stats

    o1, s1 = order(make_cfg("automode", epochs=2), burn=1)
    o2, s2 = order(make_cfg("static_qv", epochs=2), burn=99)
    assert o1 == o2 and s1["data_order_sha256"] == s2["data_order_sha256"]
    o3, _ = order(make_cfg("automode", epochs=2, data_seed=9), burn=1)
    assert o3 != o1


# ──────────────────────────────────────────────────────────────────────────────
# 4. Method allocations (also run on GPU in preflight via AUTOMODE_TEST_DEVICE)
# ──────────────────────────────────────────────────────────────────────────────

def test_fixed_allocation_is_seeded_independently_of_training_rngs():
    a = fixed_layer_allocation(42, 37, 8)
    torch.manual_seed(123); np.random.seed(5); random.seed(7)
    assert fixed_layer_allocation(42, 37, 8) == a
    assert len(a) == 37 == len(set(a)) and a == sorted(a) and all(0 <= i < 42 for i in a)
    assert fixed_layer_allocation(42, 37, 25) != a
    with pytest.raises(ValueError):
        fixed_layer_allocation(42, 43, 1)


def _trainable_names(model):
    return [n for n, p in model.named_parameters() if p.requires_grad]


@pytest.mark.parametrize("method", CORE_METHODS)
def test_method_allocations_scope_and_forward_backward(method):
    cfg = make_cfg(method, **({"lora_r": 8, "lora_alpha": 16} if method == "lora" else {}))
    torch.manual_seed(0)
    plain = tiny_gemma(n_layers=6)
    x = torch.randint(3, 100, (2, 10), device=DEVICE)
    ref = logits(plain, x)
    model = apply_method(plain, cfg)
    assert_trainable_scope(model, cfg.lora_target_modules)
    names = _trainable_names(model)
    assert torch.allclose(logits(model, x), ref, atol=1e-5)  # zero-delta start for every method
    if method in ("automode", "select_once"):
        assert names and all("lora_" in n for n in names)
    elif method == "lora":
        assert all("lora_" in n for n in names)
        assert {m.r["default"] for m in group_lora_modules_by_layer(model)["layer_0"]} == {8}
    elif method == "static_qv":
        assert sorted(names) == sorted(f"model.layers.{i}.self_attn.{t}.weight" for i in range(6) for t in ("q_proj", "v_proj"))
    elif method == "fixed_mixed":
        chosen = fixed_layer_allocation(6, 2, 3)
        assert model.automode_allocation["base_layer_indices"] == chosen
        for lid, mods in group_lora_modules_by_layer(model).items():
            i = int(lid.split("_")[1])
            for m in mods:
                assert m.get_base_layer().weight.requires_grad == (i in chosen)
                assert m.lora_A["default"].weight.requires_grad == (i not in chosen)
                assert (m.merged_adapters == ["default"]) == (i in chosen)
    model.train()
    model(input_ids=x, labels=x)["loss"].backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_unavailable_legacy_methods_raise_instead_of_substituting():
    for method in UNAVAILABLE_METHODS:
        with pytest.raises(NotImplementedError):
            apply_method(nn.Linear(2, 2), make_cfg(method))


def test_base_weight_fp32_master_path_trains_under_autocast(tmp_path):
    tok = tiny_tokenizer()
    cfg = make_cfg("static_qv", dtype="bfloat16", base_weight_dtype="float32", epochs=1)
    model = apply_method(tiny_gemma(dtype=torch.bfloat16), cfg)
    assert model.model.layers[0].self_attn.q_proj.weight.dtype == torch.float32
    assert model.model.layers[0].self_attn.k_proj.weight.dtype == torch.bfloat16
    _, stats = train_one_run(cfg, model, tiny_loader(cfg, tok), tok, run_paths(tmp_path))
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert stats["dtype_census"]["optimizer_state.exp_avg"] == {"torch.float32": n_trainable}
    assert stats["base_update_changed_fraction_mean"] > 0


# ──────────────────────────────────────────────────────────────────────────────
# 5. Training loop, select-once branch, score collection, ties
# ──────────────────────────────────────────────────────────────────────────────

def _train(method, tmp_path, **kw):
    tok = tiny_tokenizer()
    cfg = make_cfg(method, dynamic_updates=2, dynamic_threshold=50, **kw)
    torch.manual_seed(cfg.seed)
    model = apply_method(tiny_gemma(), cfg)
    paths = run_paths(tmp_path / method)
    _, stats = train_one_run(cfg, model, tiny_loader(cfg, tok), tok, paths)
    loss = json.loads((paths["logs"] / "training_loss.json").read_text())["loss_log"]
    hist = json.loads((paths["dynamic"] / "switch_history.json").read_text())
    return stats, loss, hist


def test_select_once_matches_automode_until_branch_then_holds(tmp_path):
    # 13 examples / bs2 = 7 micro-batches / accum 2 = 4 updates per epoch (the
    # last window is a tail of 1). 2 epochs -> 8 updates; interval =
    # (8 // 2 epochs) // u=2 = 2, so decisions at 2, 4, 6 and none after update 8.
    a_stats, a_loss, a_hist = _train("automode", tmp_path, epochs=2)
    s_stats, s_loss, s_hist = _train("select_once", tmp_path, epochs=2)
    assert a_stats["total_opt_steps"] == s_stats["total_opt_steps"] == 8
    assert [e["window_size"] for e in a_loss[:4]] == [2, 2, 2, 1]
    assert [h["step"] for h in a_hist["switch_history"]] == [2, 4, 6]  # no switch after update 8
    assert s_stats["branch_step"] == s_hist["branch_step"] == 2 and len(s_hist["switch_history"]) == 1
    # Identical through the second scheduled decision (the first that select_once skips).
    pre_a, pre_s = [e["loss"] for e in a_loss[:4]], [e["loss"] for e in s_loss[:4]]
    # Exact on CPU; GPU backward kernels are not guaranteed bitwise-deterministic.
    assert pre_a == (pre_s if DEVICE.type == "cpu" else pytest.approx(pre_s, rel=1e-4))
    assert a_hist["switch_history"][0]["target_modes"] == s_hist["switch_history"][0]["target_modes"]
    tl = json.loads((tmp_path / "select_once" / "dynamic" / "layer_timeline.json").read_text())["layer_timeline"]
    assert all(s["modes"] == tl[1]["modes"] for s in tl[1:])


def test_optimizer_step_scores_use_the_completed_window_once(tmp_path):
    from tests.test_switching import FakeCausalLM, wrap_with_lora

    def scores(collection):
        torch.manual_seed(0)
        model = wrap_with_lora(FakeCausalLM(n_layers=3, d=8), r=2, alpha=4)
        with torch.no_grad():
            for n, p in model.named_parameters():
                if "lora_B" in n:
                    p.normal_()
        ctrl = AutoModeController(make_cfg("automode", score_collection=collection), total_optimizer_steps=100)
        ctrl.attach(model)
        expected_sq, expected_n = {}, {}
        for window in range(2):
            for micro in range(3):
                ids = torch.randint(0, 100, (2, 5), generator=torch.Generator().manual_seed(10 * window + micro))
                (model(ids, labels=ids)["loss"] / 3).backward()
                ctrl.on_micro_step(model)
            ctrl.on_optimizer_step(model)
            for n, p in model.named_parameters():
                if p.grad is not None and p.requires_grad and ".layers." in n:
                    lid = "layer_" + n.split(".layers.")[1].split(".")[0]
                    expected_sq[lid] = expected_sq.get(lid, 0.0) + float(p.grad.pow(2).sum())
                    expected_n[lid] = expected_n.get(lid, 0) + p.numel()
            model.zero_grad(set_to_none=True)
        got = ctrl._collector.compute_scores()
        return got, {l: math.sqrt(expected_sq[l] / expected_n[l]) for l in expected_sq}

    got, expected = scores("optimizer_step")
    assert got.keys() == expected.keys()
    for l in got:
        assert got[l] == pytest.approx(expected[l], rel=1e-6)
    legacy, _ = scores("microbatch_legacy")
    assert any(abs(legacy[l] - got[l]) > 1e-3 * got[l] for l in got)  # partial windows were re-counted


def test_score_ties_are_deterministic_and_actual_counts_logged():
    from tests.test_switching import FakeCausalLM, wrap_with_lora
    cfg = make_cfg("automode", dynamic_threshold=10, dynamic_updates=10)

    def decide(scores):
        ctrl = AutoModeController(cfg, total_optimizer_steps=1000)
        ctrl.attach(wrap_with_lora(FakeCausalLM(n_layers=42, d=4), r=1, alpha=2))
        ctrl._collector.compute_scores = lambda: dict(scores)
        ctrl.maybe_switch(ctrl._model, ctrl.update_interval)
        h = ctrl.switch_history[-1]
        return h["n_selected_base"], h["n_tied_at_threshold"], {l for l, m in h["target_modes"].items() if m == "full_ft"}

    vals = list(np.linspace(1.0, 2.0, 42))
    vals[5] = vals[4]  # tie straddling the 10th-percentile position (4.1)
    scores = {f"layer_{i}": v for i, v in enumerate(vals)}
    n, ties, chosen = decide(scores)
    assert n == 38 and ties == 2  # the tie selects an extra layer; logged, not claimed as 37
    shuffled = dict(random.Random(0).sample(sorted(scores.items()), 42))
    assert decide(shuffled)[2] == chosen
    n2, ties2, _ = decide({f"layer_{i}": float(i) for i in range(42)})
    assert (n2, ties2) == (37, 0)


def test_rms_score_is_not_invariant_to_lora_gauge_rescaling():
    """A -> cA, B -> B/c leaves B@A (the function) unchanged but changes RMS
    gradient scores, so RMS normalisation alone is not representation-invariant."""
    cfg, model = _automode_model()
    x = torch.randint(3, 100, (2, 12), device=DEVICE)

    def score_and_out():
        model.zero_grad(set_to_none=True)
        model.train()
        model(input_ids=x, labels=x)["loss"].backward()
        c = GradNormCollector(); c.accumulate(model)
        return c.compute_scores()["layer_1"], logits(model, x)

    s1, out1 = score_and_out()
    with torch.no_grad():
        for m in group_lora_modules_by_layer(model)["layer_1"]:
            m.lora_A["default"].weight.mul_(10.0)
            m.lora_B["default"].weight.div_(10.0)
    s2, out2 = score_and_out()
    assert torch.allclose(out1, out2, atol=1e-4)
    assert abs(s2 / s1 - 1) > 0.5


# ──────────────────────────────────────────────────────────────────────────────
# 6. GSM8K maj@3: extraction, normalisation, vote, records, eval seeding
# ──────────────────────────────────────────────────────────────────────────────

def test_gsm8k_extraction_normalisation_and_plurality_rules():
    assert extract_gsm8k_answer("so #### 1,234.00\nQuestion: next #### 5") == ("1,234.00", "hash")
    assert normalize_number("1,234.00") == "1234"
    assert extract_gsm8k_answer("The answer is 42.") == ("42", "last_number_fallback")
    assert extract_gsm8k_answer("no digits here") == (None, "none")
    assert extract_gsm8k_answer("#### unknown") == (None, "hash_without_number")
    assert [normalize_number(s) for s in ("$18", "-3", "0.50", "-0", "abc")] == ["18", "-3", "0.5", "0", None]
    assert plurality_vote(["18", "20", None]) == "18"
    assert plurality_vote([None, None, "18"]) is None  # invalid votes count, they are not dropped
    assert plurality_vote(["20", "18", "18"]) == "18"
    assert plurality_vote([None, "18", "20"]) is None  # three-way tie -> first occurrence (invalid)


def test_gsm8k_maj3_records_and_seeding_independent_of_training_rng(tmp_path):
    tok = tiny_tokenizer()
    cfg = make_cfg("lora")
    model = apply_method(tiny_gemma(), cfg)

    def run(burn, sub):
        torch.manual_seed(burn); torch.rand(777)
        out = evaluate_gsm8k(model, tok, cfg, run_paths(tmp_path / sub), dataset=tiny_gsm8k_test(3))
        return out, json.loads((tmp_path / sub / "evals" / "gsm8k_eval.json").read_text())

    s1, e1 = run(1, "a")
    s2, e2 = run(2, "b")
    assert "gsm8k_maj3" in s1 and "gsm8k_maj1" not in s1 and s1["gsm8k_n_total"] == 3
    assert e1["metric_name"] == "gsm8k_maj3" and len(e1["records"]) == 3
    r = e1["records"][0]
    assert r["item_id"] == "gsm8k_test_0000" and r["gold"] == "0" and len(r["samples"]) == 3
    assert {"text", "extracted", "normalized", "valid", "extraction"} <= set(r["samples"][0])
    assert [x["samples"] for x in e1["records"]] == [x["samples"] for x in e2["records"]]


# ──────────────────────────────────────────────────────────────────────────────
# 7. Identity, contract, manifest, completion
# ──────────────────────────────────────────────────────────────────────────────

def test_run_id_covers_every_scientific_field_and_ignores_location():
    cfg = make_cfg("fixed_mixed")
    rid = cfg.run_id()
    assert make_cfg("fixed_mixed", max_train_samples=512).run_id() != rid  # the old collision
    for f in dataclasses.fields(cfg):
        v = getattr(cfg, f.name)
        alt = copy.copy(cfg)
        new = (not v if isinstance(v, bool) else v + 1 if isinstance(v, int) else v * 1.5 + 0.1
               if isinstance(v, float) else v + "_x" if isinstance(v, str) else v + ("x",)
               if isinstance(v, tuple) else 1 if v is None else None)
        object.__setattr__(alt, f.name, new)
        if f.name in NON_SCIENTIFIC_FIELDS:
            assert alt.run_id() == rid, f.name
        else:
            assert alt.run_id() != rid, f"{f.name} is not part of the identity"


def test_job_identity_includes_code_and_revisions():
    cfg = make_cfg()
    src = {"commit": "a" * 40, "code_sha256": "b" * 64}
    i = rebuttal.job_identity(cfg, src)
    assert rebuttal.job_identity(cfg, {**src, "commit": "c" * 40}) != i
    assert rebuttal.job_identity(cfg, {**src, "code_sha256": "d" * 64}) != i
    assert rebuttal.job_identity(make_cfg(model_revision="abc"), src) != i


def test_contract_is_json_only_and_light():
    p = subprocess.run([sys.executable, "-m", "automode.rebuttal", "contract"], cwd=REPO,
                       capture_output=True, text=True, check=True)
    lines = p.stdout.strip().splitlines()
    assert len(lines) == 1
    c = json.loads(lines[0])
    assert c == {"api_version": "1.0", "supported_methods": list(CORE_METHODS), "metric_key": "eval_gsm8k_maj3"}
    probe = "import sys, automode.rebuttal as r; r.main(['contract']); sys.exit('torch' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", probe], cwd=REPO, capture_output=True).returncode == 0


def test_supplied_manifest_resolves_to_15_distinct_runs_with_expected_queues():
    m = json.loads(MANIFEST.read_text())
    jobs = rebuttal.validate_manifest(m)
    assert len(jobs) == 15
    assert [len(rebuttal.worker_jobs(m, w)) for w in range(4)] == [4, 4, 4, 3]
    w3 = [rebuttal.job_config(m, j) for j in rebuttal.worker_jobs(m, 3)]
    assert [(c.method, c.seed, c.lora_r, c.lora_alpha) for c in w3] == [("lora", s, 32, 64) for s in (8, 25, 42)]
    cfgs = [rebuttal.job_config(m, j) for j in jobs]
    assert len({c.run_id() for c in cfgs}) == 15
    assert all(c.max_eval_samples is None and c.sampling_k == 3 for c in cfgs) and m["expected_eval_count"] == 1319
    fm = [c for c in cfgs if c.method == "fixed_mixed"]
    assert sorted(c.allocation_seed for c in fm) == [8, 25, 42] and {c.fixed_base_layers for c in fm} == {37}


@pytest.mark.parametrize("mutate,needle", [
    (lambda m: m["jobs"].append(dict(m["jobs"][0])), "duplicate"),
    (lambda m: m["common"].update(learning_rat=1e-5), "unknown common keys"),
    (lambda m: m["common"].update(max_eval_samples=100), "max_*_samples"),
    (lambda m: m["jobs"][0]["overrides"].update(method="lisa"), "not in"),
    (lambda m: m["jobs"].pop(), "must cover"),
    (lambda m: m["jobs"][1]["overrides"].update(seed=8, method="automode"), "duplicates"),
    (lambda m: m["jobs"][0]["overrides"].update(lr=1.0), "unknown override keys"),
    (lambda m: m.update(schema_version=2), "schema_version"),
])
def test_manifest_validation_rejects_changes(mutate, needle):
    m = json.loads(MANIFEST.read_text())
    mutate(m)
    with pytest.raises(rebuttal.RebuttalError, match=re_escape(needle)):
        rebuttal.validate_manifest(m)


def re_escape(s):
    import re
    return re.escape(s)


def _fake_source(tmp_path, manifest_sha, commit="a" * 40):
    meta = tmp_path / "source_metadata.json"
    meta.write_text(json.dumps({"commit": commit, "archive_sha256": "f" * 64, "manifest_sha256": manifest_sha}))
    return meta


def test_cli_rejects_mismatched_source_and_foreign_output_dirs(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOMODE_CODE_COMMIT", "a" * 40)
    _, sha = rebuttal.load_manifest(MANIFEST)
    out = tmp_path / "out"
    args = ["--manifest", str(MANIFEST), "--worker-id", "0", "--output", str(out)]
    monkeypatch.setenv("AUTOMODE_SOURCE_METADATA", str(_fake_source(tmp_path, "0" * 64)))
    assert rebuttal.main(["preflight", *args]) == 1
    assert json.loads((out / "preflight_report.json").read_text())["status"] == "failed"

    monkeypatch.setenv("AUTOMODE_SOURCE_METADATA", str(_fake_source(tmp_path, sha)))
    assert rebuttal.main(["pilot", *args]) == 1  # no preflight yet
    assert "preflight_report.json" in json.loads((out / "pilot_report.json").read_text())["error"]
    assert rebuttal.main(["run", *args, "--hours", "1"]) != 0

    monkeypatch.setenv("AUTOMODE_SOURCE_METADATA", str(_fake_source(tmp_path, sha, commit="b" * 40)))
    monkeypatch.setenv("AUTOMODE_CODE_COMMIT", "b" * 40)
    before = (out / "pilot_report.json").read_text()
    assert rebuttal.main(["pilot", *args]) == 2  # DIR belongs to commit aaaa...: refuse, write nothing
    assert (out / "pilot_report.json").read_text() == before


def test_lock_file_is_exact_complete_and_installed():
    pins = rebuttal.parse_lock()
    assert {"pytest", "transformers", "peft", "datasets", "accelerate", "safetensors", "huggingface-hub",
            "tokenizers", "numpy"} <= set(pins)
    assert not [n for n in pins if n in {"torch", "torchvision", "torchaudio", "triton"}
                or n.startswith(("nvidia-", "cuda-"))]
    assert rebuttal.verify_installed_lock() == {}


# ──────────────────────────────────────────────────────────────────────────────
# 8. End-to-end tiny jobs: completion only on verified success
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def tiny_loaders(monkeypatch, restore_signals):
    monkeypatch.setattr(rebuttal, "_load_tokenizer", lambda cfg: tiny_tokenizer())
    monkeypatch.setattr(rebuttal, "_load_base_model",
                        lambda cfg: tiny_gemma(dtype={"float32": torch.float32, "bfloat16": torch.bfloat16}[cfg.dtype]))
    monkeypatch.setattr(rebuttal, "_load_train_dataset", lambda cfg: TRAIN_EXAMPLES)
    monkeypatch.setattr(rebuttal, "_load_gsm8k_test", lambda cfg: tiny_gsm8k_test(3))


SRC = {"commit": "a" * 40, "archive_sha256": "f" * 64, "manifest_sha256": "e" * 64, "code_sha256": "c" * 64,
       "dependency_lock_sha256": "d" * 64}


def _core_spec(tmp_path, cfg, expected=3):
    job_dir = tmp_path / "core" / cfg.method
    attempt = rebuttal.next_attempt_dir(job_dir)
    return rebuttal._spec("core", cfg.method, cfg, rebuttal.job_identity(cfg, SRC), attempt, job_dir, expected, SRC)


@pytest.mark.parametrize("method", CORE_METHODS)
def test_end_to_end_tiny_core_job_with_verified_export(method, tmp_path, tiny_loaders):
    cfg = make_cfg(method, epochs=2, dynamic_updates=4, save_model=True)
    spec = _core_spec(tmp_path, cfg)
    assert rebuttal.run_single_job(spec) == 0
    job_dir = Path(spec["job_dir"])
    assert rebuttal.verify_completed(job_dir, spec["identity"], True, 3) == (True, "verified")
    res = json.loads((Path(spec["attempt_dir"]) / "result.json").read_text())
    assert res["eval_gsm8k_n_total"] == 3 and 0 <= res["eval_gsm8k_maj3"] <= 1
    assert res["export"]["status"] == "ok" and res["train"]["total_opt_steps"] == 8
    if method != "static_qv":
        adapter = Path(spec["attempt_dir"]) / "export" / "adapter" / "adapter_model.safetensors"
        adapter.rename(adapter.with_suffix(".moved"))
        assert not rebuttal.verify_completed(job_dir, spec["identity"], True, 3)[0]
        adapter.with_suffix(".moved").rename(adapter)
    diag = Path(spec["attempt_dir"]) / "dynamic" / "switch_diagnostics.json"
    assert diag.exists() == (method != "lora")
    assert not rebuttal.verify_completed(job_dir, "other-identity", True, 3)[0]
    assert not rebuttal.verify_completed(job_dir, spec["identity"], True, 1319)[0]


def test_failures_and_interruptions_never_write_complete(tmp_path, tiny_loaders, monkeypatch):
    cfg = make_cfg("automode")
    spec = _core_spec(tmp_path, cfg, expected=5)  # tiny test set has 3 items
    assert rebuttal.run_single_job(spec) == 1
    assert "gsm8k_n_total=3 != expected 5" in json.loads((Path(spec["attempt_dir"]) / "status.json").read_text())["error"]
    assert not (Path(spec["job_dir"]) / "COMPLETE.json").exists()

    def boom(*a, **k):
        raise RuntimeError("evaluation crashed")
    monkeypatch.setitem(EVALUATORS, "gsm8k", boom)
    spec = _core_spec(tmp_path, cfg)
    assert rebuttal.run_single_job(spec) == 1
    assert json.loads((Path(spec["attempt_dir"]) / "status.json").read_text())["status"] == "failed"
    assert not (Path(spec["job_dir"]) / "COMPLETE.json").exists()

    def interrupted(*a, **k):
        raise rebuttal.JobInterrupted("signal 15")
    import automode.train
    monkeypatch.setattr(automode.train, "train_one_run", interrupted)
    spec = _core_spec(tmp_path, cfg)
    assert rebuttal.run_single_job(spec) == 143
    assert json.loads((Path(spec["attempt_dir"]) / "status.json").read_text())["status"] == "interrupted"
    assert not (Path(spec["job_dir"]) / "COMPLETE.json").exists()
    assert len(list((Path(spec["job_dir"]) / "attempts").iterdir())) == 3  # failed attempts are kept


def test_queue_ledger_skip_budget_and_failure(tmp_path, tiny_loaders):
    m = json.loads(MANIFEST.read_text())
    m["expected_eval_count"] = 3
    m["common"].update(epochs=1, train_batch_size=2, grad_accum_steps=2, dtype="float32",
                       max_new_tokens=8, eval_batch_size=2)
    src = dict(SRC)
    out = tmp_path / "out"
    launched = []

    def launch(spec):
        launched.append(spec["job_id"])
        rebuttal.write_json_atomic(Path(spec["attempt_dir"]) / "spec.json", spec)
        return rebuttal.run_single_job(spec)

    est = {j["job_id"]: 0.001 for j in m["jobs"]}
    stop = rebuttal.StopState()
    assert rebuttal.execute_queue(out, 3, m, {}, src, est, 1.0, launch, stop) == 0
    ledger = json.loads((out / "run_ledger.json").read_text())
    assert [r["status"] for r in ledger["runs"]] == ["completed"] * 3 and len(launched) == 3
    assert all(r["eval_gsm8k_n_total"] == 3 and isinstance(r["eval_gsm8k_maj3"], float) for r in ledger["runs"])

    launched.clear()
    assert rebuttal.execute_queue(out, 3, m, {}, src, est, 1.0, launch, stop) == 0
    assert launched == []  # verified results are skipped, nothing re-run

    marker = out / "core" / "s25_lora_r32" / "COMPLETE.json"
    marker.write_text(json.dumps({**json.loads(marker.read_text()), "identity": "tampered"}))
    assert rebuttal.execute_queue(out, 3, m, {}, src, est, 1.0, launch, stop) == 1
    row = next(r for r in json.loads((out / "run_ledger.json").read_text())["runs"] if r["job_id"] == "s25_lora_r32")
    assert row["status"] == "failed" and "rejected" in row["error"] and launched == []

    out2 = tmp_path / "out2"
    assert rebuttal.execute_queue(out2, 3, m, {}, src, {k: 100.0 for k in est}, 1.0, launch, stop) == 1
    rows = json.loads((out2 / "run_ledger.json").read_text())["runs"]
    assert {r["status"] for r in rows} == {"budget_exhausted"} and launched == []
    assert all("eval_gsm8k_maj3" not in r for r in rows)  # no zeros for missing measurements

    def failing(spec):
        return 1
    out3 = tmp_path / "out3"
    assert rebuttal.execute_queue(out3, 3, m, {}, src, est, 1.0, failing, stop) == 1
    rows = json.loads((out3 / "run_ledger.json").read_text())["runs"]
    assert {r["status"] for r in rows} == {"failed"} and all(r["error"] and r["log_path"] for r in rows)

    def interrupt(spec):
        stop.stop_requested = True
        return 143
    out4 = tmp_path / "out4"
    assert rebuttal.execute_queue(out4, 3, m, {}, src, est, 1.0, interrupt, stop) == 143
    assert [r["status"] for r in json.loads((out4 / "run_ledger.json").read_text())["runs"]] == ["interrupted"] * 3


# ──────────────────────────────────────────────────────────────────────────────
# 9. Diagnostics and summaries
# ──────────────────────────────────────────────────────────────────────────────

def _event(step, before, target, scores=None):
    tr = [{"layer": l, "from": before[l], "to": target[l]} for l in before if before[l] != target[l]]
    return {"step": step, "modes_before": before, "target_modes": target, "layer_modes": target,
            "transitions": tr, "scores": scores or {l: 1.0 for l in before}, "n_tied_at_threshold": 0}


def test_switch_diagnostics_permanence_repromotion_and_update_weighted_occupancy():
    L, B = "lora", "full_ft"
    init = {"layer_0": L, "layer_1": L, "layer_2": L}
    m1 = {"layer_0": B, "layer_1": B, "layer_2": L}
    m2 = {"layer_0": B, "layer_1": L, "layer_2": B}
    m3 = {"layer_0": B, "layer_1": B, "layer_2": B}
    d = switch_diagnostics([_event(2, init, m1), _event(4, m1, m2), _event(6, m2, m3)], init, 10, [5] * 10)
    pl = d["per_layer"]
    assert [pl[f"layer_{i}"]["updates_in_base"] for i in range(3)] == [8, 6, 6]
    assert pl["layer_1"]["re_promotions"] == 1 and d["totals"] == {"promotions": 4, "demotions": 1, "re_promotions": 1}
    assert d["first_selection"]["layers"] == [0, 1]
    assert d["first_selected_remaining_base_rest_of_training"] == {"count": 1, "denominator": 2, "fraction": 0.5,
                                                                    "decision_events_after_first": 2}
    js = [e["jaccard_with_previous_selected"] for e in d["per_event"]]
    assert js[0] is None and js[1] == pytest.approx(1 / 3) and js[2] == pytest.approx(2 / 3)
    assert d["re_promotion"]["rate"] == 1.0 and d["optimizer_step_weighted_trainable_params"] == 5
    assert set(d["per_event"][1]["score_distribution_by_mode_before"]) == {B, L}

    z = switch_diagnostics([_event(2, init, m1)], init, 10)
    assert z["totals"]["demotions"] == 0 and z["re_promotion"]["rate"] is None and z["re_promotion"]["note"]


def test_summaries_group_exactly_and_pair_by_seed_and_item():
    rows = [{"variant": v, "train_track": "gsm8k", "protocol_version": "p", "seed": s, "m": x}
            for v, s, x in [("a", 8, .5), ("a", 25, .6), ("a", 42, .7), ("b", 42, .6), ("b", 8, .4)]]
    s = {r["variant"]: r for r in summarize_runs(rows, "m")}
    assert s["a"]["n_seeds"] == 3 and s["a"]["seed_sd"] == pytest.approx(0.1) and s["b"]["seeds"] == [8, 42]
    d = paired_seed_differences(rows, "m", "a", "b", "gsm8k", "p")
    assert d["seeds"] == [8, 42] and d["unpaired_seeds"] == [25] and d["mean_diff"] == pytest.approx(0.1)
    with pytest.raises(ValueError):
        summarize_runs(rows + [rows[0]], "m")
    t = paired_item_table([{"item_id": "x", "correct": 1}, {"item_id": "y", "correct": 0}],
                          [{"item_id": "y", "correct": 1}, {"item_id": "x", "correct": 1}])
    assert t == {"both_correct": 1, "a_only": 0, "b_only": 1, "neither": 0, "n_items": 2}
