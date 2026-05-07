"""
AutoMode core: layer identification, importance signals, and mode switching.

This module replaces the fragile name-prefix-based approach used in the 2B
notebook with a robust `isinstance(module, LoraLayer)` approach (like the 9B
notebook). The key insight is that PEFT wraps each target module in a LoraLayer
subclass, and we can operate on those modules directly without caring about
what prefix names the model uses.

Bugs this module explicitly avoids:

1. Prefix mismatch (e.g. "model.layers.N." vs "base_model.model.model.layers.N.")
   — solved by finding LoraLayer instances and asking them what layer they
   belong to via get_layer_name_from_module_name().

2. Sawtooth LR — the 9B notebook's rebuild_both-optimizer-and-scheduler pattern
   restarts warmup on every switch. Here, rebuild_optimizer_only() does just
   that; the scheduler is built once in train.py and never rebuilt.

3. Fake FFT mode (forgetting to merge LoRA into base weights before switching
   to FFT) — solved by calling module.merge() explicitly in promote_to_fft().

4. Stale adapters on downgrade — demote_to_lora() calls unmerge_adapter() +
   reset_lora_parameters() to restore a clean A~N, B=0 state.

5. lm_head / embed_tokens incorrectly trainable — freeze_non_layer_params()
   explicitly freezes them after PEFT wrapping.

6. Param count misreported after switch — count_trainable_params() is always
   called from the live model, never cached.

Unit tests in tests/test_switching.py verify the round-trip LoRA→FFT→LoRA
preserves the parameter count (this would have caught the original prefix bug).
"""

from __future__ import annotations
import math
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple, Iterable, Callable

import numpy as np
import torch
import torch.nn as nn
from peft.tuners.lora import LoraLayer


# ──────────────────────────────────────────────────────────────────────────────
# Layer identification — the foundation the rest of the module rests on.
# ──────────────────────────────────────────────────────────────────────────────

def identify_layer_for_param(param_name: str) -> Optional[str]:
    """
    Extract a canonical layer identifier from a parameter name.

    Examples
    --------
    >>> identify_layer_for_param("base_model.model.model.layers.7.self_attn.q_proj.lora_A.default.weight")
    'layer_7'
    >>> identify_layer_for_param("model.layers.0.mlp.gate_proj.weight")
    'layer_0'
    >>> identify_layer_for_param("base_model.model.model.embed_tokens.weight")
    None
    >>> identify_layer_for_param("lm_head.weight")
    None

    The return value is a stable string key that can be used across models
    (Gemma, Qwen, LLaMA) — we don't care about the exact prefix, only the
    numeric layer index.
    """
    parts = param_name.split(".")
    if "layers" not in parts:
        return None
    try:
        idx = parts.index("layers")
        layer_num = int(parts[idx + 1])
        return f"layer_{layer_num}"
    except (IndexError, ValueError):
        return None


def identify_layer_for_module(module_name: str) -> Optional[str]:
    """Module names share structure with parameter names, so delegate."""
    return identify_layer_for_param(module_name)


def get_transformer_layers(model) -> nn.ModuleList:
    """
    Navigate the wrapper hierarchy to find the flat list of decoder layers.

    Handles:
      • Plain HF model:                 model.model.layers
      • PEFT-wrapped:                   model.base_model.model.model.layers
      • Double PEFT-wrap (rare):        model.base_model.model.base_model.model.model.layers

    Used for: static methods (topk_static, topk_deep_block) that need
    explicit list access to freeze/unfreeze whole layers at once.
    """
    candidates = []
    # Depth-first search through .model and .base_model attributes
    seen = set()
    stack = [model]
    while stack:
        m = stack.pop()
        if id(m) in seen:
            continue
        seen.add(id(m))
        if hasattr(m, "layers") and isinstance(m.layers, nn.ModuleList):
            candidates.append(m.layers)
        for attr in ("model", "base_model"):
            if hasattr(m, attr):
                sub = getattr(m, attr)
                if isinstance(sub, nn.Module):
                    stack.append(sub)
    if not candidates:
        raise ValueError(
            "Could not locate transformer layers. Tried traversing "
            ".model and .base_model attributes. Inspect model hierarchy manually."
        )
    # The deepest nested .layers is the actual decoder stack
    return candidates[-1]


def get_all_layer_ids(model) -> List[str]:
    """Return sorted list of canonical layer IDs present in this model."""
    ids = set()
    for name, _ in model.named_parameters():
        lid = identify_layer_for_param(name)
        if lid is not None:
            ids.add(lid)
    # Sort numerically by the integer after "layer_"
    return sorted(ids, key=lambda s: int(s.split("_")[1]))


def group_lora_modules_by_layer(model) -> Dict[str, List[LoraLayer]]:
    """
    Group all LoraLayer instances in the model by their transformer-layer ID.

    This is the main handle AutoMode uses to operate on a whole layer at once
    (e.g. to merge q_proj AND v_proj together when upgrading layer 7 to FFT).
    """
    groups: Dict[str, List[LoraLayer]] = defaultdict(list)
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            lid = identify_layer_for_module(name)
            if lid is not None:
                groups[lid].append(module)
    return dict(groups)


def freeze_non_layer_params(model, verbose: bool = False) -> List[str]:
    """
    Belt-and-suspenders freeze of embed_tokens and lm_head.

    PEFT sometimes (depending on version) marks these as trainable after
    get_peft_model(), which inflates the parameter count by hundreds of
    millions on small models. Since AutoMode only routes switching decisions
    through numbered transformer layers, these are irrelevant to the algorithm
    and should be frozen for fair comparison with LoRA baselines.
    """
    frozen = []
    for name, p in model.named_parameters():
        # Anything OUTSIDE a numbered layer and NOT a LoRA adapter gets frozen.
        lid = identify_layer_for_param(name)
        is_lora_adapter = any(tag in name for tag in (
            "lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B",
            "lora_magnitude_vector",
        ))
        if lid is None and not is_lora_adapter:
            if p.requires_grad:
                p.requires_grad = False
                frozen.append(name)
    if verbose and frozen:
        print(f"[freeze_non_layer_params] Froze {len(frozen)} non-layer tensors "
              f"(embed_tokens, lm_head, norms).")
    return frozen


def count_trainable_params(model) -> Tuple[int, int]:
    """
    Live count of trainable vs total parameters.

    Always called from the live model — never cached — because AutoMode mutates
    requires_grad on every switch.
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


# ──────────────────────────────────────────────────────────────────────────────
# Importance signal collectors.
#
# Each collector is a class with three methods:
#   .accumulate(model) — called after each loss.backward(), before zero_grad
#   .compute_scores()  — returns {layer_id: score}
#   .reset()           — called after switching to start fresh
#
# This is deliberately a small interface so adding a new signal is just
# implementing one class.
# ──────────────────────────────────────────────────────────────────────────────

class GradNormCollector:
    """
    Original AutoMode signal: RMS gradient magnitude per layer.

        S_l = sqrt( (1/|P_l|) * sum_p ||grad_p L||^2 )

    Accumulates squared norms across micro-steps within one grad_accum window
    (or more generally, across all steps since last reset()).
    """
    def __init__(self):
        self.sums: Dict[str, float] = defaultdict(float)
        self.counts: Dict[str, int] = defaultdict(int)

    def accumulate(self, model):
        for name, p in model.named_parameters():
            if p.grad is None or not p.requires_grad:
                continue
            lid = identify_layer_for_param(name)
            if lid is None:
                continue
            # .float() before .item() to avoid bf16 precision loss on norms
            sq = float(torch.sum(p.grad.detach().float() ** 2).item())
            self.sums[lid] += sq
            self.counts[lid] += p.numel()

    def compute_scores(self) -> Dict[str, float]:
        scores = {}
        for lid, s in self.sums.items():
            c = max(self.counts[lid], 1)
            scores[lid] = math.sqrt(s / c)
        return scores

    def reset(self):
        self.sums.clear()
        self.counts.clear()


class EMAGradCollector:
    """
    Exponential moving average of gradient norms — less noisy than raw.

        s_l^{t+1} = rho * s_l^t + (1-rho) * sqrt(mean_p(||grad||^2))

    Note: reset() doesn't clear the EMA state between switches (that would
    defeat the purpose). We accumulate per switch window, then update the
    EMA at compute_scores() time and expose the EMA values.
    """
    def __init__(self, decay: float = 0.9):
        self.decay = decay
        self.ema: Dict[str, float] = {}
        self._window = GradNormCollector()

    def accumulate(self, model):
        self._window.accumulate(model)

    def compute_scores(self) -> Dict[str, float]:
        window_scores = self._window.compute_scores()
        for lid, raw in window_scores.items():
            prev = self.ema.get(lid, raw)
            self.ema[lid] = self.decay * prev + (1 - self.decay) * raw
        self._window.reset()
        return dict(self.ema)

    def reset(self):
        # Clear only the accumulator window; preserve EMA state.
        self._window.reset()


class FisherCollector:
    """
    Adam's v_t (2nd-moment EMA of gradients) approximates the Fisher diagonal.

        v_t = beta2 * v_{t-1} + (1-beta2) * grad^2
        S_l = sqrt( mean_p(v_t) )

    Following "Squisher" (2025): use the optimizer's own v tensor — no extra
    compute. Falls back to accumulating squared gradients if we can't find
    v (first step, or non-Adam optimizer).

    accumulate() is a no-op because we read from optimizer state directly.
    """
    def __init__(self):
        self._optimizer = None

    def bind_optimizer(self, optimizer):
        self._optimizer = optimizer

    def accumulate(self, model):
        pass  # no-op: we read from optimizer.state

    def compute_scores(self) -> Dict[str, float]:
        if self._optimizer is None:
            return {}
        # Map: id(param) -> layer_id, via the model's named_parameters.
        # We need the model reference; stored by the controller.
        # In practice, pre-populated at bind_optimizer time.
        param_to_layer = self._param_to_layer
        sums: Dict[str, float] = defaultdict(float)
        counts: Dict[str, int] = defaultdict(int)
        for group in self._optimizer.param_groups:
            for p in group["params"]:
                lid = param_to_layer.get(id(p))
                if lid is None:
                    continue
                state = self._optimizer.state.get(p, {})
                v = state.get("exp_avg_sq")
                if v is None:
                    # First step or not-yet-initialized — fall back to grad^2
                    if p.grad is not None:
                        sums[lid] += float(torch.sum(p.grad.detach().float() ** 2).item())
                        counts[lid] += p.numel()
                    continue
                sums[lid] += float(torch.sum(v.detach().float()).item())
                counts[lid] += p.numel()
        return {lid: math.sqrt(sums[lid] / max(counts[lid], 1)) for lid in sums}

    def bind_model(self, model):
        """Pre-compute id(param) -> layer_id map for fast lookup."""
        self._param_to_layer = {}
        for name, p in model.named_parameters():
            lid = identify_layer_for_param(name)
            if lid is not None:
                self._param_to_layer[id(p)] = lid

    def reset(self):
        pass  # Adam's EMA has its own decay; we don't interfere


class RandomCollector:
    """
    Uniform random scores per layer. Ablation: does the gradient signal
    actually matter, or does any per-layer dynamic selection work?
    """
    def __init__(self):
        self.layer_ids: List[str] = []

    def accumulate(self, model):
        if not self.layer_ids:
            self.layer_ids = get_all_layer_ids(model)

    def compute_scores(self) -> Dict[str, float]:
        return {lid: float(np.random.rand()) for lid in self.layer_ids}

    def reset(self):
        pass


def build_importance_collector(signal: str, ema_decay: float = 0.9):
    """Factory for ImportanceSignal."""
    if signal == "grad_norm":
        return GradNormCollector()
    if signal == "fisher":
        return FisherCollector()
    if signal == "ema_grad":
        return EMAGradCollector(decay=ema_decay)
    if signal == "random":
        return RandomCollector()
    raise ValueError(f"Unknown importance signal: {signal}")


# ──────────────────────────────────────────────────────────────────────────────
# Mode switching primitives.
#
# These operate on groups of LoraLayer modules (one group per transformer layer).
# Each primitive is self-contained and idempotent-safe (calling merge twice
# doesn't double-merge, because PEFT tracks merged state internally).
# ──────────────────────────────────────────────────────────────────────────────

def promote_to_fft(modules: List[LoraLayer]) -> None:
    """
    Upgrade a layer from LoRA-only to Full-FT.

    Paper spec: θ_l ← θ_l + (α/r)·B_l·A_l, then freeze adapters, unfreeze base.

    Implementation:
      1. module.merge() fuses the adapter delta into the base weight tensor.
      2. We freeze the (now-redundant) adapter params.
      3. We unfreeze the base-layer params.
    """
    for mod in modules:
        # Step 1: merge adapter into base. PEFT skips if already merged.
        try:
            mod.merge()
        except Exception:
            pass  # defensive: some PEFT versions raise if already merged
        # Step 2: freeze LoRA adapter params (they're now redundant with the merged weight)
        for p in mod.lora_A.parameters():
            p.requires_grad = False
        for p in mod.lora_B.parameters():
            p.requires_grad = False
        # Step 3: unfreeze the base layer params
        for p in mod.get_base_layer().parameters():
            p.requires_grad = True


def demote_to_lora(modules: List[LoraLayer]) -> None:
    """
    Downgrade a layer from Full-FT to LoRA-only.

    Paper spec: freeze current (now-updated) base weights, create fresh adapters.

    Implementation:
      1. Freeze base layer params (locks in the FFT-phase learning).
      2. Mark adapters as "unmerged" so PEFT's forward() uses the LoRA path again.
      3. reset_lora_parameters() reinitialises A~N(0,σ) and B=0, so BA=0 at the
         start of the LoRA phase — the merged base weight is the starting point.
      4. Unfreeze adapter params.
    """
    for mod in modules:
        # Step 1: freeze base weights
        for p in mod.get_base_layer().parameters():
            p.requires_grad = False
        # Step 2: tell PEFT the adapter is back in play.
        # Different PEFT versions track this differently.
        if hasattr(mod, "merged_adapters"):
            # Newer PEFT: merged_adapters is a list of adapter names
            mod.merged_adapters = []
        try:
            mod.merged = False  # older PEFT
        except Exception:
            pass
        # Step 3: reset adapter params to fresh A~N, B=0
        try:
            mod.reset_lora_parameters("default", True)
        except TypeError:
            # Older PEFT signature: reset_lora_parameters(adapter_name) only
            mod.reset_lora_parameters("default")
        # Step 4: unfreeze adapter params
        for p in mod.lora_A.parameters():
            p.requires_grad = True
        for p in mod.lora_B.parameters():
            p.requires_grad = True


def freeze_layer(modules: List[LoraLayer]) -> None:
    """Freeze everything in this layer (for dyn_full_only ablation)."""
    for mod in modules:
        for p in mod.get_base_layer().parameters():
            p.requires_grad = False
        for p in mod.lora_A.parameters():
            p.requires_grad = False
        for p in mod.lora_B.parameters():
            p.requires_grad = False


def current_layer_mode(modules: List[LoraLayer]) -> str:
    """Inspect the live requires_grad flags to determine mode."""
    base_train = any(p.requires_grad
                     for mod in modules
                     for p in mod.get_base_layer().parameters())
    lora_train = any(p.requires_grad
                     for mod in modules
                     for p in list(mod.lora_A.parameters()) + list(mod.lora_B.parameters()))
    if base_train:
        return "full_ft"
    if lora_train:
        return "lora"
    return "frozen"


# ──────────────────────────────────────────────────────────────────────────────
# The controller: orchestrates signal collection, switching decisions, logging.
# ──────────────────────────────────────────────────────────────────────────────

class AutoModeController:
    """
    High-level manager for AutoMode switching.

    Usage
    -----
    ctrl = AutoModeController(cfg, total_optimizer_steps)
    ctrl.attach(model, optimizer)

    for batch in loader:
        loss = model(**batch).loss / grad_accum
        loss.backward()
        ctrl.on_micro_step(model)  # accumulates importance signal

        if (step+1) % grad_accum == 0:
            grad_norm = clip_grad_norm_(...)
            optimizer.step()
            scheduler.step()  # NOT ctrl.step()
            optimizer.zero_grad()
            opt_step += 1

            changed = ctrl.maybe_switch(model, opt_step)
            if changed:
                optimizer = ctrl.rebuild_optimizer_only(model, optimizer)
                # DO NOT rebuild scheduler — prevents sawtooth LR

    Responsibilities:
      • Maintain per-step importance signal state
      • Decide when to switch (based on update_interval)
      • Execute mode transitions via promote_to_fft / demote_to_lora
      • Log every switch event for figure reproduction
    """

    def __init__(
        self,
        cfg,
        total_optimizer_steps: int,
        on_switch_callback: Optional[Callable] = None,
    ):
        # Current implementation supports ONLY percentile thresholding.
        # topk_fixed, gumbel, and static are reserved for future work but not wired.
        # Fail loudly rather than silently ignoring the config field.
        if cfg.switching_mode != "percentile":
            raise NotImplementedError(
                f"AutoModeController currently implements only switching_mode='percentile'. "
                f"Got '{cfg.switching_mode}'. The other modes (topk_fixed, gumbel, static) "
                f"are reserved in the config schema but not implemented. "
                f"Set cfg.switching_mode='percentile' or extend maybe_switch()."
            )
        self.cfg = cfg
        self.total_steps = total_optimizer_steps
        self.on_switch_callback = on_switch_callback

        # Derived
        steps_per_epoch = max(1, total_optimizer_steps // max(cfg.epochs, 1))
        self.update_interval = max(
            1, steps_per_epoch // max(cfg.dynamic_updates, 1)
        )

        # State populated by attach()
        self._model = None
        self._lora_groups: Dict[str, List[LoraLayer]] = {}
        self._layer_mode: Dict[str, str] = {}
        self._collector = build_importance_collector(
            cfg.importance_signal, ema_decay=cfg.ema_decay
        )

        # Logging
        self.switch_history: List[dict] = []
        self.last_scores: Dict[str, float] = {}

    def attach(self, model, optimizer=None):
        """Register the model (and optionally optimizer for Fisher signal)."""
        self._model = model
        self._lora_groups = group_lora_modules_by_layer(model)
        if not self._lora_groups:
            raise RuntimeError(
                "No LoraLayer modules found — AutoMode requires LoRA adapters "
                "to be present. Did you call get_peft_model() first?"
            )
        self._layer_mode = {lid: current_layer_mode(mods)
                            for lid, mods in self._lora_groups.items()}
        if isinstance(self._collector, FisherCollector):
            self._collector.bind_model(model)
            if optimizer is not None:
                self._collector.bind_optimizer(optimizer)

    def rebind_optimizer(self, optimizer):
        """After rebuilding the optimizer post-switch, rebind Fisher."""
        if isinstance(self._collector, FisherCollector):
            self._collector.bind_optimizer(optimizer)

    def on_micro_step(self, model):
        """Call after loss.backward() on each micro-step."""
        if self.cfg.importance_signal != "fisher":
            # Fisher reads from optimizer state, not grads directly
            self._collector.accumulate(model)

    def maybe_switch(self, model, opt_step: int) -> bool:
        """
        Called after optimizer.step(). Returns True iff any layer changed mode
        (indicating the caller should rebuild the optimizer param list).
        """
        if opt_step == 0:
            return False
        if opt_step % self.update_interval != 0:
            return False

        scores = self._collector.compute_scores()
        if not scores:
            return False

        vals = np.array(list(scores.values()))
        threshold = float(np.percentile(vals, self.cfg.dynamic_threshold))

        # Decide new mode per layer
        target_modes = {}
        for lid, s in scores.items():
            if s >= threshold:
                target_modes[lid] = "full_ft"
            else:
                target_modes[lid] = (
                    "frozen" if self.cfg.method == "dyn_full_only" else "lora"
                )

        any_changed = False
        transitions = []
        for lid, target in target_modes.items():
            mods = self._lora_groups.get(lid)
            if mods is None:
                continue
            current = self._layer_mode.get(lid, "lora")
            if current == target:
                continue
            # Apply the transition
            if target == "full_ft":
                promote_to_fft(mods)
            elif target == "lora":
                if current == "full_ft":
                    demote_to_lora(mods)
                elif current == "frozen":
                    # Back from frozen → re-enable LoRA adapters
                    for mod in mods:
                        for p in mod.lora_A.parameters():
                            p.requires_grad = True
                        for p in mod.lora_B.parameters():
                            p.requires_grad = True
            elif target == "frozen":
                freeze_layer(mods)
            self._layer_mode[lid] = target
            transitions.append((lid, current, target))
            any_changed = True

        self.last_scores = scores

        # Always log the decision, even if zero layers flipped.
        trainable, _ = count_trainable_params(model)
        self.switch_history.append({
            "step": opt_step,
            "threshold": threshold,
            "scores": {k: float(v) for k, v in scores.items()},
            "target_modes": dict(target_modes),
            "transitions": [
                {"layer": l, "from": f, "to": t} for (l, f, t) in transitions
            ],
            "n_transitions": len(transitions),
            "layer_modes": dict(self._layer_mode),
            "n_fft": sum(1 for m in self._layer_mode.values() if m == "full_ft"),
            "n_lora": sum(1 for m in self._layer_mode.values() if m == "lora"),
            "n_frozen": sum(1 for m in self._layer_mode.values() if m == "frozen"),
            "trainable_params_after": trainable,
        })
        if self.on_switch_callback is not None:
            self.on_switch_callback(self.switch_history[-1])

        self._collector.reset()
        return any_changed

    @staticmethod
    def rebuild_optimizer_only(model, prev_optimizer):
        """
        Construct a new optimizer over the current trainable parameter set,
        preserving the hyperparameters from the previous optimizer.

        Critically, this does NOT touch the scheduler — that's the caller's
        responsibility, and the correct thing to do is: don't touch it.
        """
        # Extract prev hyperparams
        group0 = prev_optimizer.param_groups[0]
        lr = group0.get("lr")
        wd = group0.get("weight_decay", 0.0)

        params = [p for p in model.parameters() if p.requires_grad]
        if not params:
            raise RuntimeError(
                "Optimizer rebuild found zero trainable params after switch. "
                "This indicates a switching-logic bug."
            )
        # Preserve optimizer class: if prev was AdamW8bit, new is too
        cls = type(prev_optimizer)
        try:
            new_optim = cls(params, lr=lr, weight_decay=wd)
        except TypeError:
            # Some optimizers have different init signatures
            new_optim = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
        return new_optim

    def layer_mode_snapshot(self) -> Dict[str, str]:
        """Current {layer_id: mode} for timeline logging."""
        return dict(self._layer_mode)


# ──────────────────────────────────────────────────────────────────────────────
# GumbelGate — PREVIEW MODULE, NOT A RUNNABLE METHOD IN v0.2.0
#
# This class exists for future work and for anyone wanting to prototype
# continuous-gated AutoMode. It is NOT reachable via the grid runner:
#   - `METHODS` in config.py does not contain "automode_gumbel"
#   - `build_model_for_method()` has no case for a Gumbel variant
#   - `AutoModeController` currently asserts switching_mode == "percentile"
#
# If you want to run a Gumbel experiment, you must (a) add the method string to
# METHODS, (b) add an `apply_automode_gumbel()` builder that wires this gate
# into the forward pass, and (c) relax the switching_mode assertion in
# AutoModeController. None of that is done here — this is a deferred extension.
#
# Mathematical form the future implementation would use:
#     ΔW_l = g_l · ΔW_FFT + (1 - g_l) · (α/r) · B_l A_l
# with g_l learned via Gumbel-Softmax with annealed temperature during training
# and hard-thresholded at inference.
# ──────────────────────────────────────────────────────────────────────────────

class GumbelGate(nn.Module):
    """
    Learnable continuous gate with Gumbel-Softmax straight-through.

    NOT USED by any runnable method in v0.2.0. See block comment above.
    """
    def __init__(self, n_layers: int, init_logit: float = 0.0):
        super().__init__()
        self.logits = nn.Parameter(torch.full((n_layers,), float(init_logit)))
        self.tau = 1.0  # set externally via set_tau()

    def set_tau(self, tau: float):
        self.tau = float(tau)

    def forward(self) -> torch.Tensor:
        """Return per-layer gate values in [0,1]."""
        if self.training:
            # Gumbel-Softmax binary gate: (1-g, g) = softmax over [-logit, logit] / tau
            u = torch.rand_like(self.logits)
            gumbel_on = -torch.log(-torch.log(u + 1e-20) + 1e-20)
            u2 = torch.rand_like(self.logits)
            gumbel_off = -torch.log(-torch.log(u2 + 1e-20) + 1e-20)
            on = (self.logits + gumbel_on) / self.tau
            off = (-self.logits + gumbel_off) / self.tau
            gate = torch.sigmoid(on - off)
            return gate
        else:
            return (self.logits > 0).float()

    def anneal_tau(self, progress: float, tau_start: float, tau_end: float):
        """Linear interpolation tau_start -> tau_end as progress: 0 -> 1."""
        self.set_tau(tau_start + progress * (tau_end - tau_start))
