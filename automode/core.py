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

2. Sawtooth / compounding LR — rebuild_optimizer() is the single rebuild path.
   It keeps the scheduler object (base_lrs, initial_lr, step count, lambdas)
   and only repoints scheduler.optimizer. Adam state is carried by parameter
   identity except for parameters that left the trainable set or were reset.

3. Fake FFT mode (forgetting to merge LoRA into base weights before switching
   to FFT) — promote_to_fft() merges exactly once and fails on any other state.

4. Stale adapters on downgrade — demote_to_lora() keeps the trained base weight
   (it NEVER unmerges: the base has received direct updates since the merge),
   clears PEFT's merged bookkeeping, and re-initialises the adapter to the same
   distribution PEFT uses at creation: A ~ Uniform(-1/sqrt(d_in), 1/sqrt(d_in))
   (kaiming_uniform_, a=sqrt(5)), B = 0, so B@A = 0. This is NOT Gaussian.
   Eval-mode function is preserved across promote/demote up to the precision of
   the merge: exact in fp32, but a bf16 base rounds W + delta to its grid, so part
   of the delta is lost. Training-mode behaviour is not identical, because LoRA
   dropout acts only on the adapter path, which is absent in base mode.

5. lm_head / embed_tokens incorrectly trainable — freeze_non_layer_params()
   explicitly freezes them after PEFT wrapping.

6. Param count misreported after switch — count_trainable_params() is always
   called from the live model, never cached.

Unit tests in tests/test_switching.py verify the round-trip LoRA→FFT→LoRA
preserves the parameter count (this would have caught the original prefix bug);
tests/test_rebuttal_invariants.py checks function preservation, optimizer state
and LR continuity.
"""

from __future__ import annotations
import math
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple, Iterable, Callable
import random

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
    def __init__(self, rng: Optional[np.random.Generator] = None):
        self.layer_ids: List[str] = []
        # Dedicated generator: controller randomness never touches global RNGs.
        self.rng = rng if rng is not None else np.random.default_rng(0)

    def accumulate(self, model):
        if not self.layer_ids:
            self.layer_ids = get_all_layer_ids(model)

    def compute_scores(self) -> Dict[str, float]:
        return {lid: float(self.rng.random()) for lid in self.layer_ids}

    def reset(self):
        pass


def build_importance_collector(signal: str, ema_decay: float = 0.9,
                               rng: Optional[np.random.Generator] = None):
    """Factory for ImportanceSignal."""
    if signal == "grad_norm":
        return GradNormCollector()
    if signal == "fisher":
        return FisherCollector()
    if signal == "ema_grad":
        return EMAGradCollector(decay=ema_decay)
    if signal == "random":
        return RandomCollector(rng=rng)
    raise ValueError(f"Unknown importance signal: {signal}")


# ──────────────────────────────────────────────────────────────────────────────
# Mode switching primitives.
#
# Strict by design: each primitive validates the PEFT state it expects and
# raises on anything else. No exception swallowing.
# ──────────────────────────────────────────────────────────────────────────────

ADAPTER_NAME = "default"

LORA_INIT_DESCRIPTION = (
    "A ~ Uniform(-1/sqrt(d_in), 1/sqrt(d_in)) (torch kaiming_uniform_ with a=sqrt(5), "
    "identical to PEFT 0.15.2 init_lora_weights=True); B = 0, so the initial delta "
    "B@A is exactly zero. Used both at adapter creation (by PEFT) and on demotion "
    "(by reset_adapter_zero_delta)."
)


def _check_single_default_adapter(mod: LoraLayer) -> None:
    if list(mod.active_adapters) != [ADAPTER_NAME] or list(mod.lora_A.keys()) != [ADAPTER_NAME]:
        raise RuntimeError(
            f"AutoMode expects exactly one active adapter {ADAPTER_NAME!r}; found "
            f"active={list(mod.active_adapters)}, present={list(mod.lora_A.keys())}"
        )


def _adapter_params(mod: LoraLayer) -> List[torch.nn.Parameter]:
    return list(mod.lora_A.parameters()) + list(mod.lora_B.parameters())


def promote_to_fft(modules: List[LoraLayer]) -> None:
    """
    LoRA -> base mode. θ_l ← θ_l + (α/r)·B_l·A_l exactly once (PEFT merge),
    freeze the adapter, train the base projection only.
    """
    for mod in modules:
        _check_single_default_adapter(mod)
        if mod.merged_adapters:
            raise RuntimeError(f"promote_to_fft: adapter already merged ({mod.merged_adapters})")
        if any(p.requires_grad for p in mod.get_base_layer().parameters()):
            raise RuntimeError("promote_to_fft: base weight is already trainable")
        mod.merge(safe_merge=True, adapter_names=[ADAPTER_NAME])
        if list(mod.merged_adapters) != [ADAPTER_NAME]:
            raise RuntimeError(f"promote_to_fft: merge did not register ({mod.merged_adapters})")
        for p in _adapter_params(mod):
            p.requires_grad = False
        for p in mod.get_base_layer().parameters():
            p.requires_grad = True


def reset_adapter_zero_delta(mod: LoraLayer, generator: torch.Generator) -> List[torch.nn.Parameter]:
    """Re-initialise the adapter per LORA_INIT_DESCRIPTION. Returns the reset params."""
    A = mod.lora_A[ADAPTER_NAME].weight
    B = mod.lora_B[ADAPTER_NAME].weight
    bound = 1.0 / math.sqrt(A.shape[1])
    with torch.no_grad():
        # Draw on CPU from a dedicated generator so the reset is device-independent.
        A.copy_(torch.empty(A.shape, dtype=torch.float32).uniform_(-bound, bound, generator=generator))
        B.zero_()
    if mod.lora_B[ADAPTER_NAME].bias is not None:
        raise RuntimeError("LoRA bias is not supported by AutoMode")
    return [A, B]


def demote_to_lora(modules: List[LoraLayer],
                   generator: Optional[torch.Generator] = None) -> List[torch.nn.Parameter]:
    """
    Base mode -> LoRA. Keeps the fully trained base weight (no unmerge: the base
    has had direct updates since the merge, so subtracting the old delta would be
    wrong), clears PEFT's merged bookkeeping, resets the adapter to a zero delta
    and trains the adapter only. Returns the reset parameters so the optimizer
    rebuild can discard their (same-identity) Adam state.
    """
    if generator is None:
        generator = torch.Generator().manual_seed(0)
    reset: List[torch.nn.Parameter] = []
    for mod in modules:
        _check_single_default_adapter(mod)
        if list(mod.merged_adapters) != [ADAPTER_NAME]:
            raise RuntimeError(f"demote_to_lora: expected merged adapter, got {mod.merged_adapters}")
        for p in mod.get_base_layer().parameters():
            p.requires_grad = False
        mod.merged_adapters = []  # bookkeeping only; base weight is untouched
        reset += reset_adapter_zero_delta(mod, generator)
        for p in _adapter_params(mod):
            p.requires_grad = True
    return reset


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
# Static allocations and trainable-scope invariant
# ──────────────────────────────────────────────────────────────────────────────

def fixed_layer_allocation(n_layers: int, k: int, allocation_seed: int) -> List[int]:
    """k distinct layer indices chosen by stdlib random.Random(allocation_seed).
    Independent of every torch/numpy RNG, so training seeds cannot perturb it."""
    if not 0 < k <= n_layers:
        raise ValueError(f"fixed allocation needs 0 < k <= n_layers, got k={k}, n={n_layers}")
    return sorted(random.Random(allocation_seed).sample(range(n_layers), k))


def is_target_param(name: str, target_modules: Iterable[str]) -> bool:
    return identify_layer_for_param(name) is not None and any(
        f".{t}." in name for t in target_modules)


def assert_trainable_scope(model, target_modules: Iterable[str]) -> None:
    """Only q/v (target) projections of numbered layers may be trainable —
    adapters or base weights. Embeddings, head, norms, other projections: frozen."""
    targets = tuple(target_modules)
    bad = [n for n, p in model.named_parameters() if p.requires_grad and not is_target_param(n, targets)]
    if bad:
        raise RuntimeError(f"Non-target parameters are trainable: {bad[:5]} (+{max(0, len(bad) - 5)})")
    if not any(p.requires_grad for p in model.parameters()):
        raise RuntimeError("No trainable parameters")


# ──────────────────────────────────────────────────────────────────────────────
# Optimizer rebuild — the ONLY rebuild path (train.py uses this too).
# ──────────────────────────────────────────────────────────────────────────────

def rebuild_optimizer(model, old_optimizer, scheduler=None,
                      reset_params: Iterable[torch.nn.Parameter] = ()):
    """
    New optimizer over the current trainable set.

    * Every param-group hyperparameter (lr, initial_lr, betas, eps, weight_decay,
      amsgrad, foreach, ...) and optimizer.defaults are copied verbatim.
    * Adam state is carried by parameter identity for parameters that stay
      trainable and are not in reset_params. Parameters that left the trainable
      set lose their state; newly trainable or reset parameters start fresh.
      reset_params is needed because an in-place reset keeps tensor identity.
    * The scheduler keeps its base_lrs, last_epoch and lambdas; only its
      optimizer reference is repointed. (The old helper overwrote base_lrs with
      the current decayed LR, compounding the decay at every switch.)
    * All .grad tensors are cleared.
    """
    if len(old_optimizer.param_groups) != 1:
        raise NotImplementedError("rebuild_optimizer supports exactly one param group")
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("Optimizer rebuild found zero trainable params after switch.")
    for p in model.parameters():
        p.grad = None
    hyper = {k: v for k, v in old_optimizer.param_groups[0].items() if k != "params"}
    new_optimizer = type(old_optimizer)([{"params": params, **hyper}])
    new_optimizer.defaults = dict(old_optimizer.defaults)
    reset_ids = {id(p) for p in reset_params}
    for p in params:
        if id(p) not in reset_ids and p in old_optimizer.state and old_optimizer.state[p]:
            new_optimizer.state[p] = old_optimizer.state[p]
    if scheduler is not None:
        scheduler.optimizer = new_optimizer
    return new_optimizer


# ──────────────────────────────────────────────────────────────────────────────
# The controller: orchestrates signal collection, switching decisions, logging.
# ──────────────────────────────────────────────────────────────────────────────

class AutoModeController:
    """
    Dynamic LoRA <-> base switching for `automode`, and the `select_once`
    control (identical until its first scheduled selection, then frozen).

    Loop contract (train.py):
      loss.backward()                      -> ctrl.on_micro_step(model)
      tail rescale, then BEFORE clipping   -> ctrl.on_optimizer_step(model)
      optimizer.step(); scheduler.step(); opt_step += 1
      if ctrl.maybe_switch(model, opt_step):
          optimizer = rebuild_optimizer(model, optimizer, scheduler, ctrl.last_reset_params)

    Selection rule (submitted lower-tail percentile): threshold =
    np.percentile(scores, t); layers with score >= threshold go to base mode.
    Ties at the threshold are all selected, so the selected count can exceed
    n - floor(t% of n); it is logged per event, never assumed.
    """

    def __init__(
        self,
        cfg,
        total_optimizer_steps: int,
        on_switch_callback: Optional[Callable] = None,
    ):
        if cfg.switching_mode != "percentile":
            raise NotImplementedError(
                f"AutoModeController currently implements only switching_mode='percentile'. "
                f"Got '{cfg.switching_mode}'. The other modes (topk_fixed, gumbel, static) "
                f"are reserved in the config schema but not implemented."
            )
        if cfg.method not in ("automode", "select_once"):
            raise ValueError(f"AutoModeController does not implement method {cfg.method!r}")
        self.cfg = cfg
        self.total_steps = total_optimizer_steps
        self.on_switch_callback = on_switch_callback
        self.select_once = cfg.method == "select_once"
        self.branch_step: Optional[int] = None

        steps_per_epoch = max(1, total_optimizer_steps // max(cfg.epochs, 1))
        self.update_interval = max(1, steps_per_epoch // max(cfg.dynamic_updates, 1))

        self._model = None
        self._lora_groups: Dict[str, List[LoraLayer]] = {}
        self._layer_mode: Dict[str, str] = {}
        # Separate generators: controller scores (random signal) and adapter resets.
        self._collector = build_importance_collector(
            cfg.importance_signal, ema_decay=cfg.ema_decay,
            rng=np.random.default_rng([cfg.seed, 0xC0]),
        )
        self._reset_generator = torch.Generator().manual_seed(cfg.seed * 1_000_003 + 0xAD)
        self.last_reset_params: List[torch.nn.Parameter] = []

        self.switch_history: List[dict] = []
        self.last_scores: Dict[str, float] = {}

    def attach(self, model, optimizer=None):
        """Register the model (and optionally optimizer for Fisher signal)."""
        self._model = model
        groups = group_lora_modules_by_layer(model)
        if not groups:
            raise RuntimeError(
                "No LoraLayer modules found — AutoMode requires LoRA adapters "
                "to be present. Did you call get_peft_model() first?"
            )
        self._lora_groups = dict(sorted(groups.items(), key=lambda kv: int(kv[0].split("_")[1])))
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

    @property
    def _collects(self) -> bool:
        return not (self.select_once and self.branch_step is not None) and \
            self.cfg.importance_signal != "fisher"

    def on_micro_step(self, model):
        """After each loss.backward(). Only the legacy protocol samples here."""
        if self.cfg.score_collection == "microbatch_legacy" and self._collects:
            self._collector.accumulate(model)

    def on_optimizer_step(self, model):
        """Once per optimizer update, on the completed (tail-rescaled) window
        gradient, before clipping. Default protocol 'optimizer_step'."""
        if self.cfg.score_collection == "optimizer_step" and self._collects:
            self._collector.accumulate(model)

    def maybe_switch(self, model, opt_step: int) -> bool:
        """
        Called after optimizer.step(). Returns True iff any layer changed mode.
        Never switches after the final update (nothing would train on it).
        """
        self.last_reset_params = []
        if opt_step <= 0 or opt_step % self.update_interval != 0:
            return False
        if opt_step >= self.total_steps:
            return False
        if self.select_once and self.branch_step is not None:
            return False

        scores = self._collector.compute_scores()
        if set(scores) != set(self._lora_groups):
            raise RuntimeError(
                f"Scores missing for layers {sorted(set(self._lora_groups) - set(scores))}; "
                f"was the collector hook called for this window?")
        order = list(self._lora_groups)  # numeric layer order
        vals = np.array([scores[l] for l in order], dtype=np.float64)
        if not np.all(np.isfinite(vals)):
            raise RuntimeError(f"Non-finite importance scores at step {opt_step}")
        threshold = float(np.percentile(vals, self.cfg.dynamic_threshold))
        target_modes = {l: ("full_ft" if scores[l] >= threshold else "lora") for l in order}
        modes_before = dict(self._layer_mode)

        transitions = []
        for lid in order:
            current, target = self._layer_mode[lid], target_modes[lid]
            if current == target:
                continue
            if target == "full_ft":
                promote_to_fft(self._lora_groups[lid])
            else:
                self.last_reset_params += demote_to_lora(self._lora_groups[lid], self._reset_generator)
            self._layer_mode[lid] = target
            transitions.append({"layer": lid, "from": current, "to": target})

        if self.select_once:
            self.branch_step = opt_step
        self.last_scores = scores

        trainable, _ = count_trainable_params(model)
        n_selected = sum(1 for m in target_modes.values() if m == "full_ft")
        self.switch_history.append({
            "step": opt_step,
            "percentile_t": self.cfg.dynamic_threshold,
            "threshold": threshold,
            "score_collection": self.cfg.score_collection,
            "scores": {k: float(scores[k]) for k in order},
            "modes_before": modes_before,
            "target_modes": dict(target_modes),
            "n_selected_base": n_selected,
            "n_tied_at_threshold": int(np.sum(vals == threshold)),
            "transitions": transitions,
            "n_transitions": len(transitions),
            "layer_modes": dict(self._layer_mode),
            "n_fft": sum(1 for m in self._layer_mode.values() if m == "full_ft"),
            "n_lora": sum(1 for m in self._layer_mode.values() if m == "lora"),
            "trainable_params_after": trainable,
            "select_once_branch": self.select_once,
        })
        if self.on_switch_callback is not None:
            self.on_switch_callback(self.switch_history[-1])

        self._collector.reset()
        return bool(transitions)

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
