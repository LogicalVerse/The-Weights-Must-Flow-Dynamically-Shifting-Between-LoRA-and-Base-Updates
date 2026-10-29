"""
Unified training loop for AutoMode and all baselines.

Design principles
-----------------
1. ONE train_one_run() function handles every method. Method-specific logic
   is isolated in build_model_for_method() (from models.py) and applied via
   the controller pattern (AutoMode) or via static requires_grad flags (rest).

2. Scheduler is built ONCE and NEVER rebuilt. On a switch, core.rebuild_optimizer
   (the single rebuild path) builds a new optimizer over the new trainable set,
   carries Adam state by parameter identity, and repoints scheduler.optimizer
   without touching base_lrs / initial_lr / step count.

3. Every run writes the same canonical set of logs to disk:
     configs/run_config.json       — exact hyperparameters
     logs/training_loss.json        — per-step loss (unscaled)
     logs/lr_schedule.json          — per-step LR (proves no sawtooth)
     logs/grad_norms.json           — per-step total grad norm before clip
     logs/trainable_params.json     — per-step trainable count (Figure 1)
     dynamic/switch_history.json    — AutoMode only: every switch event
     dynamic/layer_timeline.json    — per-step {layer_id: mode}
     evals/<benchmark>_eval.json    — full predictions + correctness
     dynamic/switch_diagnostics.json — permanence / re-promotion / occupancy summary
     export/                        — verified FULL inference export (cfg.save_model);
                                      written after all evaluations, see automode.export

4. Resume-safe: a run is skipped only if its COMPLETE marker records the same
   run_id and config. Failures never write COMPLETE.

5. Peak VRAM measured and reported, so the paper table has compute numbers
   alongside accuracy.
"""

from __future__ import annotations
import contextlib
import gc
import json
import math
import os
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from automode.config import RunConfig, CORE_METHODS, UNAVAILABLE_METHODS
from automode.core import (
    AutoModeController,
    assert_trainable_scope,
    count_trainable_params,
    freeze_non_layer_params,
    group_lora_modules_by_layer,
    identify_layer_for_param,
    current_layer_mode,
    get_transformer_layers,
    rebuild_optimizer,
)


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16":  torch.float16,
    "float32":  torch.float32,
}


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dirs(cfg: RunConfig) -> Dict[str, Path]:
    run_id = cfg.run_id()
    root = Path(cfg.output_root) / run_id
    paths = {
        "root": Path(cfg.output_root),
        "run": root,
        "configs": root / "configs",
        "logs": root / "logs",
        "dynamic": root / "dynamic",
        "evals": root / "evals",
        "checkpoints": root / "checkpoints",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def save_json(obj, path: Path):
    """Atomic write: readers never observe a partial file."""
    path = Path(path)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def resolve_device(cfg: RunConfig) -> torch.device:
    return torch.device(cfg.device if (torch.cuda.is_available() or not cfg.device.startswith("cuda")) else "cpu")


def autocast_ctx(cfg: RunConfig, device: torch.device):
    """bf16 autocast only when fp32 master base weights are in use."""
    if cfg.base_weight_dtype == "float32" and DTYPE_MAP[cfg.dtype] != torch.float32:
        return torch.autocast(device_type=device.type, dtype=DTYPE_MAP[cfg.dtype])
    return contextlib.nullcontext()


def cuda_sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def is_run_complete(cfg: RunConfig) -> bool:
    """True only for a COMPLETE marker written for this exact run_id and config."""
    marker = Path(cfg.output_root) / cfg.run_id() / "COMPLETE"
    if not marker.exists():
        return False
    try:
        done = load_json(marker)
    except (OSError, json.JSONDecodeError):
        return False
    return (done.get("status") == "ok" and done.get("run_id") == cfg.run_id()
            and done.get("scientific_config") == json.loads(json.dumps(cfg.scientific_dict(), default=str)))


def mark_run_complete(cfg: RunConfig, summary: dict):
    paths = ensure_dirs(cfg)
    save_json(summary, paths["run"] / "COMPLETE")


# ──────────────────────────────────────────────────────────────────────────────
# Optimizer & scheduler construction
# ──────────────────────────────────────────────────────────────────────────────

def build_optimizer(model, cfg: RunConfig):
    """
    AdamW for every method. Full-FT may request bitsandbytes AdamW8bit; if it is
    requested and unavailable we raise instead of silently changing optimizer.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError(
            "Model has zero trainable parameters — check requires_grad flags."
        )

    if cfg.method == "full_ft" and cfg.use_8bit_optim_for_full_ft:
        import bitsandbytes as bnb  # ImportError propagates: no silent fallback
        return bnb.optim.AdamW8bit(
            params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
        )
    return torch.optim.AdamW(
        params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
    )


def build_scheduler(optimizer, total_steps: int, cfg: RunConfig):
    """Linear or cosine schedule with warmup. Built ONCE, never rebuilt."""
    from transformers import get_scheduler
    warmup = max(1, int(cfg.warmup_ratio * total_steps))
    return get_scheduler(
        name=cfg.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=warmup, num_training_steps=total_steps,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Gradient norm (for logging)
# ──────────────────────────────────────────────────────────────────────────────

def compute_total_grad_norm(model) -> float:
    """L2 norm across all parameters' gradients. Called BEFORE clip."""
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += float(p.grad.detach().float().norm(2).item()) ** 2
    return math.sqrt(total)


# ──────────────────────────────────────────────────────────────────────────────
# LISA-style random layer sampling (baseline, not AutoMode)
# ──────────────────────────────────────────────────────────────────────────────

class LisaController:
    """
    UNAVAILABLE (see config.UNAVAILABLE_METHODS): its optimizer attach path
    zeroed scheduler base_lrs during warmup, so earlier LISA scores are not
    trustworthy. Kept for reference; train_one_run refuses to run it.

    LISA baseline (Pan et al. 2024, arXiv:2403.17919v4).

    Paper Algorithm 1: every K iterations, freeze all layers EXCEPT:
      - embedding layer (always trainable)
      - language-modeling head (always trainable)
      - γ randomly-sampled intermediate transformer layers
    This is the "E+H+γL" recipe the paper argues is essential.

    Our earlier implementation sampled intermediate layers but only kept lm_head
    trainable (not embed_tokens). That's a silent deviation from the paper;
    this controller now always unfreezes embed_tokens + lm_head.
    """
    def __init__(self, cfg: RunConfig, total_steps: int):
        self.cfg = cfg
        self.total_steps = total_steps
        self.update_interval = max(1, cfg.lisa_period)
        self._model = None
        self._rng = np.random.RandomState(cfg.seed)
        self.switch_history: List[dict] = []
        self._all_layers: List[int] = []

    def attach(self, model, optimizer=None):
        self._model = model
        layers = get_transformer_layers(model)
        self._all_layers = list(range(len(layers)))
        # Start with random sample active
        self._apply_random_selection(0)

    def rebind_optimizer(self, optimizer):
        pass  # LISA doesn't use optimizer state

    def on_micro_step(self, model):
        pass  # no signal collection

    def maybe_switch(self, model, opt_step: int) -> bool:
        if opt_step == 0 or opt_step % self.update_interval != 0:
            return False
        self._apply_random_selection(opt_step)
        return True

    def _keep_embed_and_lm_head_trainable(self):
        """
        LISA convention (paper §3.2): embedding + LM head are ALWAYS trainable.
        Search for embed_tokens across common model hierarchies.
        """
        # lm_head — top-level attribute on most HF causal LMs
        if hasattr(self._model, "lm_head"):
            for p in self._model.lm_head.parameters():
                p.requires_grad = True

        # embed_tokens — nested under .model (HF convention)
        # Try both model.embed_tokens and model.model.embed_tokens paths
        embed = None
        if hasattr(self._model, "embed_tokens"):
            embed = self._model.embed_tokens
        elif hasattr(self._model, "model") and hasattr(self._model.model, "embed_tokens"):
            embed = self._model.model.embed_tokens
        elif (hasattr(self._model, "base_model")
              and hasattr(self._model.base_model, "model")
              and hasattr(self._model.base_model.model, "embed_tokens")):
            embed = self._model.base_model.model.embed_tokens
        if embed is not None:
            for p in embed.parameters():
                p.requires_grad = True

    def _apply_random_selection(self, step: int):
        layers = get_transformer_layers(self._model)
        k = min(self.cfg.lisa_num_layers, len(self._all_layers))
        selected = self._rng.choice(self._all_layers, size=k, replace=False)
        selected = set(selected.tolist())

        # Freeze everything first (including embed/head), then re-enable.
        for p in self._model.parameters():
            p.requires_grad = False

        # Unfreeze the sampled intermediate layers
        for i, layer in enumerate(layers):
            if i in selected:
                for p in layer.parameters():
                    p.requires_grad = True

        # LISA's E+H convention — always trainable
        self._keep_embed_and_lm_head_trainable()

        self.switch_history.append({
            "step": step, "selected_layers": sorted(selected),
            "always_trainable": ["embed_tokens", "lm_head"],
        })

    def layer_mode_snapshot(self) -> Dict[str, str]:
        layers = get_transformer_layers(self._model)
        out = {}
        for i, layer in enumerate(layers):
            active = any(p.requires_grad for p in layer.parameters())
            out[f"layer_{i}"] = "full_ft" if active else "frozen"
        return out


# ──────────────────────────────────────────────────────────────────────────────
# AdaGradSelect-style baseline (from 2512_15764v1.pdf)
# ──────────────────────────────────────────────────────────────────────────────

class AdaGradSelectController:
    """
    UNAVAILABLE (see config.UNAVAILABLE_METHODS) — same broken attach path as
    LisaController. Kept for reference; train_one_run refuses to run it.

    AdaGradSelect baseline (Kumar et al. 2025, arXiv:2512.15764v1).

    Paper Algorithm 2:
      During epoch 1:
        - With prob ε:   EXPLORATION = top-k% blocks by cumulative gradient norm
        - With prob 1-ε: EXPLOITATION = sample k% blocks from Dirichlet(α = f + δ)
                         where f is the update-frequency count vector (prior)
        - ε decays exponentially: ε_t = ε_0 * exp(-λ * t)
      From epoch 2 onward: pure exploitation (Dirichlet sampling).

    A "block" in the paper includes transformer blocks plus embed_tokens and
    the final norm — we approximate this with transformer layers only (we update
    lm_head separately, mirroring LISA's convention for a consistent baseline).

    CHANGED from v0.1.0: Previous implementation had the explore/exploit branches
    swapped (it was doing Dirichlet on "explore" and top-k on "exploit"). Fixed.
    """
    def __init__(self, cfg: RunConfig, total_steps: int):
        self.cfg = cfg
        self.total_steps = total_steps
        # ~5 switches per epoch, matching the paper's small-ε-greedy cadence
        self.update_interval = max(1, total_steps // max(cfg.epochs * 5, 1))
        self._model = None
        self._rng = np.random.RandomState(cfg.seed)

        # Cumulative sum-of-squared-grads and param counts per layer for this window
        self._grad_accum: Dict[int, float] = {}
        self._counts: Dict[int, int] = {}

        # Historical update-frequency counts — feeds the Dirichlet α vector
        self._update_frequency: Dict[int, int] = {}

        # Exponential-decay ε parameters (paper §3.2)
        self._eps0 = float(cfg.adagrad_epsilon)
        # decay rate chosen so ε drops by ~20x over one epoch's worth of switches.
        # switches_per_epoch ≈ total_steps / (epochs * update_interval) ~ 5
        steps_per_epoch_switches = max(1, total_steps // max(cfg.epochs, 1) // self.update_interval)
        self._eps_lambda = math.log(20.0) / max(steps_per_epoch_switches, 1)

        self.switch_history: List[dict] = []
        self._all_layers: List[int] = []
        self._switch_counter = 0  # counts AdaGradSelect-driven switches (not opt_steps)

    def attach(self, model, optimizer=None):
        self._model = model
        layers = get_transformer_layers(model)
        self._all_layers = list(range(len(layers)))
        # Warm start: a random selection so we have trainable params from step 0.
        # This matches epoch-1 behavior when no frequency prior exists yet.
        self._select_and_apply(0, force_explore=True)

    def rebind_optimizer(self, optimizer):
        pass

    def on_micro_step(self, model):
        """Accumulate per-layer squared gradient norms (for exploration scoring)."""
        layers = get_transformer_layers(model)
        for i, layer in enumerate(layers):
            sq = 0.0
            cnt = 0
            for p in layer.parameters():
                if p.grad is not None and p.requires_grad:
                    sq += float(p.grad.detach().float().norm(2).item() ** 2)
                    cnt += p.numel()
            if cnt > 0:
                self._grad_accum[i] = self._grad_accum.get(i, 0.0) + sq
                self._counts[i] = self._counts.get(i, 0) + cnt

    def _current_epsilon(self) -> float:
        """ε_t = ε_0 * exp(-λ * switch_index)."""
        return self._eps0 * math.exp(-self._eps_lambda * self._switch_counter)

    def maybe_switch(self, model, opt_step: int) -> bool:
        if opt_step == 0 or opt_step % self.update_interval != 0:
            return False

        # Epoch determination: paper restricts exploration to epoch 1 only
        steps_per_epoch = max(1, self.total_steps // max(self.cfg.epochs, 1))
        current_epoch = (opt_step - 1) // steps_per_epoch + 1  # 1-indexed

        if current_epoch >= 2:
            # Epoch 2+: pure exploitation (Dirichlet sampling from historical freq)
            self._select_and_apply(opt_step, force_explore=False)
        else:
            # Epoch 1: ε-greedy with exponentially decaying ε
            eps = self._current_epsilon()
            explore = (self._rng.rand() < eps)
            self._select_and_apply(opt_step, force_explore=explore)

        self._switch_counter += 1
        self._grad_accum.clear()
        self._counts.clear()
        return True

    def _select_and_apply(self, step: int, force_explore: bool):
        """
        Apply the paper's two branches:
          force_explore=True  → top-k% by RMS grad norm (needs accumulated grads)
          force_explore=False → Dirichlet(α = frequency + δ) sampling

        If force_explore=True but no gradients have been accumulated yet (step 0),
        fall back to uniform random selection.
        """
        layers = get_transformer_layers(self._model)
        n_layers = len(self._all_layers)
        k = max(1, n_layers * self.cfg.adagrad_pct // 100)

        mode = None
        if force_explore:
            if self._grad_accum:
                # EXPLORATION: top-k by RMS gradient norm (paper's "greedy" branch)
                scores = {
                    i: math.sqrt(self._grad_accum[i] / max(self._counts[i], 1))
                    for i in self._grad_accum
                }
                ranked = sorted(scores.items(), key=lambda x: -x[1])
                selected = set(i for i, _ in ranked[:k])
                mode = "explore_topk_grad"
            else:
                # Cold start: uniform random
                selected = set(self._rng.choice(
                    self._all_layers, size=k, replace=False,
                ).tolist())
                mode = "explore_cold_uniform"
        else:
            # EXPLOITATION: Dirichlet(α = f + δ) where f is historical frequency
            f_vec = np.array([
                self._update_frequency.get(i, 0) for i in self._all_layers
            ], dtype=np.float64)
            alpha = f_vec + self.cfg.adagrad_dirichlet_alpha
            probs = self._rng.dirichlet(alpha)
            # Sample k without replacement proportional to probs
            selected = set(self._rng.choice(
                self._all_layers, size=k, replace=False, p=probs,
            ).tolist())
            mode = "exploit_dirichlet"

        # Apply the selection: freeze everything, unfreeze selected layers + lm_head
        for p in self._model.parameters():
            p.requires_grad = False
        for i, layer in enumerate(layers):
            if i in selected:
                for p in layer.parameters():
                    p.requires_grad = True
                self._update_frequency[i] = self._update_frequency.get(i, 0) + 1
        # Keep lm_head trainable (standard across selective-update baselines)
        if hasattr(self._model, "lm_head"):
            for p in self._model.lm_head.parameters():
                p.requires_grad = True

        self.switch_history.append({
            "step": step, "mode": mode,
            "epsilon_at_switch": self._current_epsilon(),
            "selected_layers": sorted(selected),
        })

    def layer_mode_snapshot(self) -> Dict[str, str]:
        layers = get_transformer_layers(self._model)
        return {
            f"layer_{i}": ("full_ft" if any(p.requires_grad for p in l.parameters())
                           else "frozen")
            for i, l in enumerate(layers)
        }


# ──────────────────────────────────────────────────────────────────────────────
# The main training function
# ──────────────────────────────────────────────────────────────────────────────

def _base_trainable_params(model) -> List[torch.nn.Parameter]:
    """Trainable non-adapter weights (base projections in base mode)."""
    return [p for n, p in model.named_parameters() if p.requires_grad and "lora_" not in n]


def _dtype_census(model, optimizer) -> Dict[str, Dict[str, int]]:
    """numel per dtype for trainable params and for each optimizer state tensor
    (the training loop keeps the max seen over the run, i.e. across modes)."""
    out: Dict[str, Dict[str, int]] = {"trainable_params": {}}
    for p in model.parameters():
        if p.requires_grad:
            d = out["trainable_params"]
            d[str(p.dtype)] = d.get(str(p.dtype), 0) + p.numel()
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v) and v.dim() > 0:
                d = out.setdefault(f"optimizer_state.{k}", {})
                d[str(v.dtype)] = d.get(str(v.dtype), 0) + v.numel()
    return out


def train_one_run(
    cfg: RunConfig,
    model,
    train_loader: DataLoader,
    tokenizer,
    paths: Dict[str, Path],
    progress_every: int = 25,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Train `model` under cfg.method. Returns (model, stats). Does NOT save a
    checkpoint: the verified full export runs after evaluation (automode.export).

    Accumulation: the loss of every micro-batch is divided by grad_accum_steps;
    the last window of an epoch may be partial and its gradient is rescaled by
    grad_accum_steps / window so it is a mean over the window. Importance scores
    (score_collection='optimizer_step') are sampled once per update, after that
    rescale and before clipping. Switches happen only after optimizer.step(),
    never after the final update.
    """
    if cfg.method in UNAVAILABLE_METHODS:
        raise NotImplementedError(f"{cfg.method} is unavailable: {UNAVAILABLE_METHODS[cfg.method]}")
    from automode.analysis import switch_diagnostics

    device = resolve_device(cfg)
    micro_per_epoch = len(train_loader)
    opt_per_epoch = math.ceil(micro_per_epoch / cfg.grad_accum_steps)
    total_opt_steps = opt_per_epoch * cfg.epochs
    if total_opt_steps == 0:
        raise ValueError(
            f"Computed total_opt_steps=0 (micro_per_epoch={micro_per_epoch}, "
            f"grad_accum={cfg.grad_accum_steps}, epochs={cfg.epochs})."
        )
    if cfg.method in CORE_METHODS:
        assert_trainable_scope(model, cfg.lora_target_modules)

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, total_opt_steps, cfg)

    controller = None
    if cfg.method in ("automode", "select_once"):
        controller = AutoModeController(cfg, total_opt_steps)
        controller.attach(model, optimizer)
        initial_modes = controller.layer_mode_snapshot()
    else:
        initial_modes = getattr(model, "automode_layer_modes", None)

    loss_log: List[dict] = []
    lr_log: List[dict] = []
    grad_norm_log: List[dict] = []
    param_log: List[dict] = []
    probe_log: List[dict] = []
    layer_timeline: List[dict] = []
    trainable_0, total_params = count_trainable_params(model)
    param_log.append({"step": 0, "trainable_params": trainable_0, "total_params": total_params})
    if initial_modes is not None:
        layer_timeline.append({"step": 0, "modes": dict(initial_modes)})
    dtype_census: Dict[str, Dict[str, int]] = {}

    def flush_logs():
        save_json({"loss_log": loss_log}, paths["logs"] / "training_loss.json")
        save_json({"lr_log": lr_log}, paths["logs"] / "lr_schedule.json")
        save_json({"grad_norm_log": grad_norm_log}, paths["logs"] / "grad_norms.json")
        save_json({"param_log": param_log}, paths["logs"] / "trainable_params.json")
        save_json({"base_update_probe": probe_log}, paths["logs"] / "base_update_probe.json")
        if controller is not None:
            save_json({"switch_history": controller.switch_history,
                       "update_interval": controller.update_interval,
                       "branch_step": controller.branch_step},
                      paths["dynamic"] / "switch_history.json")
        if layer_timeline:
            save_json({"layer_timeline": layer_timeline}, paths["dynamic"] / "layer_timeline.json")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    cuda_sync(device)
    start_time = time.perf_counter()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    opt_step = 0
    switch_overhead = 0.0
    first_promotion_step: Optional[int] = None
    trainable_per_update: List[int] = []

    for epoch in range(cfg.epochs):
        epoch_start = time.perf_counter()
        n_batches = len(train_loader)
        window_losses: List[float] = []
        t_window = time.perf_counter()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs} [{cfg.method}]", leave=False)
        for micro_step, batch in enumerate(pbar):
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            with autocast_ctx(cfg, device):
                loss = model(**batch)["loss"]
            if not torch.isfinite(loss.detach()):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch+1}, micro-step {micro_step}")
            (loss / cfg.grad_accum_steps).backward()
            window_losses.append(float(loss.detach().item()))
            if controller is not None:
                controller.on_micro_step(model)

            if len(window_losses) < cfg.grad_accum_steps and micro_step + 1 < n_batches:
                continue

            # ── one optimizer update over the completed (possibly tail) window ──
            window = len(window_losses)
            if window != cfg.grad_accum_steps:
                scale = cfg.grad_accum_steps / window
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.mul_(scale)
            if controller is not None:
                controller.on_optimizer_step(model)
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            trainable_per_update.append(sum(p.numel() for p in trainable_params))
            # Base-update efficacy probe (every update): fraction of trainable base
            # weight elements the optimizer step actually changes. With pure-bf16
            # base weights most Adam updates at small LR round to no change.
            probe = _base_trainable_params(model)
            snapshots = [p.detach().clone() for p in probe]
            lr_used = optimizer.param_groups[0]["lr"]
            grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable_params, cfg.max_grad_norm))
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            opt_step += 1
            if probe:
                changed = sum(int((p.detach() != s).sum()) for p, s in zip(probe, snapshots))
                total = sum(p.numel() for p in probe)
                probe_log.append({"step": opt_step, "lr": lr_used, "n_elements": total,
                                  "changed_fraction": changed / total})
                del snapshots
            for key, by_dtype in _dtype_census(model, optimizer).items():  # max numel seen per dtype
                for dt, n in by_dtype.items():
                    dtype_census.setdefault(key, {})[dt] = max(dtype_census.get(key, {}).get(dt, 0), n)
            cuda_sync(device)
            now = time.perf_counter()
            loss_log.append({"step": opt_step, "epoch": epoch + 1,
                             "loss": round(sum(window_losses) / window, 6),
                             "window_size": window, "step_sec": now - t_window})
            lr_log.append({"step": opt_step, "lr": float(lr_used)})
            grad_norm_log.append({"step": opt_step, "grad_norm": round(grad_norm, 6)})
            param_log.append({"step": opt_step, "epoch": epoch + 1,
                              "trainable_params": trainable_per_update[-1]})
            window_losses = []

            if controller is not None:
                n_events = len(controller.switch_history)
                if controller.maybe_switch(model, opt_step):
                    optimizer = rebuild_optimizer(model, optimizer, scheduler, controller.last_reset_params)
                    controller.rebind_optimizer(optimizer)
                    assert_trainable_scope(model, cfg.lora_target_modules)
                    layer_timeline.append({"step": opt_step, "modes": controller.layer_mode_snapshot()})
                if len(controller.switch_history) > n_events:
                    cuda_sync(device)
                    switch_overhead += time.perf_counter() - now
                    if first_promotion_step is None and controller.switch_history[-1]["n_fft"] > 0:
                        first_promotion_step = opt_step
                    flush_logs()
            if opt_step % progress_every == 0:
                flush_logs()
            pbar.set_postfix({"loss": loss_log[-1]["loss"], "lr": f"{lr_used:.2e}"})
            t_window = time.perf_counter()

        print(f"[epoch {epoch+1}/{cfg.epochs}] time={time.perf_counter() - epoch_start:.1f}s | "
              f"last_loss={loss_log[-1]['loss']:.4f} | trainable={param_log[-1]['trainable_params']:,}",
              flush=True)

    cuda_sync(device)
    train_time = time.perf_counter() - start_time
    if opt_step != total_opt_steps:
        raise RuntimeError(f"Ran {opt_step} optimizer steps, expected {total_opt_steps}")
    if layer_timeline:
        layer_timeline.append({"step": opt_step, "modes": dict(layer_timeline[-1]["modes"]), "final": True})
    flush_logs()

    # Static allocations: base mode from update 1 if a layer starts in base mode.
    if first_promotion_step is None and initial_modes and "full_ft" in initial_modes.values():
        first_promotion_step = 0
    post = [e["step_sec"] for e in loss_log
            if first_promotion_step is not None and e["step"] > first_promotion_step]
    diagnostics = None
    if initial_modes is not None:
        diagnostics = switch_diagnostics(
            controller.switch_history if controller else [], initial_modes,
            total_opt_steps, trainable_per_update)
        save_json(diagnostics, paths["dynamic"] / "switch_diagnostics.json")

    mem = {}
    if device.type == "cuda":
        mem = {"peak_vram_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
               "peak_vram_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30}
    stats = {
        "train_time_sec": train_time,
        **mem,
        "total_opt_steps": opt_step,
        "final_trainable_params": param_log[-1]["trainable_params"],
        "avg_trainable_params_step_weighted": sum(trainable_per_update) / len(trainable_per_update),
        "total_params": total_params,
        "n_switch_events": len(controller.switch_history) if controller else 0,
        "n_switch_events_with_transitions": sum(1 for h in controller.switch_history if h["n_transitions"])
        if controller else 0,
        "switch_overhead_sec": switch_overhead,
        "branch_step": controller.branch_step if controller else None,
        "first_promotion_step": first_promotion_step,
        "post_promotion_optimizer_steps": len(post) if first_promotion_step is not None else None,
        "post_promotion_mean_step_sec": (sum(post) / len(post)) if post else None,
        "mean_step_sec": sum(e["step_sec"] for e in loss_log) / len(loss_log),
        "base_update_changed_fraction_mean": (sum(e["changed_fraction"] for e in probe_log) / len(probe_log))
        if probe_log else None,
        "dtype_census": dtype_census,
        "final_loss": loss_log[-1]["loss"],
    }
    return model, stats


def _relevant_config_fields(cfg: RunConfig) -> dict:
    """
    Return only the config fields that meaningfully describe THIS method's
    hyperparameters. Fields that don't apply are set to None, so grouped
    summary tables don't mislead reviewers (e.g., 'lora_r' on a full_ft row).
    """
    lora_methods = {"lora", "dora", "adalora", "loraga", "automode", "dyn_full_only",
                    "select_once", "fixed_mixed"}
    dynamic_methods = {"automode", "dyn_full_only", "select_once"}
    m = cfg.method
    return {
        "model_name": cfg.model_name,
        "method": cfg.method,
        "train_track": cfg.train_track,
        "seed": cfg.seed,
        "epochs": cfg.epochs,
        "learning_rate": cfg.learning_rate,
        "train_batch_size": cfg.train_batch_size,
        "grad_accum_steps": cfg.grad_accum_steps,
        "protocol_version": cfg.protocol_version,
        "lora_r": cfg.lora_r if m in lora_methods else None,
        "lora_alpha": cfg.lora_alpha if m in lora_methods else None,
        "lora_target_modules": ",".join(cfg.lora_target_modules),
        "dynamic_updates": cfg.dynamic_updates if m in dynamic_methods else None,
        "dynamic_threshold": cfg.dynamic_threshold if m in dynamic_methods else None,
        "importance_signal": cfg.importance_signal if m in dynamic_methods else None,
        "score_collection": cfg.score_collection if m in dynamic_methods else None,
        "fixed_base_layers": cfg.fixed_base_layers if m == "fixed_mixed" else None,
        "allocation_seed": cfg.allocation_seed if m == "fixed_mixed" else None,
        "topk_k": cfg.topk_k if m == "topk_static" else None,
        "deep_block_start": cfg.deep_block_start if m == "topk_deep_block" else None,
        "deep_block_end": cfg.deep_block_end if m == "topk_deep_block" else None,
    }


def run_experiment(cfg: RunConfig) -> Dict[str, Any]:
    """
    Legacy grid entry point: build, train, evaluate, export, return result row.
    Any failure (including evaluation or export) returns a failed row and
    never writes COMPLETE. The rebuttal campaign uses automode.rebuttal instead.
    """
    from automode.models import build_model_for_method, get_tokenizer
    from automode.data import build_train_dataloader
    from automode.eval import run_evals
    from automode.export import export_full_inference_model

    paths = ensure_dirs(cfg)
    if is_run_complete(cfg):
        print(f"[skip] run {cfg.run_id()} already complete (verified identity).")
        return load_json(paths["run"] / "COMPLETE")

    save_json(cfg.to_dict(), paths["configs"] / "run_config.json")
    set_seed(cfg.seed)
    cleanup_memory()
    from automode.config import METHOD_FIDELITY
    base_row = {
        "run_id": cfg.run_id(),
        "variant": cfg.variant_label(),
        "paper_fidelity": METHOD_FIDELITY.get(cfg.method, "unknown"),
        **_relevant_config_fields(cfg),
    }
    try:
        tokenizer = get_tokenizer(cfg.model_name, cfg.model_revision)
        train_loader = build_train_dataloader(cfg, tokenizer)
        model = build_model_for_method(cfg, tokenizer=tokenizer, train_loader=train_loader)
        if cfg.method == "dora":  # other PEFT builders already freeze non-layer params
            freeze_non_layer_params(model)
        model, train_stats = train_one_run(cfg, model, train_loader, tokenizer, paths)
        eval_results = run_evals(cfg, model, tokenizer, paths)
        if not eval_results:
            raise RuntimeError("No evaluation produced results; refusing to mark complete")
        export_stats = {}
        if cfg.save_model:
            export_stats = export_full_inference_model(
                model, tokenizer, paths["run"] / "export", cfg, {"run_id": cfg.run_id()})
        result = {
            **base_row,
            "status": "ok",
            **train_stats,
            "data_stats": getattr(train_loader, "data_stats", None),
            **{f"eval_{k}": v for k, v in eval_results.items()},
            "export": export_stats or None,
            "config": cfg.to_dict(),
            "scientific_config": cfg.scientific_dict(),
        }
        mark_run_complete(cfg, result)
        del model
        cleanup_memory()
        return result
    except Exception as e:
        import traceback
        print(f"[ERROR] run {cfg.run_id()} failed: {e}")
        traceback.print_exc()
        cleanup_memory()
        return {**base_row, "status": f"failed: {repr(e)}", "config": cfg.to_dict()}
