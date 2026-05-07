"""
Unified training loop for AutoMode and all baselines.

Design principles
-----------------
1. ONE train_one_run() function handles every method. Method-specific logic
   is isolated in build_model_for_method() (from models.py) and applied via
   the controller pattern (AutoMode) or via static requires_grad flags (rest).

2. Scheduler is built ONCE and NEVER rebuilt. This fixes the sawtooth LR bug
   that still exists in the 9B notebook. When a switch happens, we rebuild
   the optimizer's param list but preserve the scheduler's internal step count
   by swapping optimizer references.

3. Every run writes the same canonical set of logs to disk:
     configs/run_config.json       — exact hyperparameters
     logs/training_loss.json        — per-step loss (unscaled)
     logs/lr_schedule.json          — per-step LR (proves no sawtooth)
     logs/grad_norms.json           — per-step total grad norm before clip
     logs/trainable_params.json     — per-step trainable count (Figure 1)
     dynamic/switch_history.json    — AutoMode only: every switch event
     dynamic/layer_timeline.json    — per-step {layer_id: mode}
     evals/<benchmark>_eval.json    — full predictions + correctness
     checkpoints/final_model/       — saved only if cfg.save_model

4. Resume-safe: if logs/run_config.json already exists AND evals/*.json shows
   a completed run, the runner skips this config.

5. Peak VRAM measured and reported, so the paper table has compute numbers
   alongside accuracy.
"""

from __future__ import annotations
import gc
import json
import math
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from automode.config import RunConfig
from automode.core import (
    AutoModeController,
    count_trainable_params,
    freeze_non_layer_params,
    group_lora_modules_by_layer,
    identify_layer_for_param,
    current_layer_mode,
    get_transformer_layers,
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
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def is_run_complete(cfg: RunConfig) -> bool:
    """Check if this run has written a successful completion marker."""
    paths = ensure_dirs(cfg)
    marker = paths["run"] / "COMPLETE"
    return marker.exists()


def mark_run_complete(cfg: RunConfig, summary: dict):
    paths = ensure_dirs(cfg)
    save_json(summary, paths["run"] / "COMPLETE")


# ──────────────────────────────────────────────────────────────────────────────
# Optimizer & scheduler construction
# ──────────────────────────────────────────────────────────────────────────────

def build_optimizer(model, cfg: RunConfig):
    """
    Choose the optimizer based on method.

    Full-FT on memory-constrained GPUs uses AdamW8bit from bitsandbytes, which
    stores optimizer states in int8 and cuts optimizer memory ~4×. Every other
    method uses standard AdamW.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError(
            "Model has zero trainable parameters — check requires_grad flags."
        )

    if cfg.method == "full_ft" and cfg.use_8bit_optim_for_full_ft:
        try:
            import bitsandbytes as bnb
            return bnb.optim.AdamW8bit(
                params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
            )
        except Exception as e:
            print(f"[optimizer] AdamW8bit unavailable ({e}); falling back to AdamW.")
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


def rebuild_optimizer_preserving_lr(prev_optim, model, cfg: RunConfig):
    """
    When switching changes the trainable parameter set, we need a new
    optimizer with the new param list. The CURRENT learning rate (after
    whatever the scheduler has already applied) must be preserved.

    We do this by:
      1. Getting the current effective LR from prev_optim's param groups.
      2. Constructing the new optimizer with that LR as its 'base'.
      3. NOT rebuilding the scheduler — the existing scheduler keeps stepping
         its internal counter and will compute future LRs based on total_steps.

    Critical: This means the scheduler and optimizer diverge in their sense
    of "base LR", but since scheduler.step() computes LR relative to the
    optimizer's CURRENT lr (not a stored base), this works out. We also set
    param_group["lr"] to what the scheduler just computed, so it's consistent.
    """
    current_lr = prev_optim.param_groups[0]["lr"]
    current_wd = prev_optim.param_groups[0].get("weight_decay", cfg.weight_decay)

    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("Rebuild_optimizer found zero trainable params.")

    # Preserve optimizer class
    cls = type(prev_optim)
    try:
        new_optim = cls(params, lr=current_lr, weight_decay=current_wd)
    except TypeError:
        new_optim = torch.optim.AdamW(params, lr=current_lr, weight_decay=current_wd)
    return new_optim


def attach_optimizer_to_scheduler(scheduler, new_optimizer):
    """
    Swap the scheduler's optimizer reference in-place.

    The scheduler's .step() reads/writes optimizer.param_groups[i]["lr"].
    By swapping the reference, future scheduler.step() calls update the NEW
    optimizer's LR without resetting any step counters or warmup progress.
    """
    scheduler.optimizer = new_optimizer
    # Some schedulers cache base LR per param group; reset those to the
    # first param group's LR from the new optimizer.
    if hasattr(scheduler, "base_lrs"):
        base = new_optimizer.param_groups[0]["lr"]
        scheduler.base_lrs = [
            base for _ in scheduler.base_lrs
        ]


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

def train_one_run(
    cfg: RunConfig,
    model,
    train_loader: DataLoader,
    tokenizer,
    paths: Dict[str, Path],
) -> Tuple[Any, Dict[str, Any]]:
    """
    Train `model` under the policy specified by cfg.method. Returns (model, stats).

    Assumes:
      - model has been built by build_model_for_method() (in models.py)
      - train_loader yields tokenized batches with input_ids, attention_mask, labels
      - paths from ensure_dirs()
    """
    start_time = time.time()
    device = cfg.device if torch.cuda.is_available() else "cpu"

    # ─── Total optimizer steps across all epochs ─────────────────────────────
    # We count both full windows AND the tail (partial) window, because we
    # flush the tail at end-of-epoch below with properly-rescaled gradients.
    # This keeps opt_step count consistent between the code and the scheduler.
    micro_per_epoch = len(train_loader)
    full_opt_per_epoch = micro_per_epoch // cfg.grad_accum_steps
    has_tail = (micro_per_epoch % cfg.grad_accum_steps) != 0
    opt_per_epoch = full_opt_per_epoch + (1 if has_tail else 0)
    total_opt_steps = opt_per_epoch * cfg.epochs
    if total_opt_steps == 0:
        raise ValueError(
            f"Computed total_opt_steps=0 (micro_per_epoch={micro_per_epoch}, "
            f"grad_accum={cfg.grad_accum_steps}, epochs={cfg.epochs}). "
            f"Reduce grad_accum_steps or increase dataset size."
        )

    # ─── Optimizer + scheduler ───────────────────────────────────────────────
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, total_opt_steps, cfg)

    # ─── Controller (for dynamic methods) ────────────────────────────────────
    controller = None
    if cfg.method in ("automode", "dyn_full_only"):
        controller = AutoModeController(cfg, total_opt_steps)
        controller.attach(model, optimizer)
    elif cfg.method == "lisa":
        controller = LisaController(cfg, total_opt_steps)
        controller.attach(model, optimizer)
        # LISA changes trainable params at attach() — rebuild optimizer once
        optimizer = rebuild_optimizer_preserving_lr(optimizer, model, cfg)
        attach_optimizer_to_scheduler(scheduler, optimizer)
    elif cfg.method == "adagradselect":
        controller = AdaGradSelectController(cfg, total_opt_steps)
        controller.attach(model, optimizer)
        optimizer = rebuild_optimizer_preserving_lr(optimizer, model, cfg)
        attach_optimizer_to_scheduler(scheduler, optimizer)

    # ─── Logging buffers ─────────────────────────────────────────────────────
    loss_log: List[dict] = []
    lr_log: List[dict] = []
    grad_norm_log: List[dict] = []
    param_log: List[dict] = []
    layer_timeline: List[dict] = []

    # Initial snapshot (step 0)
    trainable_0, total_params = count_trainable_params(model)
    param_log.append({"step": 0, "trainable_params": trainable_0, "total_params": total_params})
    if controller is not None:
        layer_timeline.append({"step": 0, "modes": controller.layer_mode_snapshot()})

    # ─── Training loop ───────────────────────────────────────────────────────
    model.train()
    optimizer.zero_grad(set_to_none=True)
    opt_step = 0
    running_micro_loss = 0.0
    micro_in_window = 0   # tracks how many micro-batches are in the current accumulation window

    def _do_optimizer_step(window_size: int, epoch_1based: int):
        """
        Execute one optimizer step given the current accumulated gradients.
        `window_size` is the number of micro-batches that contributed to the grads.
        If window_size < grad_accum_steps, we rescale so the gradient represents
        the mean gradient over the actual window, matching full-window semantics.
        """
        nonlocal opt_step, running_micro_loss, micro_in_window, optimizer
        if window_size == 0:
            return False

        # Rescale gradients if this is a partial (tail) window.
        # Loss was divided by grad_accum_steps each micro-step, giving
        # sum_i(g_i / grad_accum). We want sum_i(g_i / window_size) so the
        # gradient magnitude matches a full window. Scale by grad_accum/window_size.
        if window_size != cfg.grad_accum_steps:
            scale = cfg.grad_accum_steps / window_size
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.mul_(scale)

        grad_norm_before_clip = compute_total_grad_norm(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        opt_step += 1

        current_lr = scheduler.get_last_lr()[0]

        # Mean loss per micro-batch in this window (unit-consistent across full/tail)
        mean_loss = (running_micro_loss * cfg.grad_accum_steps) / window_size

        loss_log.append({
            "step": opt_step, "epoch": epoch_1based,
            "loss": round(mean_loss, 6),
            "window_size": window_size,     # so figures can mark partial windows
        })
        lr_log.append({"step": opt_step, "lr": float(current_lr)})
        grad_norm_log.append({
            "step": opt_step, "grad_norm": round(grad_norm_before_clip, 6),
        })
        trainable_now, _ = count_trainable_params(model)
        param_log.append({
            "step": opt_step, "epoch": epoch_1based,
            "trainable_params": trainable_now,
        })

        running_micro_loss = 0.0
        micro_in_window = 0
        return True

    for epoch in range(cfg.epochs):
        epoch_start = time.time()
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{cfg.epochs} [{cfg.method}]",
            leave=False,
        )
        n_batches = len(train_loader)
        for micro_step, batch in enumerate(pbar):
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

            # Forward + backward
            outputs = model(**batch)
            loss = outputs.loss / cfg.grad_accum_steps
            loss.backward()
            running_micro_loss += loss.item()
            micro_in_window += 1

            # Importance signal accumulation (AutoMode etc.)
            if controller is not None:
                controller.on_micro_step(model)

            is_full_window = (micro_step + 1) % cfg.grad_accum_steps == 0
            is_last_micro = (micro_step + 1) == n_batches
            # Trigger opt step if we've completed a full window OR we're at
            # the final micro-batch of the epoch with leftover grads in the window.
            if is_full_window or (is_last_micro and micro_in_window > 0):
                did_step = _do_optimizer_step(micro_in_window, epoch + 1)

                if did_step and controller is not None:
                    changed = controller.maybe_switch(model, opt_step)
                    if changed:
                        optimizer = rebuild_optimizer_preserving_lr(
                            optimizer, model, cfg,
                        )
                        attach_optimizer_to_scheduler(scheduler, optimizer)
                        controller.rebind_optimizer(optimizer)
                        layer_timeline.append({
                            "step": opt_step,
                            "modes": controller.layer_mode_snapshot(),
                        })

                if loss_log:
                    pbar.set_postfix({
                        "loss": loss_log[-1]["loss"],
                        "lr": f"{lr_log[-1]['lr']:.2e}",
                    })

        epoch_time = time.time() - epoch_start
        print(f"[epoch {epoch+1}/{cfg.epochs}] "
              f"time={epoch_time:.1f}s | "
              f"last_loss={loss_log[-1]['loss']:.4f} | "
              f"trainable={param_log[-1]['trainable_params']:,}")

    train_time = time.time() - start_time
    peak_vram_gb = (
        torch.cuda.max_memory_allocated() / (1024 ** 3)
        if torch.cuda.is_available() else 0.0
    )

    # ─── Save all logs ───────────────────────────────────────────────────────
    save_json({"loss_log": loss_log}, paths["logs"] / "training_loss.json")
    save_json({"lr_log": lr_log}, paths["logs"] / "lr_schedule.json")
    save_json({"grad_norm_log": grad_norm_log}, paths["logs"] / "grad_norms.json")
    save_json({"param_log": param_log}, paths["logs"] / "trainable_params.json")

    if controller is not None:
        save_json(
            {"switch_history": controller.switch_history},
            paths["dynamic"] / "switch_history.json",
        )
        save_json(
            {"layer_timeline": layer_timeline},
            paths["dynamic"] / "layer_timeline.json",
        )

    # ─── Save checkpoint if requested ────────────────────────────────────────
    if cfg.save_model:
        save_dir = paths["checkpoints"] / "final_model"
        save_dir.mkdir(parents=True, exist_ok=True)
        try:
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            print(f"[checkpoint] Saved to {save_dir}")
        except Exception as e:
            print(f"[checkpoint] Save failed: {e}")

    stats = {
        "train_time_sec": train_time,
        "peak_vram_gb": peak_vram_gb,
        "total_opt_steps": opt_step,
        "final_trainable_params": param_log[-1]["trainable_params"],
        "total_params": total_params,
        "n_switches": len(controller.switch_history) if controller else 0,
    }
    return model, stats


def _relevant_config_fields(cfg: RunConfig) -> dict:
    """
    Return only the config fields that meaningfully describe THIS method's
    hyperparameters. Fields that don't apply are set to None, so grouped
    summary tables don't mislead reviewers (e.g., 'lora_r' on a full_ft row).
    """
    # Always relevant
    row = {
        "model_name": cfg.model_name,
        "method": cfg.method,
        "train_track": cfg.train_track,
        "seed": cfg.seed,
        "epochs": cfg.epochs,
        "learning_rate": cfg.learning_rate,
        "train_batch_size": cfg.train_batch_size,
        "grad_accum_steps": cfg.grad_accum_steps,
    }
    # Method-specific fields: only include when relevant
    m = cfg.method
    row["lora_r"] = cfg.lora_r if m in {"lora", "dora", "adalora", "loraga",
                                         "automode", "dyn_full_only"} else None
    row["lora_alpha"] = cfg.lora_alpha if m in {"lora", "dora", "adalora", "loraga",
                                                 "automode", "dyn_full_only"} else None
    row["lora_target_modules"] = (",".join(cfg.lora_target_modules)
                                   if m in {"lora", "dora", "adalora", "loraga",
                                            "automode", "dyn_full_only"} else None)
    row["dynamic_updates"] = cfg.dynamic_updates if m in {"automode", "dyn_full_only"} else None
    row["dynamic_threshold"] = cfg.dynamic_threshold if m in {"automode", "dyn_full_only"} else None
    row["importance_signal"] = cfg.importance_signal if m in {"automode", "dyn_full_only"} else None
    row["topk_k"] = cfg.topk_k if m == "topk_static" else None
    row["deep_block_start"] = cfg.deep_block_start if m == "topk_deep_block" else None
    row["deep_block_end"] = cfg.deep_block_end if m == "topk_deep_block" else None
    row["adalora_init_r"] = cfg.adalora_init_r if m == "adalora" else None
    row["adalora_target_r"] = cfg.adalora_target_r if m == "adalora" else None
    row["lisa_num_layers"] = cfg.lisa_num_layers if m == "lisa" else None
    row["lisa_period"] = cfg.lisa_period if m == "lisa" else None
    row["adagrad_pct"] = cfg.adagrad_pct if m == "adagradselect" else None
    return row


def run_experiment(cfg: RunConfig) -> Dict[str, Any]:
    """
    End-to-end single run: build, train, evaluate, log, return result row.

    This is the main entry point the grid runner calls for each config.
    """
    # Import here to avoid circular imports
    from automode.models import build_model_for_method, get_tokenizer
    from automode.data import build_train_dataloader
    from automode.eval import run_evals

    paths = ensure_dirs(cfg)

    # Resume check
    if is_run_complete(cfg):
        print(f"[skip] run {cfg.run_id()} already complete.")
        return load_json(paths["run"] / "COMPLETE")

    # Save config immediately so we have a record even if training crashes
    save_json(cfg.to_dict(), paths["configs"] / "run_config.json")

    set_seed(cfg.seed)
    cleanup_memory()

    try:
        tokenizer = get_tokenizer(cfg.model_name)
        train_loader = build_train_dataloader(cfg, tokenizer)
        model = build_model_for_method(cfg, tokenizer=tokenizer, train_loader=train_loader)

        # Belt-and-suspenders freeze (in case the method builder didn't)
        if cfg.method in ("automode", "dyn_full_only", "lora", "dora",
                         "adalora", "loraga"):
            freeze_non_layer_params(model)

        print(f"\n{'=' * 80}")
        print(f"[run] {cfg.run_id()} | method={cfg.method} | model={cfg.model_name}")
        print(f"      track={cfg.train_track} | seed={cfg.seed} | epochs={cfg.epochs}")
        trainable, total = count_trainable_params(model)
        print(f"      trainable={trainable:,} / total={total:,} ({100*trainable/total:.3f}%)")
        print(f"{'=' * 80}\n")

        model, train_stats = train_one_run(cfg, model, train_loader, tokenizer, paths)
        eval_results = run_evals(cfg, model, tokenizer, paths)

        # Import locally to avoid circular dependency at module load
        from automode.config import METHOD_FIDELITY

        result = {
            "run_id": cfg.run_id(),
            "method": cfg.method,
            "variant": cfg.variant_label(),
            "paper_fidelity": METHOD_FIDELITY.get(cfg.method, "unknown"),
            "status": "ok",
            **_relevant_config_fields(cfg),
            **train_stats,
            **{f"eval_{k}": v for k, v in eval_results.items()},
            "config": cfg.to_dict(),  # full config for reproducibility
        }
        mark_run_complete(cfg, result)
        del model
        cleanup_memory()
        return result

    except Exception as e:
        import traceback
        from automode.config import METHOD_FIDELITY
        print(f"[ERROR] run {cfg.run_id()} failed: {e}")
        traceback.print_exc()
        cleanup_memory()
        return {
            "run_id": cfg.run_id(),
            "method": cfg.method,
            "variant": cfg.variant_label(),
            "paper_fidelity": METHOD_FIDELITY.get(cfg.method, "unknown"),
            "status": f"failed: {repr(e)}",
            **_relevant_config_fields(cfg),
            "config": cfg.to_dict(),
        }
