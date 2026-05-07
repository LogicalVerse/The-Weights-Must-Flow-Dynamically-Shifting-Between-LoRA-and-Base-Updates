"""
Unified configuration for all AutoMode experiments.

Design principles
-----------------
1. One RunConfig dataclass handles every method (no per-method config classes).
2. Method-specific fields are optional; defaults are sensible if unused.
3. Serializable to JSON via dataclasses.asdict for per-run reproducibility.
4. Frozen=False because we mutate fields in preset builders; every run gets
   a fresh deepcopy before training to prevent cross-run contamination.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Optional, Tuple
import hashlib
import json

# Deferred torch import: kept out of top-level so config.py can be imported
# in tooling contexts where torch isn't available (e.g. CI linting).
def _dtype_map():
    import torch
    return {
        "bfloat16": torch.bfloat16,
        "float16":  torch.float16,
        "float32":  torch.float32,
    }


class ImportanceSignal(str, Enum):
    """
    How AutoMode computes the per-layer importance score.

    GRAD_NORM: S_l = sqrt(mean_p(||grad_p L||^2))          (original paper)
    FISHER:    S_l = sqrt(mean_p(v_t))                     (Adam's 2nd-moment as Fisher diagonal)
    EMA_GRAD:  exponential moving average of GRAD_NORM     (smoother than raw)
    RANDOM:    uniform random per layer                    (ablation: does gradient matter?)
    """
    GRAD_NORM = "grad_norm"
    FISHER = "fisher"
    EMA_GRAD = "ema_grad"
    RANDOM = "random"


class SwitchingMode(str, Enum):
    """
    How AutoMode assigns modes to layers once importance scores are computed.

    PERCENTILE: layers above t-th percentile of scores go to FFT (original)
    TOPK_FIXED: always exactly k layers in FFT (compute-budgeted variant)
    GUMBEL:     continuous gate g_l in [0,1], annealed via Gumbel-Softmax (NeurIPS extension)
    STATIC:     fixed layer set from start (for static baselines)
    """
    PERCENTILE = "percentile"
    TOPK_FIXED = "topk_fixed"
    GUMBEL = "gumbel"
    STATIC = "static"


# Method strings. Kept as strings (not enum) because they're used in file paths
# and serialized config. Enum-checking happens in build_model_for_method.
METHODS = frozenset({
    "full_ft",          # update all parameters
    "lora",             # pure LoRA, no switching
    "bitfit",           # update bias parameters only (strict — fails on bias-free models)
    "topk_static",      # static top-k layers in FFT, rest frozen
    "topk_deep_block",  # static deep block from 2B heatmap
    "automode",         # dynamic LoRA<->FFT switching (THE method)
    "adalora",          # adaptive LoRA rank allocation
    "dora",             # magnitude-direction LoRA
    "loraga",           # gradient-aligned LoRA init (falls back to LoRA if fork absent)
    "lisa",             # LISA-style random full-parameter updates (keeps embed+head)
    "adagradselect",    # AdaGradSelect-style gradient-guided block selection
    "dyn_full_only",    # AutoMode controller but non-selected layers FROZEN (ablation)
})


@dataclass
class RunConfig:
    """
    Complete specification for one fine-tuning run.

    Attributes grouped by purpose; method-specific ones at the bottom.
    """

    # ─── Model & Precision ───────────────────────────────────────────────────
    model_name: str = "google/gemma-2-2b"
    dtype: str = "bfloat16"           # bfloat16 | float16 | float32
    gradient_checkpointing: bool = True
    low_cpu_mem_usage: bool = True
    device: str = "cuda:0"

    # ─── Training Track & Eval Benchmarks ────────────────────────────────────
    # train_track controls the training corpus.
    # eval_benchmarks is a list of benchmarks to run after training.
    train_track: str = "gsm8k"        # gsm8k | math | metamath | alpaca
    eval_benchmarks: Tuple[str, ...] = ("gsm8k",)   # gsm8k, math, mmlu, arc_c

    # ─── Method ──────────────────────────────────────────────────────────────
    method: str = "lora"

    # ─── Core Optimization ───────────────────────────────────────────────────
    seed: int = 42
    epochs: int = 2
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler: str = "cosine"       # cosine | linear
    max_grad_norm: float = 1.0
    train_batch_size: int = 1
    grad_accum_steps: int = 16
    use_8bit_optim_for_full_ft: bool = True  # bnb.optim.AdamW8bit for full_ft only

    # ─── Data Sizing ─────────────────────────────────────────────────────────
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None
    max_source_len: int = 384
    max_target_len: int = 128
    max_new_tokens: int = 256

    # ─── Generation (Eval) ───────────────────────────────────────────────────
    eval_batch_size: int = 8
    do_sample: bool = True
    temperature: float = 0.7
    top_p: float = 0.95
    num_beams: int = 1
    sampling_k: int = 3               # majority-vote samples (1 = greedy deterministic)

    # ─── LoRA (shared by lora, dora, loraga, automode, dyn_full_only, lisa_lora) ──
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Tuple[str, ...] = ("q_proj", "v_proj")

    # ─── AdaLoRA ─────────────────────────────────────────────────────────────
    adalora_init_r: int = 16
    adalora_target_r: int = 8
    adalora_tinit: int = 200
    adalora_tfinal: int = 1000
    adalora_deltaT: int = 10

    # ─── TopK (static) ───────────────────────────────────────────────────────
    topk_k: int = 6                   # used when method = topk_static
    deep_block_start: int = 13        # used when method = topk_deep_block (2B defaults)
    deep_block_end: int = 19          # 9B override: 22..35

    # Whether static baselines (topk_static, topk_deep_block) keep lm_head
    # trainable. Default True — matches LISA/AdaGradSelect convention.
    # Set False for a stricter "layers-only" baseline.
    keep_lm_head_trainable: bool = True

    # ─── AutoMode (and variants) ─────────────────────────────────────────────
    dynamic_updates: int = 10         # switches per epoch (u)
    dynamic_threshold: int = 10       # percentile (t)
    importance_signal: str = "grad_norm"   # see ImportanceSignal enum
    switching_mode: str = "percentile"     # see SwitchingMode enum
    ema_decay: float = 0.9            # for EMA_GRAD
    gumbel_tau_start: float = 5.0     # initial temperature for Gumbel-Softmax
    gumbel_tau_end: float = 0.1       # final temperature
    gumbel_anneal_epochs: Optional[int] = None  # None = full training

    # ─── LISA-style ──────────────────────────────────────────────────────────
    lisa_num_layers: int = 4          # k: number of layers to update per step
    lisa_period: int = 50             # how often to resample

    # ─── AdaGradSelect-style ─────────────────────────────────────────────────
    adagrad_pct: int = 20             # percent of blocks to update
    adagrad_epsilon: float = 0.3      # epsilon-greedy exploration
    adagrad_dirichlet_alpha: float = 1.0

    # ─── LoRA-GA ─────────────────────────────────────────────────────────────
    loraga_steps_for_grad_est: int = 16

    # ─── Output & Checkpointing ──────────────────────────────────────────────
    output_root: str = "runs"
    save_model: bool = False          # save final checkpoint? (needed for post-hoc forgetting evals)
    save_eval_predictions: bool = True

    # ─── Misc ────────────────────────────────────────────────────────────────
    tier: int = 0                     # 0=headline, 1=generality, 2=forgetting, 3=ablation
    notes: str = ""

    # ─── Post-init: validate ─────────────────────────────────────────────────
    def __post_init__(self):
        if self.method not in METHODS:
            raise ValueError(
                f"Unknown method: {self.method!r}. "
                f"Must be one of: {sorted(METHODS)}"
            )
        if self.importance_signal not in {s.value for s in ImportanceSignal}:
            raise ValueError(f"Unknown importance_signal: {self.importance_signal}")
        if self.switching_mode not in {s.value for s in SwitchingMode}:
            raise ValueError(f"Unknown switching_mode: {self.switching_mode}")
        if self.dtype not in {"bfloat16", "float16", "float32"}:
            raise ValueError(f"Unknown dtype: {self.dtype}")

    # ─── Identity ────────────────────────────────────────────────────────────
    def run_id(self) -> str:
        """
        12-char hash of the config values that affect training outcome.
        Excludes notes and tier. Used for resume logic and folder naming.
        """
        payload = {
            "model": self.model_name,
            "method": self.method,
            "track": self.train_track,
            "eval": sorted(self.eval_benchmarks),
            "seed": self.seed,
            "epochs": self.epochs,
            "lr": self.learning_rate,
            "bs": self.train_batch_size,
            "grad_accum": self.grad_accum_steps,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_targets": sorted(self.lora_target_modules),
            "dyn_u": self.dynamic_updates,
            "dyn_t": self.dynamic_threshold,
            "imp": self.importance_signal,
            "mode": self.switching_mode,
            "topk_k": self.topk_k,
            "deep_block": (self.deep_block_start, self.deep_block_end),
            "adalora_init_r": self.adalora_init_r,
            "adalora_target_r": self.adalora_target_r,
            "lisa_num": self.lisa_num_layers,
            "adagrad_pct": self.adagrad_pct,
        }
        s = json.dumps(payload, sort_keys=True)
        return hashlib.md5(s.encode()).hexdigest()[:12]

    def variant_label(self) -> str:
        """Human-readable method label for plots and CSVs."""
        m = self.method
        if m == "automode":
            return (f"automode_u{self.dynamic_updates}_t{self.dynamic_threshold}"
                    f"_{self.importance_signal}_r{self.lora_r}")
        if m == "lora":
            return f"lora_r{self.lora_r}_a{self.lora_alpha}"
        if m == "dora":
            return f"dora_r{self.lora_r}"
        if m == "adalora":
            return f"adalora_init{self.adalora_init_r}_target{self.adalora_target_r}"
        if m == "topk_static":
            return f"topk_k{self.topk_k}"
        if m == "topk_deep_block":
            return f"topk_block_{self.deep_block_start}_{self.deep_block_end}"
        if m == "lisa":
            return f"lisa_k{self.lisa_num_layers}"
        if m == "adagradselect":
            return f"adagrad_pct{self.adagrad_pct}"
        if m == "dyn_full_only":
            return f"dyn_full_u{self.dynamic_updates}_t{self.dynamic_threshold}"
        return m

    def to_dict(self) -> dict:
        return asdict(self)


# Paper-fidelity classification. Used for labels and honest reviewer-facing reporting.
# "faithful"   — implementation matches the original paper as closely as practical
# "approximate" — implementation follows the paper's idea but deviates in ways that
#                 should be noted in the write-up; NOT a drop-in reproduction
# "preview"    — partial implementation, not intended for headline comparisons
METHOD_FIDELITY = {
    "full_ft":          "faithful",
    "lora":             "faithful",
    "bitfit":           "faithful",
    "topk_static":      "faithful",
    "topk_deep_block":  "faithful",
    "automode":         "faithful",   # the contribution itself
    "dyn_full_only":    "faithful",   # ablation of automode
    "adalora":          "faithful",   # uses upstream peft.AdaLoraConfig
    "dora":             "faithful",   # uses upstream peft.LoraConfig(use_dora=True)
    "lisa":             "approximate",      # now keeps embed+head; still missing period-mutation details
    "adagradselect":    "approximate",      # fixed explore/exploit; block definition differs from paper
    "loraga":           "approximate",      # relies on fork; falls back to LoRA if unavailable
}




def preset_full_ft(**kw) -> RunConfig:
    return RunConfig(method="full_ft", **kw)

def preset_lora(r=16, alpha=32, **kw) -> RunConfig:
    return RunConfig(method="lora", lora_r=r, lora_alpha=alpha, **kw)

def preset_dora(r=16, alpha=32, **kw) -> RunConfig:
    return RunConfig(method="dora", lora_r=r, lora_alpha=alpha, **kw)

def preset_adalora(**kw) -> RunConfig:
    return RunConfig(method="adalora", **kw)

def preset_bitfit(**kw) -> RunConfig:
    return RunConfig(method="bitfit", **kw)

def preset_topk_static(k=6, **kw) -> RunConfig:
    return RunConfig(method="topk_static", topk_k=k, **kw)

def preset_topk_deep_block(start=13, end=19, **kw) -> RunConfig:
    return RunConfig(method="topk_deep_block",
                     deep_block_start=start, deep_block_end=end, **kw)

def preset_automode(
    u=10, t=10,
    signal="grad_norm",
    switching_mode="percentile",
    r=16, alpha=32,
    **kw,
) -> RunConfig:
    return RunConfig(
        method="automode",
        dynamic_updates=u,
        dynamic_threshold=t,
        importance_signal=signal,
        switching_mode=switching_mode,
        lora_r=r,
        lora_alpha=alpha,
        **kw,
    )

def preset_dyn_full_only(u=10, t=10, signal="grad_norm", **kw) -> RunConfig:
    """AutoMode controller, but non-selected layers are FROZEN instead of LoRA.
    This isolates whether the LoRA fallback is what gives AutoMode its edge."""
    return RunConfig(
        method="dyn_full_only",
        dynamic_updates=u, dynamic_threshold=t,
        importance_signal=signal,
        **kw,
    )

def preset_lisa(k=4, period=50, **kw) -> RunConfig:
    return RunConfig(method="lisa", lisa_num_layers=k, lisa_period=period, **kw)

def preset_adagradselect(pct=20, **kw) -> RunConfig:
    return RunConfig(method="adagradselect", adagrad_pct=pct, **kw)

def preset_loraga(r=16, alpha=32, steps_for_grad_est=16, **kw) -> RunConfig:
    return RunConfig(
        method="loraga", lora_r=r, lora_alpha=alpha,
        loraga_steps_for_grad_est=steps_for_grad_est, **kw,
    )
