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
    # Rebuttal core controls (see README "Method definitions").
    "select_once",      # AutoMode until its FIRST scheduled selection, then held fixed
    "fixed_mixed",      # fixed random set of q/v base layers from step 0, rest LoRA
    "static_qv",        # q/v base weights of every layer, no adapters
})

# The five methods the rebuttal campaign accepts (automode.rebuttal contract).
CORE_METHODS = ("automode", "lora", "select_once", "fixed_mixed", "static_qv")

# Legacy methods whose implementations are known-broken or unvalidated. They are
# kept in METHODS so legacy configs still parse, but building them raises.
UNAVAILABLE_METHODS = {
    "lisa": "optimizer attach path zeroed scheduler base_lrs during warmup; not revalidated",
    "adagradselect": "same broken optimizer attach path as LISA; not revalidated",
    "loraga": "silently fell back to plain LoRA when the LoRA-GA fork was missing",
    "adalora": "update_and_allocate() was never called, so ranks never adapted",
    "dyn_full_only": "frozen<->merged state transitions were never validated",
}

# Score-collection protocols. 'optimizer_step' samples the completed
# accumulation-window gradient once per optimizer update (after tail rescaling,
# before clipping). 'microbatch_legacy' is the submitted collector, which sampled
# the partially accumulated gradient after every micro-batch.
SCORE_COLLECTIONS = ("optimizer_step", "microbatch_legacy")

# Fields that locate or label a run but cannot change its outcome. Everything
# else in RunConfig is part of the scientific identity hash.
NON_SCIENTIFIC_FIELDS = frozenset({"output_root", "device", "notes", "tier"})


@dataclass
class RunConfig:
    """
    Complete specification for one fine-tuning run.

    Attributes grouped by purpose; method-specific ones at the bottom.
    """

    # ─── Model & Precision ───────────────────────────────────────────────────
    model_name: str = "google/gemma-2-2b"
    model_revision: Optional[str] = None     # immutable HF commit sha (rebuttal resolves it)
    dataset_revision: Optional[str] = None   # immutable HF dataset commit sha
    dtype: str = "bfloat16"           # bfloat16 | float16 | float32
    # Gemma-2 soft-caps attention logits; sdpa/flash paths historically ignored it.
    attn_implementation: str = "eager"
    # Storage dtype of trainable-eligible q/v BASE weights. "model" keeps them in
    # cfg.dtype (submitted protocol: pure-bf16 AdamW updates, which round away most
    # updates at small LR). "float32" keeps fp32 master weights and runs the forward
    # under autocast(cfg.dtype). Recorded in the run identity either way.
    base_weight_dtype: str = "model"
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
    seed: int = 42                     # parameter init / dropout (global torch RNG)
    data_seed: Optional[int] = None    # data order generator; None -> seed
    evaluation_seed: int = 20261029    # generation sampling, independent of training RNG
    epochs: int = 2
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler: str = "cosine"       # cosine | linear
    max_grad_norm: float = 1.0
    train_batch_size: int = 1
    grad_accum_steps: int = 16
    use_8bit_optim_for_full_ft: bool = True  # bnb.optim.AdamW8bit for full_ft only; raises if bnb missing

    # ─── Data Sizing ─────────────────────────────────────────────────────────
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None
    max_source_len: int = 384
    max_target_len: int = 128
    max_new_tokens: int = 256
    # Context budget for multiple-choice retention evals (MMLU/ARC), separate from
    # the GSM8K training source length so few-shot prompts never lose the question.
    mc_max_source_len: int = 2048
    mmlu_n_shot: int = 5

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
    score_collection: str = "optimizer_step"  # see SCORE_COLLECTIONS
    ema_decay: float = 0.9            # for EMA_GRAD
    gumbel_tau_start: float = 5.0     # initial temperature for Gumbel-Softmax
    gumbel_tau_end: float = 0.1       # final temperature
    gumbel_anneal_epochs: Optional[int] = None  # None = full training

    # ─── fixed_mixed control ─────────────────────────────────────────────────
    fixed_base_layers: Optional[int] = None   # number of layers training q/v base weights
    allocation_seed: Optional[int] = None     # seeds the layer choice only (stdlib random)

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
    protocol_version: str = "unspecified"

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
        if self.base_weight_dtype not in {"model", "float32"}:
            raise ValueError(f"Unknown base_weight_dtype: {self.base_weight_dtype}")
        if self.score_collection not in SCORE_COLLECTIONS:
            raise ValueError(f"Unknown score_collection: {self.score_collection}")
        if self.sampling_k < 1:
            raise ValueError("sampling_k must be >= 1")
        # JSON manifests give lists; keep the hashable/immutable tuple form.
        self.eval_benchmarks = tuple(self.eval_benchmarks)
        self.lora_target_modules = tuple(self.lora_target_modules)
        if self.method == "fixed_mixed" and (
                self.fixed_base_layers is None or self.allocation_seed is None):
            raise ValueError("fixed_mixed requires fixed_base_layers and allocation_seed")

    @property
    def effective_data_seed(self) -> int:
        return self.seed if self.data_seed is None else self.data_seed

    # ─── Identity ────────────────────────────────────────────────────────────
    def scientific_dict(self) -> dict:
        """Every field that can change the outcome (all but NON_SCIENTIFIC_FIELDS)."""
        return {k: v for k, v in asdict(self).items() if k not in NON_SCIENTIFIC_FIELDS}

    def run_id(self) -> str:
        """
        16-hex-char SHA-256 of scientific_dict(). Used for folder naming and
        resume. Earlier versions hashed a hand-picked subset that omitted e.g.
        max_train_samples, so smoke and full-data runs collided on one id.
        Code/protocol identity is added on top of this by automode.rebuttal.
        """
        s = json.dumps(self.scientific_dict(), sort_keys=True, default=str)
        return hashlib.sha256(s.encode()).hexdigest()[:16]

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
        if m == "select_once":
            return (f"select_once_u{self.dynamic_updates}_t{self.dynamic_threshold}"
                    f"_{self.importance_signal}_r{self.lora_r}")
        if m == "fixed_mixed":
            # allocation_seed is deliberately not in the label: it varies with seed.
            return f"fixed_mixed_b{self.fixed_base_layers}_r{self.lora_r}"
        return m

    def to_dict(self) -> dict:
        return asdict(self)


# Paper-fidelity classification. Used for labels and honest reviewer-facing reporting.
# "faithful"   — implementation matches the original paper as closely as practical
# "approximate" — implementation follows the paper's idea but deviates in ways that
#                 should be noted in the write-up; NOT a drop-in reproduction
# "preview"    — partial implementation, not intended for headline comparisons
# "unavailable" — building the method raises (see UNAVAILABLE_METHODS)
METHOD_FIDELITY = {
    "full_ft":          "faithful",
    "lora":             "faithful",
    "bitfit":           "faithful",
    "topk_static":      "faithful",
    "topk_deep_block":  "faithful",
    "automode":         "faithful",   # the contribution itself
    "select_once":      "faithful",   # rebuttal control, defined in README
    "fixed_mixed":      "faithful",   # rebuttal control, defined in README
    "static_qv":        "faithful",   # rebuttal control, defined in README
    "dora":             "faithful",   # uses upstream peft.LoraConfig(use_dora=True)
    **{m: "unavailable" for m in UNAVAILABLE_METHODS},
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
