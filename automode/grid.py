"""
Tiered experiment grid with priority-based execution.

Philosophy
----------
Given ~18 days × 3 GPUs ~= 1300 GPU-hours available and 450+ desired runs,
a flat "run everything" grid will fail. This grid stratifies by defensibility:

  Tier 0 (HEADLINE)    — 3 models × 2 benchmarks × core methods × 3 seeds
  Tier 1 (GENERALITY)  — 2 additional models × 2 benchmarks × 3 methods × 1 seed
  Tier 2 (FORGETTING)  — MMLU + ARC-C on all saved checkpoints (no retraining)
  Tier 3 (ABLATIONS)   — Fisher vs grad_norm, dyn_full_only, threshold/cadence sweeps (2B only)

Execution order is strict: Tier 0 must complete before Tier 1 starts, etc.
If compute runs out at Tier 1, the paper still has a defensible Tier 0 story.

Per-GPU sharding
----------------
Each of the three GPUs gets a deterministic slice of the pending runs via
int(run_id, 16) % n_gpus == gpu_rank — stable across processes because
run_id() is MD5 hex. This means the three driver notebooks can
run independently without coordination — each GPU just claims its slice.
"""

from __future__ import annotations
from dataclasses import replace
from typing import List, Optional

from automode.config import (
    RunConfig,
    preset_full_ft, preset_lora, preset_dora, preset_adalora,
    preset_bitfit, preset_topk_static, preset_topk_deep_block,
    preset_automode, preset_dyn_full_only, preset_lisa, preset_adagradselect,
    preset_loraga,
)


# ──────────────────────────────────────────────────────────────────────────────
# Model-specific defaults
# ──────────────────────────────────────────────────────────────────────────────

MODEL_DEFAULTS = {
    "google/gemma-2-2b": dict(
        lr=5e-5, bs=1, grad_accum=16, max_src=384, max_tgt=128, max_new=256,
        deep_block=(13, 19), gpu="rtx_5070",
    ),
    "google/gemma-2-9b": dict(
        lr=2e-5, bs=4, grad_accum=8, max_src=384, max_tgt=128, max_new=256,
        deep_block=(22, 35), gpu="rtx_pro_6000",
    ),
    "Qwen/Qwen2.5-3B-Instruct": dict(
        lr=3e-5, bs=2, grad_accum=16, max_src=384, max_tgt=128, max_new=256,
        deep_block=(15, 25), gpu="rtx_5090",
    ),
    "meta-llama/Meta-Llama-3-8B": dict(
        lr=2e-5, bs=2, grad_accum=16, max_src=384, max_tgt=128, max_new=256,
        deep_block=(16, 28), gpu="rtx_pro_6000",
    ),
    "google/gemma-3-4b": dict(
        lr=3e-5, bs=2, grad_accum=16, max_src=384, max_tgt=128, max_new=256,
        deep_block=(15, 26), gpu="rtx_5090",
    ),
}


def _apply_model_defaults(cfg: RunConfig) -> RunConfig:
    """Populate cfg fields based on cfg.model_name."""
    d = MODEL_DEFAULTS.get(cfg.model_name)
    if d is None:
        print(f"[grid] No defaults for {cfg.model_name}; using cfg as-is.")
        return cfg
    cfg.learning_rate = d["lr"]
    cfg.train_batch_size = d["bs"]
    cfg.grad_accum_steps = d["grad_accum"]
    cfg.max_source_len = d["max_src"]
    cfg.max_target_len = d["max_tgt"]
    cfg.max_new_tokens = d["max_new"]
    cfg.deep_block_start, cfg.deep_block_end = d["deep_block"]
    return cfg


# ──────────────────────────────────────────────────────────────────────────────
# Benchmark-specific defaults
# ──────────────────────────────────────────────────────────────────────────────

def _apply_track_defaults(cfg: RunConfig) -> RunConfig:
    if cfg.train_track == "math":
        cfg.max_source_len = max(cfg.max_source_len, 512)
        cfg.max_target_len = max(cfg.max_target_len, 256)
        cfg.max_new_tokens = max(cfg.max_new_tokens, 512)
        cfg.do_sample = False
        cfg.sampling_k = 1
    elif cfg.train_track == "metamath":
        # MetaMathQA: HF dataset "meta-math/MetaMathQA" is the full 395K-row
        # corpus. The paper protocol (AdaGradSelect, LoRA-GA) uses a 40K
        # subset. Cap here unless the user has overridden max_train_samples.
        if cfg.epochs > 1:
            cfg.epochs = 1
        if cfg.max_train_samples is None:
            cfg.max_train_samples = 40000
    return cfg


# ──────────────────────────────────────────────────────────────────────────────
# The core "method lineup" that every (model × track) cell runs
# ──────────────────────────────────────────────────────────────────────────────

def _core_methods_for_cell(model: str, track: str, seeds: List[int],
                            output_root: str, tier: int = 0) -> List[RunConfig]:
    """
    Tier-0 headline lineup — ONLY the baselines we trust as paper-faithful
    for this codebase:

        full_ft           (upper-bound reference)
        lora r=16, r=32   (primary PEFT baseline)
        automode u=6/t=10 and u=10/t=10  (the contribution)
        topk_deep_block   (static-baseline: 'just train the deep layers')

    LISA, AdaGradSelect, AdaLoRA, DoRA, LoRA-GA are deliberately excluded here
    and live in Tier 4 instead, because:
      - LISA and AdaGradSelect are labeled 'approximate' in METHOD_FIDELITY
      - AdaLoRA/DoRA are faithful but change rank allocation dynamics
        (best reported as ablations, not alongside headline numbers)
      - LoRA-GA depends on an optional fork (fallback to LoRA if missing)

    This keeps the Tier-0 headline reviewer-defensible. The supplementary
    baselines appear in Tier 4 with explicit fidelity labels.
    """
    cfgs = []
    base = dict(
        model_name=model,
        train_track=track,
        eval_benchmarks=("gsm8k", "math") if track in ("metamath",) else (track,),
        output_root=output_root,
        tier=tier,
    )

    # Save checkpoints only for key methods (needed for post-hoc forgetting evals)
    checkpoint_methods = {"full_ft", "automode_u10_t10"}  # canonical runs
    for seed in seeds:
        sb = {**base, "seed": seed}

        # Full FT (primary baseline, upper bound)
        c = preset_full_ft(**sb)
        c.save_model = (seed == seeds[0])  # only first seed to save disk
        cfgs.append(c)

        # LoRA r=16 (primary PEFT baseline)
        c = preset_lora(r=16, alpha=32, **sb)
        c.save_model = (seed == seeds[0])
        cfgs.append(c)

        # LoRA r=32 (rank ablation)
        cfgs.append(preset_lora(r=32, alpha=64, **sb))

        # AutoMode (primary contribution) — two best configs
        for u, t in [(6, 10), (10, 10)]:
            c = preset_automode(
                u=u, t=t, signal="grad_norm",
                r=16, alpha=32, **sb,
            )
            c.save_model = (seed == seeds[0] and u == 10 and t == 10)
            cfgs.append(c)

        # TopK deep block (static baseline — defeats "just tune deep layers")
        cfgs.append(preset_topk_deep_block(**sb))

    return [_apply_track_defaults(_apply_model_defaults(c)) for c in cfgs]


def _supplementary_baselines_cell(model: str, track: str, seeds: List[int],
                                   output_root: str, tier: int = 0) -> List[RunConfig]:
    """
    Extra reviewer-requested baselines. These are SEPARATE from the core set
    because they're less essential — we run them after the core completes.
    """
    cfgs = []
    base = dict(
        model_name=model, train_track=track,
        eval_benchmarks=("gsm8k", "math") if track == "metamath" else (track,),
        output_root=output_root, tier=tier,
    )
    for seed in seeds:
        sb = {**base, "seed": seed}
        # LISA — the most important competitor
        cfgs.append(preset_lisa(k=4, period=50, **sb))
        # AdaGradSelect — recent 2025 baseline
        cfgs.append(preset_adagradselect(pct=20, **sb))
        # AdaLoRA — standard adaptive-LoRA baseline
        cfgs.append(preset_adalora(**sb))
        # DoRA — magnitude+direction LoRA
        cfgs.append(preset_dora(r=16, alpha=32, **sb))
    return [_apply_track_defaults(_apply_model_defaults(c)) for c in cfgs]


# ──────────────────────────────────────────────────────────────────────────────
# TIER 0: HEADLINE — 3 models × 2 benchmarks × core methods × 3 seeds
# ──────────────────────────────────────────────────────────────────────────────

def build_tier0(output_root: str = "runs/tier0") -> List[RunConfig]:
    """
    Defensible minimum: 3 scales (2B, 3B, 9B) × GSM8K + MATH × core × 3 seeds.
    This must finish for the paper to exist. Everything else is bonus.

    Estimated size:
      3 models × 2 tracks × 6 methods × 3 seeds = 108 runs
    """
    models = [
        "google/gemma-2-2b",
        "Qwen/Qwen2.5-3B-Instruct",
        "google/gemma-2-9b",
    ]
    tracks = ["metamath", "gsm8k"]
    # Note: we train on metamath and eval on gsm8k + math (AdaGradSelect protocol).
    # Also run GSM8K-train → GSM8K-eval as the paper's original setup, for
    # apples-to-apples comparison with the published 2B baseline numbers.
    seeds = [8, 25, 42]

    grid = []
    for model in models:
        for track in tracks:
            grid.extend(_core_methods_for_cell(model, track, seeds, output_root, tier=0))
    return grid


# ──────────────────────────────────────────────────────────────────────────────
# TIER 1: GENERALITY — 2 more models × 2 benchmarks × 3 methods × 1 seed
# ──────────────────────────────────────────────────────────────────────────────

def build_tier1(output_root: str = "runs/tier1") -> List[RunConfig]:
    """
    Cross-architecture generality: LLaMA-3-8B, Gemma-3-4B × core methods × 1 seed.

    Limited to: full_ft, lora_r16, automode_u10_t10.
    This gives a "does it generalize?" answer at 1/3 the cost of full Tier 0.

    Estimated size: 2 models × 2 tracks × 3 methods × 1 seed = 12 runs.
    """
    models = [
        "meta-llama/Meta-Llama-3-8B",
        "google/gemma-3-4b",
    ]
    tracks = ["metamath", "gsm8k"]
    seeds = [42]

    grid = []
    for model in models:
        for track in tracks:
            base = dict(
                model_name=model, train_track=track,
                eval_benchmarks=("gsm8k", "math") if track == "metamath" else (track,),
                output_root=output_root, tier=1,
            )
            for seed in seeds:
                sb = {**base, "seed": seed}
                grid.append(preset_full_ft(**sb))
                grid.append(preset_lora(r=16, alpha=32, **sb))
                c = preset_automode(u=10, t=10, r=16, alpha=32, **sb)
                c.save_model = True
                grid.append(c)
    return [_apply_track_defaults(_apply_model_defaults(c)) for c in grid]


# ──────────────────────────────────────────────────────────────────────────────
# TIER 2: FORGETTING — MMLU + ARC-C on saved checkpoints (no retraining)
# ──────────────────────────────────────────────────────────────────────────────

def build_tier2_eval_only(
    checkpoint_dirs: List[str],
    output_root: str = "runs/tier2",
) -> List[dict]:
    """
    Tier 2 is EVAL ONLY — no retraining. For each saved checkpoint from Tier 0/1,
    we run MMLU-5shot and ARC-Challenge.

    Returns a list of dicts (not RunConfigs) because this isn't a training run.
    The tier2 driver notebook loads the checkpoint and calls evaluate_mmlu + evaluate_arc.
    """
    return [{"checkpoint_dir": d, "output_root": output_root} for d in checkpoint_dirs]


# ──────────────────────────────────────────────────────────────────────────────
# TIER 3: ABLATIONS (2B only, GSM8K only, targeted questions)
# ──────────────────────────────────────────────────────────────────────────────

def build_tier3(output_root: str = "runs/tier3") -> List[RunConfig]:
    """
    Ablations to preempt the most common reviewer objections:

      (a) Does the importance signal matter? → grad_norm vs fisher vs random vs ema_grad
      (b) Is the LoRA fallback necessary? → dyn_full_only (freezes instead)
      (c) Threshold sensitivity? → t ∈ {10, 25, 50, 75}
      (d) Switch cadence sensitivity? → u ∈ {6, 10, 20}

    (Note: a continuous-gated Gumbel-Softmax variant is deferred to future work
    and is NOT scheduled here — see the GumbelGate block comment in core.py.)

    Only run on Gemma-2-2B + metamath track (eval on gsm8k + math), 3 seeds each.

    Estimated size: ~15 ablations × 3 seeds = 45 runs (on the cheapest model).
    """
    model = "google/gemma-2-2b"
    track = "metamath"
    eval_benches = ("gsm8k", "math")
    seeds = [8, 25, 42]

    grid = []
    base = dict(
        model_name=model, train_track=track,
        eval_benchmarks=eval_benches, output_root=output_root, tier=3,
    )

    # (a) Importance signal ablation
    for seed in seeds:
        for signal in ["grad_norm", "fisher", "random", "ema_grad"]:
            c = preset_automode(
                u=10, t=10, signal=signal, r=16, alpha=32,
                **{**base, "seed": seed},
            )
            c.notes = f"ablation_signal_{signal}"
            grid.append(c)

    # (b) LoRA fallback necessity
    for seed in seeds:
        c = preset_dyn_full_only(u=10, t=10, **{**base, "seed": seed})
        c.notes = "ablation_no_lora_fallback"
        grid.append(c)

    # (d) Threshold sensitivity
    for seed in seeds:
        for t in [10, 25, 50, 75]:
            if t == 10:  # already in (a) with grad_norm
                continue
            c = preset_automode(
                u=10, t=t, signal="grad_norm", **{**base, "seed": seed},
            )
            c.notes = f"ablation_threshold_t{t}"
            grid.append(c)

    # (e) Switch cadence sensitivity
    for seed in seeds:
        for u in [6, 20]:  # u=10 already covered
            c = preset_automode(
                u=u, t=10, signal="grad_norm", **{**base, "seed": seed},
            )
            c.notes = f"ablation_cadence_u{u}"
            grid.append(c)

    return [_apply_track_defaults(_apply_model_defaults(c)) for c in grid]


# ──────────────────────────────────────────────────────────────────────────────
# TIER 4: SUPPLEMENTARY BASELINES (LISA, AdaGradSelect, AdaLoRA, DoRA)
# ──────────────────────────────────────────────────────────────────────────────

def build_tier4(output_root: str = "runs/tier4") -> List[RunConfig]:
    """
    Additional baselines requested by reviewers of dynamic PEFT methods.

    Run only on the 3 core models × 2 benchmarks × 4 baselines × 3 seeds.
    Scheduled AFTER Tier 0 so we have full resolution on the main claim first.
    """
    models = [
        "google/gemma-2-2b",
        "Qwen/Qwen2.5-3B-Instruct",
        "google/gemma-2-9b",
    ]
    tracks = ["metamath", "gsm8k"]
    seeds = [8, 25, 42]

    grid = []
    for model in models:
        for track in tracks:
            grid.extend(_supplementary_baselines_cell(
                model, track, seeds, output_root, tier=4,
            ))
    return grid


# ──────────────────────────────────────────────────────────────────────────────
# Unified grid builder with tier selection
# ──────────────────────────────────────────────────────────────────────────────

def build_tier_grid(
    tiers: List[int],
    output_root: str = "runs",
) -> List[RunConfig]:
    """
    Build a combined grid across the requested tiers, in priority order.

    Usage: build_tier_grid([0, 1, 3]) returns Tier 0 configs first, then Tier 1,
    then Tier 3. The runner executes in list order, so Tier 0 completes before
    Tier 1 begins.
    """
    out = []
    if 0 in tiers:
        out.extend(build_tier0(f"{output_root}/tier0"))
    if 1 in tiers:
        out.extend(build_tier1(f"{output_root}/tier1"))
    if 3 in tiers:
        out.extend(build_tier3(f"{output_root}/tier3"))
    if 4 in tiers:
        out.extend(build_tier4(f"{output_root}/tier4"))
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Per-GPU sharding
# ──────────────────────────────────────────────────────────────────────────────

def shard_grid_by_gpu(
    grid: List[RunConfig],
    gpu_rank: int,
    n_gpus: int = 3,
    assignment: Optional[dict] = None,
) -> List[RunConfig]:
    """
    Partition the grid across GPUs. Two assignment strategies:

    (1) By model (recommended — co-locates model loads, reuses HF cache):
          Pass assignment={"google/gemma-2-9b": 0, "google/gemma-2-2b": 1, ...}
          Each GPU claims all runs for its assigned models.

    (2) By stable hash (fallback when assignment is None):
          gpu_rank gets all runs where int(run_id, 16) % n_gpus == gpu_rank.
          Uses the MD5-derived run_id (already hex) rather than Python's
          built-in hash(), which is process-randomized unless PYTHONHASHSEED
          is set — the latter would put the same run on different GPUs
          across restarts and break resume behavior.

    For this project (RTX PRO 6000 = big, 5090 = mid, 5070 = small), use (1)
    with an explicit assignment that puts the big models on the big GPU.
    """
    if assignment is not None:
        mine = [c for c in grid if assignment.get(c.model_name, 0) == gpu_rank]
    else:
        # Stable across processes and machines: run_id is MD5 hex (12 chars).
        mine = [c for c in grid if (int(c.run_id(), 16) % n_gpus) == gpu_rank]
    return mine


# Recommended assignment for your setup
DEFAULT_GPU_ASSIGNMENT = {
    # RTX PRO 6000 (96 GB) — the big models
    "google/gemma-2-9b": 0,
    "meta-llama/Meta-Llama-3-8B": 0,
    # RTX 5090 (32 GB) — mid-size
    "Qwen/Qwen2.5-3B-Instruct": 1,
    "google/gemma-3-4b": 1,
    # RTX 5070 (32 GB) — small model, cheapest runs (all ablations)
    "google/gemma-2-2b": 2,
}


# ──────────────────────────────────────────────────────────────────────────────
# Top-level runner
# ──────────────────────────────────────────────────────────────────────────────

def run_grid(
    configs: List[RunConfig],
    csv_path: Optional[str] = None,
    stop_on_error: bool = False,
    verbose: bool = True,
) -> List[dict]:
    """
    Execute a list of configs sequentially. Resume-safe (skips completed runs).
    Appends each result to csv_path as it completes so partial progress is preserved.
    """
    import pandas as pd
    from automode.train import run_experiment, is_run_complete
    from pathlib import Path

    results = []

    # Load existing CSV if present
    completed_ids = set()
    if csv_path and Path(csv_path).exists():
        df = pd.read_csv(csv_path)
        completed_ids = set(df["run_id"].tolist())
        results = df.to_dict("records")
        if verbose:
            print(f"[resume] loaded {len(completed_ids)} completed rows from {csv_path}")

    for i, cfg in enumerate(configs):
        rid = cfg.run_id()
        if rid in completed_ids or is_run_complete(cfg):
            if verbose:
                print(f"[{i+1}/{len(configs)}] skip {rid} ({cfg.variant_label()})")
            continue

        if verbose:
            print(f"\n[{i+1}/{len(configs)}] {rid} | {cfg.model_name} | {cfg.variant_label()} | seed={cfg.seed}")

        try:
            result = run_experiment(cfg)
            results.append(result)
            # Flatten for CSV: drop nested 'config' dict
            flat = {k: v for k, v in result.items() if k != "config"}
            if csv_path:
                pd.DataFrame(results).drop(
                    columns=["config"], errors="ignore",
                ).to_csv(csv_path, index=False)
        except Exception as e:
            if stop_on_error:
                raise
            print(f"[run_grid] {rid} crashed: {e}")
            continue

    return results
