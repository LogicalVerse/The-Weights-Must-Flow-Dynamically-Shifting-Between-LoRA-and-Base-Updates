# AutoMode — NeurIPS Submission Codebase

**Deadline**: May 4, 2026 (~18 days from Apr 16 handoff).

**Paper claim**: AutoMode — dynamic per-layer switching between LoRA and Full-FT,
driven by gradient-norm importance — outperforms both baselines on GSM8K and MATH
while running faster than Full-FT.

## What's in here

```
automode_pkg/
├── automode/
│   ├── __init__.py      Public API
│   ├── config.py        RunConfig dataclass + preset builders for every method
│   ├── core.py          Switching logic: LoraLayer identification, promote/demote,
│   │                    importance signal collectors (GradNorm/Fisher/EMA/Random),
│   │                    AutoModeController, GumbelGate for the continuous extension
│   ├── models.py        Per-method model builders (11 methods + automode)
│   ├── data.py          Dataset loaders: GSM8K, MATH, MetaMathQA, Alpaca
│   ├── eval.py          Benchmark evaluators: GSM8K, MATH, MMLU 5-shot, ARC-C
│   ├── train.py         Unified training loop with sawtooth-LR bug fix
│   └── grid.py          Tiered experiment grid + per-GPU sharding + run_grid
├── tests/
│   └── test_switching.py   Unit tests including the one that catches the prefix bug
└── notebooks/
    ├── driver_gpu0_pro6000.ipynb    RTX PRO 6000: Gemma-2-9B, LLaMA-3-8B
    ├── driver_gpu1_5090.ipynb       RTX 5090: Qwen2.5-3B, Gemma-3-4B
    ├── driver_gpu2_5070.ipynb       RTX 5070: Gemma-2-2B + all ablations
    ├── tier2_forgetting_evals.ipynb MMLU + ARC-C on saved checkpoints
    └── figures.ipynb                Reproduces all NeurIPS figures from JSONs
```

## Bugs this codebase fixes from the previous notebooks

1. **Prefix bug (2B notebook, unfixed despite chat-index claim)**: identifying
   "layer N" by string prefix `"model.layers.N."` failed against PEFT-wrapped
   names `"base_model.model.model.layers.N."`, so every AutoMode switch was a
   silent no-op and 2B runs produced pure-LoRA numbers. Here we use
   `isinstance(module, LoraLayer)` instead — robust across PEFT versions.

2. **Sawtooth LR (9B notebook)**: `get_scheduler(... num_warmup_steps=W)` was
   being called on every switch, restarting warmup each time. Here the scheduler
   is built exactly once, and rebuilding the optimizer swaps the scheduler's
   `.optimizer` reference in place — step counter preserved.

3. **lm_head/embed_tokens silently trainable**: PEFT-wrapped models sometimes
   flag these as `requires_grad=True`, inflating parameter count by 100s of
   millions. `freeze_non_layer_params()` runs belt-and-suspenders after wrapping.

4. **Accumulator timing**: importance scores were accumulated across switch
   windows without clearing. Here `collector.reset()` runs right after
   `compute_scores()` in `maybe_switch()`, so each window represents fresh
   signal.

5. **Fake sanity check**: the 2B notebook asserted `trainable_init < 15M`
   (lower bound, LoRA passes). Here `tests/test_switching.py::test_promote_increases_trainable_count`
   asserts the count MUST increase after FFT promotion.

## Setup

```bash
cd automode_pkg
pip install -e .   # or pip install -r requirements.txt
pip install torch transformers peft datasets accelerate bitsandbytes tqdm pandas scipy matplotlib pytest

# Gated models (Gemma, LLaMA) need HF_TOKEN
export HF_TOKEN=hf_...
```

Then RUN THE TESTS before ANY training:

```bash
cd automode_pkg
pytest tests/test_switching.py tests/test_training_invariants.py -v
```

`test_switching.py` verifies the switching primitives (it's the suite that
catches the classic prefix-matching no-op bug). `test_training_invariants.py`
verifies two things the earlier validator round flagged:

  - **scheduler continuity across switches** (the fix for the sawtooth-LR bug
    that existed in the original 9B notebook)
  - **gradient-accumulation tail flush** (so the last partial window of each
    epoch isn't silently dropped)

All tests must pass. They use a tiny FakeCausalLM — no model downloads needed.

## Execution plan (18 days → May 4)

### Day 1–2: Sanity

1. `pytest tests/test_switching.py tests/test_training_invariants.py -v` — all green
2. Each GPU runs its driver notebook's smoke cell (1 fast config, ~5 min) to
   catch data-loading / tokenizer / model-loading failures
3. Quick visual check: open `figures.ipynb` and verify `fig7_lr_schedule` from
   the smoke run shows a monotone curve with NO sawtooth

### Day 3–12: Tier 0 + Tier 4

All three driver notebooks kick off simultaneously. Each resumes on restart.

| GPU | Assignment | Tiers | Est. runs | Est. wallclock |
|---|---|---|---|---|
| RTX PRO 6000 | Gemma-2-9B, LLaMA-3-8B | 0+1+4 | ~70 | 10-14 days |
| RTX 5090 | Qwen2.5-3B, Gemma-3-4B | 0+1+4 | ~50 | 6-8 days |
| RTX 5070 | Gemma-2-2B, all ablations | 0+3+4 | ~100 | 5-7 days |

Monitor: `runs/gpu{0,1,2}_results.csv` grows as runs complete.

### Day 10–14: Tier 2 forgetting evals (parallel, eval-only)

Once Tier 0 produces saved checkpoints (full_ft seed 8, automode_u10_t10 seed 8
for each model × track), kick off `tier2_forgetting_evals.ipynb` on whichever
GPU is free. Each checkpoint eval is ~1–2 hr.

### Day 14–18: Writing + figures

1. `figures.ipynb` reproduces all plots from logged JSONs
2. Paper draft integration
3. Buffer for re-running any failed runs

## Scope note

As of the Apr 16 handoff, you requested maximal scope: 5 models × 5 benchmarks
× all methods × 3 seeds. Compute math says this is ~5× what's available.

This codebase implements a **tiered** approach that gives you that scope as a
target but fails gracefully:

- **Tier 0 (headline, 108 runs)** must complete for the paper to exist.
  3 models × 2 tracks × 6 methods × 3 seeds = 108 runs. All subsequent tiers
  are bonus.
- **Tier 1 (generality, 12 runs)** adds LLaMA-3-8B and Gemma-3-4B at 1 seed.
  Gives the "does it generalize?" answer cheaply.
- **Tier 2 (forgetting, ~20 evals)** is eval-only on saved checkpoints.
  Gives MMLU and ARC-C numbers essentially for free.
- **Tier 3 (ablations, ~45 runs)** is 2B-only. Fisher vs grad_norm, threshold
  sweeps, cadence sweeps, dyn_full_only (no-LoRA-fallback).
- **Tier 4 (supplementary, 72 runs)** adds LISA, AdaGradSelect, AdaLoRA, DoRA
  baselines on the 3 core models.

If Tier 0 finishes but Tier 1 doesn't, the paper still reports 3 models. If
Tier 4 doesn't finish, the baseline comparison gets LISA + AdaGradSelect on
the 2B only — still sufficient.

## Resume semantics

- Each run writes `COMPLETE` marker on success.
- `run_grid()` skips any config whose `run_id()` already has a COMPLETE marker.
- CSVs are written incrementally so a crash only loses the current run.
- To re-run a failed run: delete the corresponding `runs/tier*/RUN_ID/` folder.

## Interpreting the figures

- **fig7_lr_schedule** is the trust-building figure. Reviewers familiar with
  dynamic PEFT methods know the sawtooth failure mode. This figure must show a
  clean cosine curve for the AutoMode run — if it shows sawtooth teeth, the
  fix didn't land.

- **fig5_layer_freq** shows which layers AutoMode prefers to upgrade. The
  paper's narrative is that AutoMode finds a middle-deep block (layers
  ~13–19 on 2B, ~22–35 on 9B) that matches intuitions about task-specific
  computation in transformers. If the heatmap is uniform, the gradient signal
  isn't informative and the random-signal ablation will probably match
  performance — check Tier 3 results to know.

- **fig2_pareto** is the "paper in one figure" — AutoMode should sit on the
  upper-left frontier, dominating LoRA and matching Full-FT at lower param count.

## Known limitations (to state explicitly in the paper)

- LoRA-GA baseline requires a forked PEFT install; falls back to standard LoRA
  with a warning if unavailable. Paper should either install the fork or drop
  LoRA-GA from baselines.

- Tier 1 runs single-seed, so no error bars for LLaMA/Gemma-3-4B results.
  Paper should report these as "generality check, single seed" and not claim
  statistical significance at those scales.

- Fisher importance uses Adam's `exp_avg_sq` as a diagonal approximation.
  This is standard ("Squisher", 2025) but not the empirical Fisher. Appendix
  should note this.
