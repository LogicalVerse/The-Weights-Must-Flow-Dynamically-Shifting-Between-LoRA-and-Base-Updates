# AutoMode: Dynamic Update-Mode Allocation for Efficient Fine-Tuning

This repository accompanies the submission *"The Weights Must Flow: Dynamically Shifting Between LoRA and Base Updates for Efficient Fine-Tuning"* (NeurIPS 2026). It contains the complete AutoMode training framework, all baseline implementations, evaluation code, unit tests, and experiment-grid infrastructure needed to reproduce the paper's results.

AutoMode is a fine-tuning framework for large language models that treats the **update-mode assignment** itself—LoRA adapter vs. direct base-weight update—as a per-step allocation problem solved online during training. Rather than committing to a fixed strategy at initialization, AutoMode continuously measures layer-level importance signals and reallocates compute capacity across the network at configurable switching intervals.

---

## Method Overview

### Update Modes

Every targeted projection (by default `q_proj` and `v_proj`) operates in one of two modes at any given point during training:

- **Adapter mode (LoRA):** The base projection weight is frozen. A low-rank adapter `(α/r)·BA` is added to the frozen weight and updated by the optimizer. This is the default starting mode for all layers.
- **Base-update mode (Full-FT):** The accumulated LoRA delta is merged into the base projection weight (`θ ← θ + (α/r)·BA`), the adapter parameters are frozen (they are now redundant), and the base weight is trained directly.

### Promotion and Demotion

- **Promotion (LoRA → base-update):** Executed by `promote_to_fft()` in `automode/core.py`. The adapter is merged into the base weight via `module.merge()`, adapter parameters are frozen, and base-layer parameters are unfrozen for direct gradient updates.
- **Demotion (base-update → LoRA):** Executed by `demote_to_lora()`. The current base weight (which now incorporates all FFT-phase learning) is frozen. A fresh adapter is initialized (`A ~ N(0, σ)`, `B = 0`, so `BA = 0` at the start of the LoRA phase), and the adapter parameters become the sole trainable elements for that layer.

LoRA remains the default fallback path: a layer that is not promoted—or that has been demoted—continues to be updated through its adapter.

### Importance Signal

At each micro-step, AutoMode accumulates a per-layer gradient-activity score. The default signal (`grad_norm`) computes the RMS gradient magnitude across all parameters in a layer:

```
S_l = sqrt( (1/|P_l|) · sum_p ||grad_p L||^2 )
```

Alternative signals implemented in `automode/core.py` include an exponential moving average of gradient norms (`ema_grad`), a Fisher-diagonal approximation via Adam's second-moment estimates (`fisher`), and a uniform random baseline (`random`) for ablation purposes.

### Switching Schedule

After every `update_interval` optimizer steps (derived from `dynamic_updates` switches per epoch), AutoMode calls `AutoModeController.maybe_switch()`. This method:

1. Computes per-layer importance scores from the accumulated signal.
2. Determines a threshold at the `dynamic_threshold`-th percentile of the current score distribution.
3. Promotes layers scoring at or above the threshold to base-update mode; demotes layers below it to LoRA mode.
4. Resets the importance accumulator, leaving a clean slate for the next window.
5. Logs the full switching decision (scores, transitions, trainable-parameter count) for figure reproduction.

When any layer changes mode, the optimizer is rebuilt over the new trainable parameter set while the scheduler's step counter and internal state are preserved—preventing warmup restarts at each switching event.

---

## Repository Layout

```
.
├── automode/                    # Main Python package
│   ├── __init__.py              # Public API surface: RunConfig, AutoModeController, run_experiment, …
│   ├── config.py                # RunConfig dataclass, ImportanceSignal/SwitchingMode enums,
│   │                            #   preset builders, METHOD_FIDELITY labels
│   ├── core.py                  # Layer identification, importance collectors, promote/demote
│   │                            #   primitives, AutoModeController orchestrator
│   ├── models.py                # build_model_for_method(): PEFT wrapping for all methods
│   ├── data.py                  # Dataset loading for GSM8K, MATH, MetaMathQA (HF datasets)
│   ├── train.py                 # Unified training loop, scheduler management, LisaController,
│   │                            #   AdaGradSelectController, run_experiment entry point
│   ├── eval.py                  # evaluate_gsm8k, evaluate_math, evaluate_mmlu, evaluate_arc
│   └── grid.py                  # build_tier_grid, build_tier0/1/2/3/4, run_grid, GPU sharding
│
├── tests/
│   ├── test_switching.py        # Unit tests: layer ID, LoRA grouping, promote/demote round-trip,
│   │                            #   importance-signal collectors, controller integration
│   └── test_training_invariants.py  # Scheduler continuity across switches, gradient-accumulation
│                                    #   tail-flush correctness, config assertion checks
│
├── notebooks/
│   ├── figures.ipynb            # Figure generation from completed run outputs
│   └── tier2_forgetting_evals.ipynb  # MMLU / ARC-C evaluation on saved checkpoints
│
├── pyproject.toml               # Package metadata and dependencies
└── README.md
```

> **Note on `runs/`:** Experiment output directories are written to `runs/` (or a user-specified `output_root`) at runtime and are not tracked in version control. See [Outputs and Figure Reproduction](#outputs-and-figure-reproduction) for the directory structure.

---

## Installation

### Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.1
- CUDA-capable GPU (all experiments use `bfloat16`)

### Editable Install

```bash
git clone <repository-url>
cd <repository-root>
pip install -e ".[dev]"
```

The `[dev]` extra installs `pytest`. The optional `[bnb]` extra installs `bitsandbytes` for 8-bit AdamW, which is used by default for `full_ft` runs on memory-constrained hardware:

```bash
pip install -e ".[dev,bnb]"
```

### Hugging Face Token

Several baseline models (e.g., `meta-llama/Meta-Llama-3-8B`) are gated. Log in before running:

```bash
huggingface-cli login
```

or set the `HF_TOKEN` environment variable. Gemma and Qwen models may similarly require acceptance of usage terms on the Hugging Face model hub.

---

## Quick Verification / Tests

Run the full test suite without downloading any models:

```bash
pytest tests/ -v
```

### What the Tests Verify

**`tests/test_switching.py`**

| Test class | What is verified |
|---|---|
| `TestLayerIdentification` | `identify_layer_for_param` correctly parses plain, PEFT-wrapped, and doubly-wrapped parameter names; layer IDs are sorted numerically, not lexicographically |
| `TestLoraGrouping` | `group_lora_modules_by_layer` finds all `LoraLayer` instances and groups them by transformer layer, independently of parameter-name prefixes |
| `TestSwitchingRoundtrip` | **Critical:** `promote_to_fft` strictly increases the trainable parameter count; the LoRA→FFT→LoRA round-trip returns to the original count; `demote_to_lora` resets `lora_B` to zero |
| `TestImportanceSignals` | `GradNormCollector`, `EMAGradCollector`, and `RandomCollector` each produce per-layer scores of the correct type and cardinality after a forward–backward pass |
| `TestControllerIntegration` | `AutoModeController` attaches cleanly, fires switching decisions at the correct interval, and produces non-trivial changes to trainable parameter counts |
| `TestFreezeNonLayerParams` | `freeze_non_layer_params` freezes `lm_head` and `embed_tokens` without disabling LoRA adapter trainability |

**`tests/test_training_invariants.py`**

| Test class | What is verified |
|---|---|
| `TestSchedulerContinuity` | After an optimizer rebuild triggered by a mode switch, the learning-rate schedule continues to decrease monotonically—no warmup restart or sawtooth pattern |
| `TestGradAccumTailFlush` | When the dataset size is not divisible by `grad_accum_steps`, the partial final window still produces an optimizer step, and gradients are rescaled to match full-window semantics |
| `TestAutoModeConfigAssertions` | `AutoModeController` raises `NotImplementedError` for unsupported `switching_mode` values (`topk_fixed`, `gumbel`) rather than silently ignoring them |
| `TestMethodFidelity` | Every method string in `METHODS` has a corresponding `METHOD_FIDELITY` label (`faithful`, `approximate`, or `preview`) |

---

## Running Experiments

### Single Run (Python API)

```python
from automode import run_experiment, preset_automode

cfg = preset_automode(
    u=10,                            # 10 switching decisions per epoch
    t=10,                            # promote layers above the 10th-percentile threshold
    signal="grad_norm",
    r=16, alpha=32,
    model_name="google/gemma-2-2b",
    train_track="metamath",
    eval_benchmarks=("gsm8k", "math"),
    epochs=2,
    seed=42,
    output_root="runs/single",
)
result = run_experiment(cfg)
print(result)
```

### Grid Runner

The grid runner executes a list of `RunConfig` objects in order, skipping any run whose output directory already contains a `COMPLETE` marker:

```python
from automode import run_grid, build_tier_grid

grid = build_tier_grid(tiers=[0], output_root="runs")
results = run_grid(grid, csv_path="runs/tier0_results.csv")
```

To re-run a failed experiment, delete its run directory (identified by its 12-character MD5 `run_id`) and re-launch the grid:

```bash
rm -rf runs/tier0/<run_id>
```

The grid will detect the missing `COMPLETE` marker and re-execute that configuration.

### Resumability

`run_experiment` checks for a `COMPLETE` file before training. Completed runs are skipped by both `run_experiment` and `run_grid`. Partial runs (where training started but did not complete) are re-executed from scratch; partial log files from the interrupted run are overwritten.

### Output Location

Each run is written to `<output_root>/<run_id>/`, where `run_id` is a deterministic 12-character hash of all training-outcome-relevant hyperparameters. The directory structure is:

```
<output_root>/<run_id>/
├── configs/
│   └── run_config.json          # Exact hyperparameters for this run
├── logs/
│   ├── training_loss.json       # Per-step unscaled loss
│   ├── lr_schedule.json         # Per-step learning rate (verify no sawtooth)
│   ├── grad_norms.json          # Per-step total gradient norm before clipping
│   └── trainable_params.json    # Per-step trainable parameter count
├── dynamic/                     # AutoMode runs only
│   ├── switch_history.json      # Every switching decision with scores and transitions
│   └── layer_timeline.json      # Per-step {layer_id: mode} mapping
├── evals/
│   ├── gsm8k_eval.json          # maj@1 accuracy with per-example predictions
│   ├── math_eval.json           # Exact-match accuracy
│   ├── mmlu_eval.json           # 5-shot accuracy (if evaluated)
│   └── arc_c_eval.json          # ARC-Challenge accuracy (if evaluated)
├── checkpoints/
│   └── final_model/             # Saved only if cfg.save_model = True
└── COMPLETE                     # Written last; presence indicates a successful run
```

---

## Experiment Organization

Experiments are stratified into tiers that can be constructed and executed independently using `build_tier_grid` or the individual `build_tier*` functions in `automode/grid.py`.

### Tier 0 — Headline Results

**Scope:** 3 model scales (Gemma-2-2B, Qwen2.5-3B-Instruct, Gemma-2-9B) × 2 training tracks (MetaMathQA, GSM8K) × 6 methods × 3 seeds.

**Methods:** Full-FT, LoRA (r=16), LoRA (r=32), AutoMode (u=6/t=10), AutoMode (u=10/t=10), TopK-DeepBlock (static layer selection baseline).

```python
from automode.grid import build_tier0, run_grid
grid = build_tier0(output_root="runs/tier0")
run_grid(grid, csv_path="runs/tier0_results.csv")
```

### Tier 1 — Cross-Architecture Generalization

**Scope:** 2 additional architectures (LLaMA-3-8B, Gemma-3-4B) × 2 training tracks × 3 methods × 1 seed.

**Methods:** Full-FT, LoRA (r=16), AutoMode (u=10/t=10).

```python
from automode.grid import build_tier1, run_grid
grid = build_tier1(output_root="runs/tier1")
run_grid(grid, csv_path="runs/tier1_results.csv")
```

### Tier 2 — Knowledge Retention Evaluation

Tier 2 is evaluation-only. It loads saved checkpoints from Tier 0/1 and evaluates MMLU (5-shot) and ARC-Challenge accuracy without any additional training. See `notebooks/tier2_forgetting_evals.ipynb`.

### Tier 3 — Ablation Studies

**Scope:** Gemma-2-2B, MetaMathQA → GSM8K + MATH, 3 seeds.

**Ablations:**
- **(a) Importance signal:** `grad_norm` vs. `fisher` vs. `ema_grad` vs. `random` (does the gradient-guided signal add value over arbitrary selection?)
- **(b) LoRA fallback:** `dyn_full_only` variant freezes non-promoted layers instead of applying LoRA adapters, isolating the contribution of the LoRA fallback path.
- **(c) Threshold sensitivity:** `dynamic_threshold` ∈ {10, 25, 50, 75}
- **(d) Switch cadence:** `dynamic_updates` ∈ {6, 10, 20}

```python
from automode.grid import build_tier3, run_grid
grid = build_tier3(output_root="runs/tier3")
run_grid(grid, csv_path="runs/tier3_results.csv")
```

### Tier 4 — Supplementary Baselines

Runs LISA, AdaGradSelect, AdaLoRA, and DoRA on the same 3-model × 2-track × 3-seed setup as Tier 0. These are executed after Tier 0 to ensure the primary claims are established first.

```python
from automode.grid import build_tier4, run_grid
grid = build_tier4(output_root="runs/tier4")
run_grid(grid, csv_path="runs/tier4_results.csv")
```

---

## Methods and Baselines

The following methods are implemented in `automode/models.py` and selectable via `RunConfig.method`:

| Method string | Description | Fidelity |
|---|---|---|
| `full_ft` | All parameters updated; upper-bound reference | Faithful |
| `lora` | Pure LoRA, no switching; primary PEFT baseline | Faithful |
| `dora` | Magnitude-direction decomposition via `peft.LoraConfig(use_dora=True)` | Faithful |
| `adalora` | Adaptive rank allocation via `peft.AdaLoraConfig` | Faithful |
| `topk_static` | Fixed set of top-k layers trained in full-FT; rest frozen | Faithful |
| `topk_deep_block` | Fixed contiguous deep-layer block trained in full-FT | Faithful |
| `automode` | **AutoMode**: dynamic LoRA↔base-update switching | Faithful |
| `dyn_full_only` | AutoMode controller, but non-promoted layers are frozen (ablation) | Faithful |
| `lisa` | LISA-style random layer sampling; embed and LM head always trainable | Approximate |
| `adagradselect` | Gradient-guided ε-greedy layer selection with Dirichlet exploitation | Approximate |
| `loraga` | Gradient-aligned LoRA initialization; falls back to LoRA if fork unavailable | Approximate |

**How AutoMode differs from related methods:**

- **vs. fixed LoRA / DoRA / AdaLoRA:** These methods fix the update mode at initialization. AutoMode reallocates between update modes continuously during training.
- **vs. full fine-tuning:** AutoMode applies full-parameter updates only to a dynamically selected subset of layers at any given time, substantially reducing the number of simultaneously active trainable parameters.
- **vs. LISA / AdaGradSelect / TopK-static:** These methods select *which layers* to update but fix the *form* of the update (typically full-parameter). AutoMode selects both which layers receive the most expressive updates *and* what form those updates take—base-weight modification for high-signal layers, LoRA adaptation for the remainder.
- **vs. AdaLoRA:** AdaLoRA adapts the rank of the LoRA decomposition. AutoMode adapts the update mode itself, toggling between the adapter and the base weight.

`METHOD_FIDELITY` in `automode/config.py` documents the degree to which each implementation matches the corresponding published algorithm. Methods labeled `approximate` deviate in ways that should be acknowledged when reporting comparisons.

---

## Outputs and Figure Reproduction

### Metrics Logged

Every completed run contains the following metrics in its `COMPLETE` file and in the per-benchmark `evals/*.json` files:

- **GSM8K maj@1:** majority-vote exact-match accuracy on the 1,319-example test set, using `k` samples per question (default `k=3`, configurable via `sampling_k`).
- **MATH exact match:** greedy-decoded accuracy with `\boxed{}` extraction on the MATH test set.
- **MMLU 5-shot accuracy:** across all 57 subjects (Tier 2 only).
- **ARC-Challenge accuracy:** (Tier 2 only).
- **Wall-clock training time** (`train_time_sec`).
- **Peak VRAM** (`peak_vram_gb`, measured via `torch.cuda.max_memory_allocated()`).
- **Trainable parameter fraction:** per-step `trainable_params` log enables computing mean fraction over the run.

### Figure Generation

With completed `runs/` directories, figures can be regenerated using `notebooks/figures.ipynb`. The notebook reads `logs/`, `dynamic/`, and `evals/` subdirectories across runs and produces:

- Training loss curves
- Per-step LR schedule (verifying scheduler continuity)
- Trainable parameter count over time per method
- Layer-mode timeline heatmaps for AutoMode runs (from `dynamic/layer_timeline.json`)
- Accuracy comparison tables and bar charts across methods and model scales

Summary results are collected across runs by matching `run_config.json` hyperparameters to their corresponding `COMPLETE` files, enabling reproducible figure generation without re-running experiments.

---

## Reproducibility Notes

The following implementation details are critical for obtaining results consistent with those reported in the paper. They are enforced in the current codebase but noted here for users adapting the code.

**1. LoRA layer detection via `isinstance(module, LoraLayer)`.**  
Layer identification uses `peft.tuners.lora.LoraLayer` instance checks rather than parameter-name prefix matching. Prefix-based approaches are brittle across PEFT versions and model families; instance-based detection is robust to changes in wrapper hierarchy.

**2. Scheduler warmup is never restarted after a switch.**  
When a mode switch changes the trainable parameter set, `AutoModeController.rebuild_optimizer_only()` constructs a new optimizer over the updated parameter list but preserves the scheduler's `_step_count` and base-LR by swapping the optimizer reference in place (`attach_optimizer_to_scheduler`). This prevents the sawtooth learning-rate pattern that arises when the scheduler is rebuilt from scratch on each switch.

**3. The LoRA delta is explicitly merged before promoting to base-update mode.**  
`promote_to_fft()` calls `module.merge()` before unfreezing base weights. Failing to merge means the adapter delta is silently discarded, and the "full fine-tuning" phase begins from an incorrect weight.

**4. Fresh adapter initialization on demotion.**  
`demote_to_lora()` calls `module.reset_lora_parameters()` to reinitialize `A ~ N(0, σ)` and `B = 0` after locking in the base weight. This ensures the LoRA delta starts from zero relative to the merged base, matching the paper's description.

**5. Non-target modules are frozen after PEFT wrapping.**  
`freeze_non_layer_params()` explicitly freezes `embed_tokens`, `lm_head`, and layer normalization weights. PEFT may mark these trainable depending on version; not freezing them inflates trainable parameter counts and introduces spurious differences between baselines.

**6. Importance accumulators are reset after each switching window.**  
`AutoModeController.maybe_switch()` calls `self._collector.reset()` at the end of each decision step. Scores from prior windows do not carry over into the next window's threshold computation.

**7. Results are reported as mean ± standard deviation across seeds.**  
All Tier 0 and Tier 3 configurations are evaluated across three seeds (8, 25, 42). Single-seed observations should not be treated as statistically conclusive.

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{automode2026,
  title     = {The Weights Must Flow: Dynamically Shifting Between LoRA and Base Updates for Efficient Fine-Tuning},
  author    = {Anonymous Authors},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

---

## License

License information will be added before public release.
