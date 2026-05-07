# Changelog

## v0.2.1 — Second validator pass (2026-04-17)

Fifth-round review caught five residual issues, all of which are now fixed.
No new behavior — this release is entirely hygiene and honesty.

### Fixes

**1. Reproducible GPU sharding** (`grid.py::shard_grid_by_gpu`)
- **Before**: Fallback hash sharding used Python's built-in `hash(run_id)`,
  which is salted per-process by default (PEP 456). The same run set would
  land on different GPUs across shell restarts or machines — silently
  breaking resume behavior when the explicit `DEFAULT_GPU_ASSIGNMENT` wasn't
  used.
- **After**: `int(c.run_id(), 16) % n_gpus`. The `run_id()` is already MD5
  hex and stable across processes. Docstring updated.

**2. Gumbel-Softmax no longer claimed to be runnable** (`core.py`, `grid.py`)
- **Before**: The `GumbelGate` block comment said the path was "wired as a
  separate method `automode_gumbel`"; `build_tier3` docstring listed
  "continuous gating work? → automode_gumbel single run" as an ablation
  item. Neither was true — there is no `automode_gumbel` in `METHODS`, no
  builder in `build_model_for_method()`, and `AutoModeController` explicitly
  asserts percentile-only switching.
- **After**: `GumbelGate` docstring now says "PREVIEW MODULE, NOT A RUNNABLE
  METHOD IN v0.2.0" with explicit instructions for the three changes a user
  would need to make to wire it up. `build_tier3` docstring removed the
  Gumbel bullet and added a note that continuous-gated variants are deferred.
  The grid module's top-level overview no longer mentions Gumbel.

**3. Smoke-test instructions now reference both test suites** (`README.md`,
  all three driver notebooks)
- **Before**: README setup and driver notebooks ran only
  `pytest tests/test_switching.py -v` — missing the scheduler continuity
  and gradient-accumulation tail tests added in v0.2.0.
- **After**: README and all three driver notebooks now run
  `pytest tests/test_switching.py tests/test_training_invariants.py -v`
  as their mandatory pre-training gate.

**4. BitFit comment matches the strict implementation** (`config.py::METHODS`)
- **Before**: Dictionary comment said "update only bias + lm_head".
- **After**: Comment says "update bias parameters only (strict — fails on
  bias-free models)", matching the v0.2.0 rewrite of `apply_bitfit()`.

**5. Headline figures filter to paper-faithful baselines** (`figures.ipynb`)
- **Before**: The notebook computed `df_ok = df[df['status'] == 'ok']` and
  used it for everything, mixing faithful baselines with approximate ones in
  the same main results table, bar chart, Pareto plot, and significance
  tests. Reviewer-facing plots could accidentally claim AutoMode "beats
  LISA" when LISA is implementation-approximate per our own
  `METHOD_FIDELITY` labels.
- **After**: New `df_faithful_ok` and `df_approx_ok` dataframes, split by
  `paper_fidelity` column. Headline plots (bar chart, Pareto, main results
  table, significance tests) use `df_faithful_ok`. New supplementary cell
  writes `paper/supplementary_approximate_baselines.csv` separately for
  approximate methods. Pre-v0.2.0 CSVs without a `paper_fidelity` column
  fall back to the old behavior with a printed warning.

### Unchanged

No code-behavior changes — v0.2.1 is purely documentation, sharding
determinism, and plotting hygiene. If your v0.2.0 smoke runs passed, v0.2.1
is a drop-in replacement with no retraining needed.

---

## v0.2.0 — Validator-driven refinement (2026-04-17)

This release addresses every concern raised in an external validator review
conducted against the four primary source papers (AdaLoRA, LISA, LoRA-GA,
AdaGradSelect). Every change below is a fix, not a new feature.

### Critical fixes (blocked execution)

**Gradient-accumulation tail flush** (`automode/train.py`)
- **Before**: `ceil(len(train_loader) / grad_accum_steps) * epochs` counted
  partial final windows in the scheduler's `total_opt_steps`, but the inner
  loop only produced an optimizer step on `(micro_step+1) % grad_accum == 0`.
  The final partial window was silently dropped, misaligning the scheduler
  and AutoMode switching cadence.
- **After**: `total_opt_steps = full_windows + (1 if has_tail else 0)`, and
  a new `_do_optimizer_step(window_size, epoch)` helper triggers on the last
  micro-batch of an epoch if leftover grads exist. Gradients are rescaled by
  `grad_accum / window_size` so the tail step matches full-window semantics.
  Log output now includes `window_size` per step.
- **Test**: `tests/test_training_invariants.py::TestGradAccumTailFlush`.

**`cfg.switching_mode` silently ignored** (`automode/core.py`)
- **Before**: The `RunConfig` schema exposed `switching_mode ∈ {percentile,
  topk_fixed, gumbel, static}`, but `AutoModeController.maybe_switch()` always
  used percentile thresholding. Any non-percentile setting was a silent no-op.
- **After**: `AutoModeController.__init__` raises `NotImplementedError` if
  `switching_mode != "percentile"`, with a clear message pointing at the
  reserved-but-unwired modes.
- **Test**: `TestAutoModeConfigAssertions`.

### Paper-fidelity fixes

**LISA baseline** (`automode/train.py::LisaController`) — cross-check: Pan et
al. 2024 (2403.17919v4) Algorithm 1 and §3.2
- **Before**: Kept `lm_head` trainable but not `embed_tokens`, silently
  deviating from the paper's E+H+γL recipe (the whole thesis of LISA is that
  the weight-norm imbalance starts at the *embedding* layer).
- **After**: `_keep_embed_and_lm_head_trainable()` unfreezes both across the
  three common hierarchies (`.embed_tokens`, `.model.embed_tokens`,
  `.base_model.model.embed_tokens`). Switch history records `always_trainable`.

**AdaGradSelect baseline** (`automode/train.py::AdaGradSelectController`) —
cross-check: Kumar et al. 2025 (2512.15764v1) Algorithm 2
- **Before**: Explore/exploit branches were swapped. Our code did Dirichlet
  sampling in the "explore" branch and top-k-by-gradient in the "exploit"
  branch. The paper says the opposite: explore = top-k by gradient norm
  (greedy, informed), exploit = Dirichlet sampling based on historical
  update-frequency prior.
- **Before**: ε decay used a step-threshold cutoff (`opt_step < total//2`).
  The paper uses exponential decay `ε_t = ε_0 · exp(-λt)`.
- **After**: Branches swapped to match paper. ε now decays exponentially with
  λ chosen so ε drops ~20× over one epoch's worth of switches. Epoch-1-only
  exploration is now properly enforced.
- `switch_history` now records `mode` (`explore_topk_grad` /
  `explore_cold_uniform` / `exploit_dirichlet`) and `epsilon_at_switch`.

**BitFit baseline** (`automode/models.py::apply_bitfit`)
- **Before**: Fell back to "all 1D parameters" if no `.bias` tensors found,
  and also unfroze `lm_head`. Neither matches the strict Ben Zaken et al.
  2021 formulation, and would confuse reviewers.
- **After**: Strict bias-only. Raises a clear error on models without biases
  (e.g., Gemma/LLaMA families frequently have `bias=False`), so the method
  doesn't silently become "train LayerNorm + LM head" in disguise.

### Design decisions documented (not behavior changes)

**Static baselines + lm_head** (`automode/models.py::apply_topk_*`)
- Added `keep_lm_head_trainable: bool = True` to `RunConfig`. Docstrings now
  state the default and link it to LISA/AdaGradSelect convention. Reviewer
  objections can be addressed by setting this to `False` for strict "layers-
  only" variants.

### Instrumentation

**Per-method fidelity labels** (`automode/config.py::METHOD_FIDELITY`)
- Every method is labeled `faithful`, `approximate`, or `preview`. Each result
  row written to the CSV now includes `paper_fidelity`, so grouped summary
  tables can be filtered to paper-faithful baselines for the headline, and
  approximate baselines for the supplementary.

**Cleaner result schema** (`automode/train.py::_relevant_config_fields`)
- Method-irrelevant config fields are written as `None` instead of carrying
  unused defaults. A `full_ft` row no longer reports `lora_r=16`; an
  `adagradselect` row no longer reports `dynamic_updates=10`. Summary
  groupings are now meaningful without manual filtering.

**Tier-0 lineup tightened** (`automode/grid.py::_core_methods_for_cell`)
- Tier-0 headline now only includes the 6 paper-faithful methods:
  `full_ft`, `lora r=16`, `lora r=32`, `automode u=6/t=10`, `automode u=10/t=10`,
  `topk_deep_block`. LISA, AdaGradSelect, AdaLoRA, DoRA, LoRA-GA moved to
  Tier 4 supplementary.

### Testing

**New test file**: `tests/test_training_invariants.py`
- `TestSchedulerContinuity.test_lr_continuous_across_optimizer_rebuild`
- `TestSchedulerContinuity.test_scheduler_internal_step_counter_preserved`
- `TestGradAccumTailFlush.test_opt_step_count_matches_ceil`
- `TestGradAccumTailFlush.test_tail_window_size_is_remainder`
- `TestGradAccumTailFlush.test_grad_rescale_factor_math`
- `TestGradAccumTailFlush.test_no_tail_when_divisible`
- `TestAutoModeConfigAssertions.test_switching_mode_*`
- `TestMethodFidelity.test_all_methods_have_fidelity_label`

### Smoke-test protocol (validator recommendation)

Before launching any real grid, users should:

1. Run `pytest tests/test_switching.py tests/test_training_invariants.py -v`
2. Kick off ONE small `automode` run on 32 samples, then inspect:
   - `logs/lr_schedule.json` — LR should be monotone post-warmup, no sawtooth
   - `logs/trainable_params.json` — count should change at switch points
   - `dynamic/switch_history.json` — non-empty, with reasonable threshold/mode
3. Kick off ONE small `full_ft` and ONE small `lora` run for comparison
4. Only then launch `run_grid()`.

The three driver notebooks now include these smoke cells at the top.

### Items explicitly NOT fixed (documented as limitations)

- **Gumbel-Softmax continuous gate**: `core.py::GumbelGate` exists as a
  module but there is NO `automode_gumbel` method in `METHODS`. It is not
  runnable via the grid and will not appear in any results. This is
  intentional — the Gumbel path is deferred to a follow-up paper.

- **LoRA-GA upstream dependency**: If the LoRA-GA fork isn't installed,
  `apply_loraga` falls back to standard LoRA with a warning. Results for
  `loraga` method should be dropped from the paper unless the fork is
  confirmed installed (check `tests/` passes for it, or install from
  <https://github.com/Outsider565/LoRA-GA>).

- **MT-Bench evaluator**: Not implemented. MT-Bench requires a GPT-4 judge,
  which has cost/reproducibility issues. The paper's benchmark set is
  GSM8K, MATH, MMLU, ARC-C.

- **Catastrophic forgetting framing**: Tier-2 MMLU/ARC-C evals measure
  post-training benchmark drop. The paper should phrase this as "consistent
  with overfitting or forgetting" rather than "proven catastrophic
  forgetting" — Tier 2 doesn't directly measure forgetting of pretraining
  distributions.

---

## v0.1.0 — Initial rebuild (2026-04-17)

First clean-room rebuild from the original 2B and 9B notebooks. See
README.md for the list of bugs fixed from those notebooks.
