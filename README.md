# AutoMode: Dynamic Update-Mode Allocation for Efficient Fine-Tuning

Code for *"The Weights Must Flow: Dynamically Shifting Between LoRA and Base Updates for Efficient Fine-Tuning"*, AAAI-27 submission #44087 (under review; not published anywhere). The repository contains the AutoMode training framework, the controls used for the rebuttal, evaluation code, tests, and the original experiment grid (kept as **legacy**).

AutoMode trains each targeted projection (`q_proj`, `v_proj`) either through a LoRA adapter or directly in its base weight, and reassigns that mode per layer during training from a gradient-activity score.

**Read [CORRECTIONS.md](CORRECTIONS.md) before using any number produced by earlier versions of this code.** Several bugs affected the submitted results; the rebuttal campaign uses a corrected protocol and fresh outputs, and does not reproduce or overwrite old numbers.

---

## The rebuttal campaign (15 core runs)

Gemma-2-9B base, GSM8K, five methods × seeds 8 / 25 / 42, four independent single-GPU workers (no DDP/FSDP).

| Worker | Jobs (manifest `configs/rebuttal_core.json`) |
|---|---|
| 0 | seed 8: `automode`, `select_once`, `fixed_mixed`, `static_qv` |
| 1 | seed 25: same four |
| 2 | seed 42: same four |
| 3 | `lora` r32, seeds 8, 25, 42 |

The question is whether **continued** dynamic mode assignment beats comparable static and select-once controls. It is a correction-and-control campaign, not test-score maximisation: no hyperparameter is chosen on test data.

### Method definitions (exact)

All five use `q_proj`/`v_proj` only. Embeddings, LM head, norms and all other projections are frozen. They share the same optimizer and schedule: AdamW (torch), LR 2e-5, linear warmup (3%) and decay, weight decay 0.01, clip 1.0, batch 4 × accumulation 8, 2 epochs, bf16 model, eager attention. The invariant is asserted at start and after every switch (`assert_trainable_scope`).

- **`automode`** — every layer starts in LoRA mode (r16, α32, dropout 0.05). Every `update_interval = (total_updates / epochs) // 10` optimizer updates (u = 10 per epoch), scores are computed and the lower-tail rule is applied: `threshold = np.percentile(scores, 10)` and layers with `score >= threshold` train base weights. All other layers train LoRA. A layer may be promoted, demoted and re-promoted. No decision is made after the final update.
- **`select_once`** — identical to `automode` (same initialisation, data order, LR, and the same first decision) until its **first scheduled selection**. That allocation is then held for the rest of training; low-score layers keep training through LoRA. The branch step is logged (`branch_step`).
- **`fixed_mixed`** — from update 1, a fixed set of 37 of the 42 layers trains q/v base weights and the other 5 train q/v LoRA r16/α32. The set is drawn once by `random.Random(allocation_seed).sample` (manifest `allocation_seed` = run seed), independent of every training RNG and of loss or test data. The exact layer IDs are written to `dynamic/allocation.json`. This matches AutoMode's post-selection instantaneous budget (37 base layers at t = 10 with no ties), not necessarily its time-integrated budget; both are reported.
- **`static_qv`** — q/v base weights of all 42 layers train throughout, with no adapters. Its trainable budget is *larger* than the 37-layer allocation; it is not budget-matched.
- **`lora`** — q/v LoRA r32, α64, dropout 0.05, no switching.

**Promotion** merges the current adapter exactly once (`θ ← θ + (α/r)BA`, PEFT `merge(safe_merge=True)`), freezes the adapter and trains the base projection. **Demotion** keeps the fully trained base weight (it never unmerges, because the base has had direct updates since the merge), clears PEFT's merged bookkeeping, and re-initialises the adapter to a zero delta. The adapter is then trained alone. Adapter initialisation, both at creation and on demotion, is **A ~ Uniform(−1/√d_in, 1/√d_in)** (kaiming-uniform, a = √5, as in PEFT 0.15.2 `init_lora_weights=True`) with **B = 0**. It is not Gaussian. Promotion and demotion preserve the eval-mode function. They do not preserve training-mode behaviour exactly, because LoRA dropout acts only on the adapter path, which does not exist in base mode.

**Score** (`importance_signal=grad_norm`, `score_collection=optimizer_step`): per layer, `S_l = sqrt(mean of squared gradient components)` over that layer's currently trainable q/v parameters. It uses the completed accumulation-window gradient (after tail rescaling, before clipping), sampled once per optimizer update and averaged over the switching window. Adapter-mode and base-mode layers are scored over different parameter sets, so the score is **not** invariant across modes or under LoRA gauge rescaling (A → cA, B → B/c changes it; see `test_rms_score_is_not_invariant_to_lora_gauge_rescaling`). Ties at the threshold are all selected, and the actual count is logged per event (`n_selected_base`, `n_tied_at_threshold`).

### Corrected protocol (summary; details in CORRECTIONS.md)

- **Data:** GSM8K train (all 7,473 items) with the submitted prompt template unchanged. Prompt and target are tokenized separately, `" " + target + EOS`, and the loss mask covers exactly the target tokens. Over-long prompts are left-truncated keeping BOS; truncation and empty-target counts are logged and nothing is dropped. Data order comes from a dedicated generator seeded by `data_seed`, identical across methods.
- **Evaluation:** all 1,319 GSM8K test questions, `sampling_k = 3`, T = 0.7, top-p = 0.95, 256 new tokens. Generation is seeded per batch by `evaluation_seed`, independent of training RNG. Extraction takes the number after the first `####`, else the last number; amounts are normalised numerically. The vote is a plurality in which invalid samples count as one "invalid" option, and ties go to the first-seen option. Metric `gsm8k_maj3`, result key **`eval_gsm8k_maj3`**. Every response, extraction, gold answer, item ID and correctness flag is stored.
- **Identity:** each job's identity hashes every outcome-relevant setting, the resolved model and dataset revisions, the protocol version, the commit, a hash of the code, and the extraction and template versions. A job is skipped only when its `COMPLETE.json` carries exactly that identity and its artefacts verify.

### Open decision before launching: bf16 base updates

The manifest keeps the submitted precision. The model and its trainable q/v base weights are bf16, and AdamW updates them in place with bf16 optimizer state. At LR ≈ 2e-5, an Adam step is far smaller than bf16's spacing for typical weight magnitudes, so most base-weight updates round to no change. A synthetic check (CORRECTIONS.md) shows only 20–30% of elements changing per step for |w| ~ 0.02–0.03, and fewer as the LR decays. This affects every method that trains base weights (`automode`, `select_once`, `fixed_mixed`, `static_qv`), not LoRA. The training loop measures the real effect (`base_update_changed_fraction_mean` in each result and in the pilot report). `base_weight_dtype: "float32"` keeps fp32 master copies of the q/v base weights and runs the forward pass under bf16 autocast. It is implemented and CPU-tested, but it is **not** enabled in the supplied manifest. Enabling it is a protocol change that changes every job identity. Decide after reading the pilot report.

### Step-by-step: running the rebuttal campaign

Everything is driven from your laptop by `notebooks/AutoMode_Four_GPU_Colab.ipynb`; no local GPU is needed. Section numbers below refer to the notebook.

**On your laptop, once**

1. Check the code locally (Python 3.11; Python 3.14 has no torch 2.8 wheel):
   ```bash
   uv venv --python 3.11 .venv && source .venv/bin/activate
   uv pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
   uv pip install -r requirements-rebuttal.lock && uv pip install --no-deps -e .
   python -m pytest -q tests && python tools/test_launcher.py    # expect 82 passed, 7 OK
   ```
   Use `python -m pytest`, not bare `pytest`, so a conda `pytest` cannot shadow the venv's.
2. Push, then copy the full commit SHA: `git rev-parse HEAD`. The notebook deploys that exact commit, not a branch.
3. Accept the Gemma-2 licence on Hugging Face (`google/gemma-2-9b`), and have a **read-only** HF token ready.

**Rent and prepare the GPUs**

4. Rent four single-GPU Linux machines (ideally 96 GB RTX PRO 6000; ≥ 70 GiB VRAM required) with a CUDA-enabled PyTorch image, Python ≥ 3.10, and ~200 GB disk each.
5. SSH into each machine once from your terminal, check the host fingerprint against the provider's, and accept it into `known_hosts`. The notebook never bypasses host-key checking. Load your SSH key into your agent (`ssh-add`), or set `key_path` per worker.

**In the notebook (Jupyter, or Colab on a local runtime)**

6. **§1 settings:**
   - set `REPO_URL` (the same URL form you clone with, e.g. `git@github.com:LogicalVerse/The-Weights-Must-Flow-Dynamically-Shifting-Between-LoRA-and-Base-Updates.git`);
   - set `COMMIT` to the 40-character SHA;
   - fill in `host`, `port`, `user` and `gpu` for the four `WORKERS`;
   - leave `LAUNCH_CORE = False`.
7. Run **§2–§3**: helpers, manifest, fetch the commit, check SSH connections.
8. Run **§4** to deploy, pasting the HF token at the hidden prompt. Never put the token in a cell.
9. Run **§5** (bootstrap). Watch it with **§9** (`PHASE="bootstrap"`) until all four are `completed`. It fails if installing the lock would change torch.
10. Run **§6** to start preflight; wait with §9 (`PHASE="preflight"`).
11. Run **§7**. It checks all four preflight reports (tests passed on GPU, same revisions, lock and torch, distinct GPU UUIDs), then starts the pilots. Wait with §9 (`PHASE="pilot"`).
12. Run **§8** and **read the four pilot reports** before going further:
    - `estimated_core_queue_hours` must fit in `CORE_BUDGET_HOURS` (14 h) with 15% to spare;
    - workers 0–2 need `post_promotion_optimizer_steps` ≥ 5;
    - check `base_update_changed_fraction_mean` for the bf16 decision below.
13. **bf16 decision.** If the pilot shows that few base weights actually change per step, switch to fp32 master weights:
    - add `"base_weight_dtype": "float32"` to `common` in **both** `configs/rebuttal_core.json` and the notebook's `CORE_MANIFEST` (§2);
    - commit, push, and put the new SHA in §1;
    - redeploy under a **new** `CAMPAIGN` name and redo steps 7–12.

    Do not mix outputs from the two settings.
14. Launch the core run: execute `LAUNCH_CORE = True` in a new cell, then rerun §8. It starts exactly the 15 jobs.
15. Monitor with §9 (`PHASE="core"`). The jobs run detached, so closing the notebook doesn't stop them. To reconnect, rerun §1–§2 with identical values.

**Afterwards**

16. Run **§10** to download logs, predictions and reports (you can also do this mid-run). Run **§11** to validate the downloads and summarize them.
17. Copy the full exports (seed-8 jobs, `core/<job>/attempts/<n>/export/`, including `adapter/` and `layer_state.json`) off workers 0 and 3. The download does not include weights.
18. Analyse with `notebooks/figures.ipynb`, pointed at the download.
19. Only then release the rentals through the provider. Stopping jobs (§12) does not stop billing.

**If something fails:** read the job log via §9, fix the cause, push a new commit, and redeploy under a new `CAMPAIGN`. To rerun the same code, use `launch(..., retry=True)`; verified-complete jobs are skipped, everything else starts a fresh attempt directory.

### Running it

Use `notebooks/AutoMode_Four_GPU_Colab.ipynb` (local Jupyter or Colab on a local runtime) against an **exact pushed commit**. It deploys the snapshot to each worker, then runs bootstrap → preflight → pilot → core, each detached under `tools/remote_job.py`. The worker-side commands (the API in [RUNNER_CONTRACT.md](RUNNER_CONTRACT.md)) are:

```bash
python -m automode.rebuttal contract
python -m automode.rebuttal preflight --manifest FILE --worker-id N --output DIR
python -m automode.rebuttal pilot     --manifest FILE --worker-id N --output DIR
python -m automode.rebuttal run       --manifest FILE --worker-id N --output DIR --hours 14
```

- **preflight**:
  - verifies the source metadata, the manifest (schema 1, 15 distinct jobs, full coverage) and that the installed packages match `requirements-rebuttal.lock`;
  - checks CUDA with exactly one visible device, bf16 support, a real bf16 kernel launch, and VRAM ≥ `minimum_vram_gib`;
  - runs the full test suite on the GPU (`AUTOMODE_TEST_DEVICE=cuda`);
  - resolves and freezes the Hugging Face model and dataset revisions to commit SHAs, downloads and fingerprints them, checks there are 1,319 test items, and checks free disk;
  - writes `resolved_manifest.json` and `preflight_report.json`.

  Later phases reuse the frozen revisions and refuse a different manifest or source.
- **pilot**:
  - runs a tiny train+eval of every assigned method on the real model;
  - runs the main pilot on 512 train examples, 1 epoch, 100 eval questions (AutoMode with u = 2 on workers 0–2, LoRA r32 on worker 3), including the full export if the worker's queue requests one; the exported weights are deleted after verification;
  - writes `pilot_report.json`: post-promotion step time, peak memory, generation throughput, base-update efficacy, and a conservative queue ETA (× 1.25).

  The pilot score is not a result.
- **run** executes only the worker's assigned jobs, sequentially, each in a fresh subprocess and a fresh attempt directory, under a per-worker lock and a per-run lock. It does not start a job whose pilot-based estimate exceeds the remaining `--hours` (marked `budget_exhausted`). `run_ledger.json` is rewritten atomically after every change. Exit code is 0 only if every job is verified complete.

**Retry.** Failures are never retried automatically and batch sizes are never changed. To retry, relaunch the phase explicitly. In the notebook use `launch(..., retry=True)`; the supervisor refuses to relaunch a failed job without `--retry`. Verified jobs are skipped. Every other job gets a **new** attempt directory, and earlier failed or interrupted attempts and their logs are kept. Interrupted runs restart from scratch: there is no mid-run resume.

**Stop.** Run the notebook's stop cell, or `python3 WORKER_ROOT/remote_job.py stop --job-dir WORKER_ROOT/jobs/core`. On SIGTERM the runner forwards the signal to the job, which records `interrupted` and writes no success marker; the ledger records the interruption. Stopping a job does not stop rental billing.

**Export.** Jobs with `save_model: true` (all seed-8 jobs) write a **full inference export** to `core/<job>/attempts/<n>/export/`. This happens after all evaluation. PEFT models are saved **unmerged**: the learned base as a normal HF checkpoint, every adapter in `adapter/`, and each layer's merged state in `layer_state.json`. Load them with `automode.export.load_unmerged_export(export_dir, device)`. Loading the root alone gives the base without the unmerged adapters. `static_qv` has no adapters and is a plain checkpoint. The export is reloaded and its logits are compared with the live model on fixed prompts (`export_metadata.json`, `checks.reload_vs_live`, limit 1e-4). A merge is then audited on the live model and reported separately (`checks.merge_vs_live`, `merge_audit`); it is never saved. It is **not** a resumable training checkpoint. The notebook's download excludes weights; copy exports off the workers before releasing them.

**Disk.** Gemma-2-9B bf16 weights are ~18.5 GB, and each full export is about the same size. Preflight measures the snapshot and requires at least `(exports + 1) × snapshot + 10 GiB` free after download. With ~17.2 GiB of weights, that is ≈ 96 GiB for worker 0 (4 exports), ≈ 44 GiB for worker 3 (1 export) and ≈ 27 GiB for workers 1–2, on top of the ~17 GiB HF cache. Provision ~200 GB per worker, as the kit recommends.

**Results.** Download with the notebook (section 10), then use `notebooks/figures.ipynb`. It groups by exact variant, track and protocol, pairs by seed and `item_id`, shows every seed, reports the seed SD descriptively (three seeds support no significance claims), uses update-weighted layer occupancy from `switch_diagnostics.json`, and reports measured time and memory separately from parameter counts.

**Not launched automatically** (each needs a separate plan): random same-count dynamic selection, untouched-base evaluation, a full-FT LR sweep selected on a fixed *training* holdout and then confirmed separately, and retention evaluation (`notebooks/tier2_forgetting_evals.ipynb`, verified exports only). LISA, AdaGradSelect, LoRA-GA, AdaLoRA and `dyn_full_only` are **unavailable** (they raise) until revalidated. They are never substituted as controls.

### Per-run outputs

```
core/<job_id>/COMPLETE.json                 identity + attempt; written last, only after verification
core/<job_id>/attempts/attempt_NNN/
  spec.json status.json job.log result.json environment.json
  configs/run_config.json
  logs/{training_loss,lr_schedule,grad_norms,trainable_params,base_update_probe}.json   (flushed during training)
  dynamic/{switch_history,layer_timeline,switch_diagnostics,allocation}.json
  evals/gsm8k_eval.json                     every sample, extraction, vote, gold, item_id
  export/                                   full inference export: base, adapter/, layer_state.json, export_metadata.json (seed-8 jobs)
```

`switch_diagnostics.json` gives:

- the first selected set and the actual count at every decision event;
- promotions, demotions and re-promotions per layer, with explicit denominators and a note when there were no demotions;
- the fraction of first-selected layers that stayed in base mode for the rest of training;
- the Jaccard overlap between successive selections;
- update-weighted base/LoRA occupancy, with the initial mode reported separately;
- score distributions split by each layer's mode before selection;
- the optimizer-step-weighted trainable count.

---

## Tests

```bash
pip install -r requirements-rebuttal.lock && pip install --no-deps -e .   # torch from your environment
pytest -q tests                                   # CPU: tiny local Gemma-2 + PEFT, no downloads
AUTOMODE_TEST_DEVICE=cuda pytest -q tests         # GPU variant; preflight runs this and fails without CUDA
python tools/test_launcher.py                     # process supervisor + notebook/manifest consistency
```

`tests/test_rebuttal_invariants.py` covers:

- exact per-step LR sequences with several forced rebuilds, including during warmup (and a check that it catches the old base-LR bug);
- full param-group preservation;
- Adam state carried over by identity, and cleared on leave, reset or activation;
- eval-function preservation through promote → base training → demote, with no unmerge and active adapter gradients afterwards;
- uniform adapter initialisation;
- a full export round trip, versus the adapter-only save that loses base updates;
- exact loss masks, padding, EOS, and truncation and empty-target accounting;
- seeded data order;
- all five allocations, including forward/backward on the requested device;
- that unavailable methods raise;
- the fp32-master path;
- the select-once branch (equal pre-branch trajectory) and no terminal switch;
- that `optimizer_step` scoring counts each completed window once;
- deterministic, logged score ties;
- gauge non-invariance;
- GSM8K extraction and voting rules, records, and seeding;
- run-id coverage of every scientific field;
- the CLI contract and manifest validation;
- refusal of foreign output directories;
- the dependency lock;
- end-to-end tiny jobs for all five methods with a verified export;
- that failures and interruptions never write COMPLETE;
- ledger, skip, budget and interruption behaviour;
- switch diagnostics and the seed/item pairing helpers.

The pre-existing `test_switching.py` and `test_training_invariants.py` were updated to the corrected API. See [TEST_REPORT.md](TEST_REPORT.md) for what was actually run and [RUNTIME.md](RUNTIME.md) for the validated software combination.

---

## Repository layout

```
automode/
  config.py      RunConfig (all fields hashed into run_id except output_root/device/notes/tier), method lists
  core.py        layer grouping, score collectors, strict promote/demote, rebuild_optimizer, AutoModeController
  models.py      model loading (pinned revision, eager attention), method builders incl. fixed_mixed/static_qv
  data.py        prompt/target construction and masking, collator, seeded epoch sampler
  train.py       training loop (timing, memory, base-update probe, logs), legacy run_experiment
  eval.py        GSM8K maj@k, MATH, MMLU/ARC with their own context budget
  export.py      verified full inference export
  analysis.py    switch diagnostics, exact-variant summaries, seed/item pairing
  rebuttal.py    campaign CLI (contract / preflight / pilot / run)
  grid.py        LEGACY tiered grid (not used by the rebuttal)
configs/rebuttal_core.json        core manifest (supplied)
notebooks/AutoMode_Four_GPU_Colab.ipynb   SSH launcher (supplied)
notebooks/figures.ipynb           rebuttal tables/figures
notebooks/tier2_forgetting_evals.ipynb    optional retention eval on verified exports
notebooks/driver_gpu*.ipynb       LEGACY grid drivers
tools/remote_job.py               detached single-job supervisor (supplied); tools/test_launcher.py its tests
requirements-rebuttal.lock        exact non-CUDA dependency closure;  RUNTIME.md  validated runtime
RUNNER_CONTRACT.md  CORRECTIONS.md  TEST_REPORT.md
```

---

## Legacy experiment grid

`automode/grid.py` and the `driver_gpu*.ipynb` notebooks build and run the original tiered grid (108/222 runs across several models). They are preserved for reference and are **not** part of the rebuttal; do not launch them for rebuttal results. They now share the corrected core: run ids changed (old outputs are never treated as complete), only verified COMPLETE markers are skipped, evaluation and export failures produce failed rows, and unavailable methods raise. Single runs: `from automode.train import run_experiment; run_experiment(RunConfig(...))`.

---

## Citation

Anonymous submission under review at AAAI-27. Please do not cite as a published paper.

## License

License information will be added before public release.
