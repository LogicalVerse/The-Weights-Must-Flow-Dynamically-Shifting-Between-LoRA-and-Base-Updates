# AutoMode rebuttal launcher API 1.0

The notebook supplies a deployment of one immutable GitHub commit to each worker. It expects Claude's changes to implement this contract. The current uploaded package does not implement it. Do not bypass the gates to run the old code.

## Files the repository must contain after Claude's changes

- `automode/rebuttal.py` implementing the commands below.
- `requirements-rebuttal.lock`: exact tested non-CUDA dependencies, including pytest. GPU image supplies PyTorch/CUDA. Include transitive dependencies so pinning is reproducible.
- `tests/test_rebuttal_invariants.py` plus existing applicable tests.
- `pyproject.toml` with an editable-installable automode package.

The notebook sends the supplied manifest separately; it does not require a copy in the GitHub repo. It sends `source_metadata.json` next to the remote repo (commit, archive SHA256, manifest SHA256). Commands should verify these and include them in their reports.

## Commands

All paths are absolute. N is an integer 0–3. Every command exits zero only on success; logs may go to stderr. `contract` must emit only one JSON object on stdout.

`python -m automode.rebuttal contract`
```json
{"api_version":"1.0","supported_methods":["automode","lora","select_once","fixed_mixed","static_qv"],"metric_key":"eval_gsm8k_maj3"}
```

`python -m automode.rebuttal preflight --manifest FILE --worker-id N --output DIR`

Validate manifest schema 1, common configuration, 15 distinct job IDs, assignment, hardware, access and tests. Resolve model/dataset revision identifiers and freeze them in `DIR/resolved_manifest.json`. Reject different source/manifest identity in an existing DIR. Persist `preflight_report.json` only with status ok when everything succeeds. On failure, write status failed and exit nonzero. Required successful report fields:
```json
{"status":"ok","api_version":"1.0","worker_id":0,"commit":"40-hex-sha","manifest_sha256":"sha256","archive_sha256":"sha256","model_revision":"resolved-hf-sha","dataset_revision":"resolved-hf-sha","gpu_uuid":"GPU-...","gpu_name":"...","vram_gib":95.0,"torch_version":"...","dependency_lock_sha256":"sha256","tests_passed":true}
```
The revisions in the manifest may initially be null. Resolve them consistently, then pilot/run MUST load the saved resolved revisions rather than follow main again. Report dataset split fingerprint and software/environment details as additional fields.

`python -m automode.rebuttal pilot --manifest FILE --worker-id N --output DIR`

Require successful matching preflight. Pilot 512 training examples, 1 epoch, 100 evaluation questions. Use AutoMode u2 on workers0–2 and LoRA-r32 on worker3. Test allocations of all assigned methods on a small GPU batch. This must not write a core COMPLETE marker. Use `DIR/pilot/`. Required `pilot_report.json`:
```json
{"status":"ok","worker_id":0,"commit":"40-hex-sha","manifest_sha256":"sha256","n_eval":100,"peak_vram_gib":35.0,"post_promotion_optimizer_steps":8,"estimated_core_queue_hours":9.0,"estimate_is_approximate":true}
```
Workers0–2 must demonstrate at least five post-promotion optimizer steps. Worker3 sets post_promotion_optimizer_steps to null. Include raw timing inputs, model/data revisions, successful tiny method checks and conservative assumptions used for ETA. Numbers above are schema examples, NOT measured results.

`python -m automode.rebuttal run --manifest FILE --worker-id N --output DIR --hours H`

Require matching successful preflight and pilot. Execute only assigned core runs, sequentially, each in a separate process and fresh directory. `H` is a wall-time scheduling budget, not permission to change epoch count. Do not start a run estimated to exceed remaining time. Write a partial campaign status if not all jobs fit. Exit nonzero on an experimental failure or incomplete campaign; do not fabricate successful completion.

`DIR/run_ledger.json` must be an atomic object:
```json
{"schema_version":1,"worker_id":0,"commit":"40-hex-sha","manifest_sha256":"sha256","runs":[{"job_id":"s8_automode","method":"automode","seed":8,"status":"completed","eval_gsm8k_maj3":0.0,"eval_gsm8k_n_total":1319,"train_time_sec":0.0,"eval_time_sec":0.0,"peak_vram_gib":0.0,"result_dir":"core/s8_automode"}]}
```
Rows may be pending/running/completed/failed/interrupted/budget_exhausted. Completed rows require verified metrics/counts, exact identity and complete requested artifacts. Sample values above are schema placeholders. Do not write zeros as missing measurements. Failure rows include an error and log path. The notebook merges completed rows by unique job_id and rejects mismatches; it does not pool variants.

## Manifest interpretation

Merge common values with each run's overrides, then set output/device locally. Manifest `common` maps to RunConfig fields including new explicit revision/score/seed fields; implement them without ignoring unknown keys. `jobs[].worker_id` determines assignment. `expected_eval_count` is validation metadata, not a training field. `save_model=true` means the validated full inference export for AutoMode/control methods.

`minimum_vram_gib` is a preflight threshold for this intended rental plan, not a universal minimum model requirement. Do not silently lower it. Each process sees exactly one physical device through CUDA_VISIBLE_DEVICES and uses cuda:0 internally.

## Optional work

The launcher only schedules the 15 core runs. Full-FT tuning, random allocation, untouched-base evaluation and retention require explicit separate plans after the core; the main button does not start them. No GPUs are rented or terminated by this kit. Stopping a process does not stop rental billing.
