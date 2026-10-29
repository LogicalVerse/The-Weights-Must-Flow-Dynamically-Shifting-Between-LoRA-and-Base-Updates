# Test report (2026-10-29)

Every command below was actually executed. Unless stated otherwise it ran on the development machine: macOS 27.0.1 arm64, **no NVIDIA GPU**, CPython 3.11.15, torch 2.8.0 (CPU wheel), with the exact `requirements-rebuttal.lock` installed. **No GPU check was executed.** Nothing here is a model result. The tiny-model runs exercise plumbing on locally constructed toy models.

## CPU checks

| Command | Result |
|---|---|
| `python -m pytest -q tests` | **82 passed**, 3 warnings, 4.1 s. The warnings are the intentional scheduler-before-optimizer order in a legacy test, and PEFT noting that the tiny model has no hub config. |
| `python -m pytest -q tests/test_rebuttal_invariants.py` | **48 passed**, 1 warning, 3.0 s |
| `python tools/test_launcher.py` (kit supervisor + notebook/manifest consistency, run from the repo copy) | **Ran 7 tests, OK**, 7.6 s |
| `python -m automode.rebuttal contract` | `{"api_version": "1.0", "supported_methods": ["automode", "lora", "select_once", "fixed_mixed", "static_qv"], "metric_key": "eval_gsm8k_maj3"}`, exit 0. Importing for `contract` does not import torch (`test_contract_is_json_only_and_light`). |
| `validate_manifest(configs/rebuttal_core.json)` | 15 jobs; queues `[4, 4, 4, 3]`; 15 distinct run ids; `expected_eval_count` 1319; worker 3 = three LoRA r32/α64 jobs (seeds 8, 25, 42). |
| `AUTOMODE_TEST_DEVICE=cuda python -m pytest -q tests/test_rebuttal_invariants.py` (on this CPU-only machine) | **19 failed, 29 passed**: expected. It confirms that GPU-requested tests fail instead of skipping when CUDA is absent, so remote preflight cannot pass on a worker without working CUDA. |
| `uvx pyflakes automode/*.py tests/test_rebuttal_invariants.py` | Only unused-import notices, almost all pre-existing; one introduced regression (the dropped `dora` freeze) was found this way and fixed. |
| All code cells of `figures.ipynb`, `tier2_forgetting_evals.ipynb` and the launcher notebook | compile (launcher cells via `test_launcher.py`). |

### Worker bootstrap rehearsal (local, CPU)

This replays the notebook's deploy and bootstrap steps against a scratch "worker root", using **plain pip** as the notebook does:

1. Tar the working tree, then extract it to `worker0/repo`. Write `manifest.json` (notebook formatting) and `source_metadata.json` (the archive and manifest SHA-256 values; the commit is a placeholder of 40 zeros).
2. `python3.11 -m venv --system-site-packages venv`; install torch 2.8.0 CPU into it to stand in for the image's torch; record `torch_before.txt`.
3. `venv/bin/python -m pip install -r requirements-rebuttal.lock` → OK.
4. `venv/bin/python -m pip install --no-deps -e .` → OK.
5. `torch_after.txt`; `cmp torch_before.txt torch_after.txt` → **identical (2.8.0)**.
6. `venv/bin/python -m automode.rebuttal contract > contract.json` → contract JSON as above.
7. In the deployed tree: `venv/bin/python -m pytest -q tests` → **82 passed**.
8. `python3 remote_job.py start --job-dir worker0/jobs/preflight ... -- venv/bin/python -m automode.rebuttal preflight --manifest worker0/manifest.json --worker-id 0 --output worker0/results`:
   * supervisor status **failed** (non-zero exit);
   * `preflight_report.json` written with `status: failed` and `error: "RebuttalError: CUDA is not available"`. The installed-lock verification ran before the GPU check and passed.

### Tiny-model mechanics (toy models, not results)

* **All 15 manifest jobs** were run through `execute_queue` for workers 0–3, in-process. The loaders were replaced by a locally constructed 6-layer Gemma-2 (hidden size 32), a character tokenizer, 13 toy training items and 3 toy test items, with `expected_eval_count` set to 3 and `fixed_base_layers` set to 4 for the 6-layer model. All 15 rows were `completed`, and the requested exports were verified.
* The **figures notebook's code cells** were then executed against that output layout (with the 1319 assertion swapped for 3). They ran end to end:
  * exact-variant table;
  * seed-paired differences;
  * per-item 2×2 tables;
  * one distinct LR sequence across all 15 runs;
  * update-weighted occupancy; zero-demotion notes;
  * measured-cost table; 9 figure files.

### bf16 rounding check (supports CORRECTIONS D1)

For w ~ N(0, σ) stored in bf16, it applied one update of magnitude lr (random sign) and counted the elements that changed after rounding back to bf16:

| σ | lr = 2e-5 | lr = 1e-5 |
|---|---|---|
| 0.01 | 0.566 | 0.305 |
| 0.02 | 0.304 | 0.155 |
| 0.03 | 0.206 | 0.104 |

## GPU checks

**None executed** (no NVIDIA GPU available). The remote **preflight** enforces them:

* CUDA, a single visible device, bf16 support, a real bf16 matmul+backward, and VRAM ≥ 70 GiB;
* the full suite with `AUTOMODE_TEST_DEVICE=cuda`, including forward/backward and the end-to-end tiny jobs for all five methods on the GPU;
* lock verification; frozen HF revisions; data fingerprints; 1,319 test items; disk space.

The **pilot** then runs every assigned method on Gemma-2-9B for a few updates, plus the 512-example timing pilot with export verification.

## Not executed

* Any SSH connection, rental machine, or GitHub push.
* Gated Hugging Face access or download of Gemma-2-9B / GSM8K.
* Real training, evaluation, export, timing, or memory measurement.
* Pilot or core runs.
