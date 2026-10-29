# Runtime declaration

## What was actually validated (CPU only)

| Component | Version |
|---|---|
| OS / arch | macOS 27.0.1, arm64 (no NVIDIA GPU) |
| Python | CPython 3.11.15 (uv-managed) |
| PyTorch | 2.8.0, CPU wheel from download.pytorch.org/whl/cpu |
| Application deps | exactly `requirements-rebuttal.lock` (transformers 4.51.3, peft 0.15.2, datasets 3.6.0, accelerate 1.7.0, huggingface-hub 0.36.2, tokenizers 0.21.4, safetensors 0.8.0, numpy 2.2.6, pytest 9.2.0, …) |

The full test suite and launcher tests passed in this environment. The bootstrap sequence the notebook runs on each worker was also reproduced (fresh `venv --system-site-packages`, `pip install -r requirements-rebuttal.lock`, `pip install --no-deps -e .`, unchanged torch, `contract`). See `TEST_REPORT.md`.

## How the lock was produced

`uv pip compile --python-version 3.10 --python-platform x86_64-unknown-linux-gnu` from the top-level pins listed in the lock header. The following were then removed:

* `torch`, `triton`, all `nvidia-*` and `cuda-*` wheels: the GPU image supplies these;
* the torch-only dependencies `sympy`, `mpmath`, `networkx`, `jinja2`, `markupsafe`, `setuptools`, so the install cannot change the image's torch stack;
* `scipy`: no package code imports it, and the significance tests that used it were removed. Its macOS wheel also failed to load on the development machine.

Pinned versions were chosen by the resolver, not written by hand. Installation was tested only on CPython 3.11 / macOS arm64. Linux x86_64 wheels exist for these pins, but installing them on Linux was not tested here.

## GPU runtime: NOT validated here

No GPU code path was executed by the author of these changes. The rental image must provide:

* Linux x86_64;
* Python ≥ 3.10 with `venv`;
* a CUDA-enabled PyTorch whose build supports the GPU architecture. With `numpy==2.2.6` from the lock this should be torch ≥ 2.3, which is built against the NumPy 2 ABI; tested here only with 2.8.0.

**Blackwell / RTX PRO 6000 (sm_120) compatibility is not claimed.** The remote checks enforce it instead:

1. **bootstrap** records `torch.__version__` before and after installing the lock and fails if it changed (`cmp`).
2. **preflight**:
   * requires exactly one visible CUDA device, `torch.cuda.is_bf16_supported()`, VRAM ≥ `minimum_vram_gib` (70), and a successful bf16 matmul + backward on the device (fails with "no kernel image" if the build lacks the architecture);
   * records `torch_arch_list`, `compute_capability`, CUDA and cuDNN versions and the driver version;
   * runs the full test suite on the GPU (`AUTOMODE_TEST_DEVICE=cuda`), which fails rather than skips without CUDA;
   * verifies every installed package against the lock.
3. **pilot** runs every assigned method on the real Gemma-2-9B for a few steps, plus a 512-example AutoMode/LoRA pilot with export verification.

The notebook then requires all four workers to agree on `torch_version`, `dependency_lock_sha256`, and the model and dataset revisions, and to have distinct GPU UUIDs. Record the combination that passes in your rebuttal notes.
