"""
AutoMode: Dynamic per-layer switching between LoRA and Full Fine-Tuning.

Entry points
------------
RunConfig          - all hyperparameters for one run
run_experiment     - end-to-end single experiment (legacy grid)
run_grid           - resume-safe legacy grid runner
AutoModeController - the switching logic itself (can be reused standalone)
automode.rebuttal  - rebuttal campaign CLI (python -m automode.rebuttal ...)

Attributes are imported lazily so that `python -m automode.rebuttal contract`
and automode.config stay free of torch/transformers imports.
"""

from importlib import import_module

_EXPORTS = {
    "RunConfig": "automode.config",
    "ImportanceSignal": "automode.config",
    "SwitchingMode": "automode.config",
    "METHOD_FIDELITY": "automode.config",
    "AutoModeController": "automode.core",
    "identify_layer_for_param": "automode.core",
    "run_experiment": "automode.train",
    "train_one_run": "automode.train",
    "run_grid": "automode.grid",
    "build_tier_grid": "automode.grid",
    "evaluate_gsm8k": "automode.eval",
    "evaluate_math": "automode.eval",
    "evaluate_mmlu": "automode.eval",
    "evaluate_arc": "automode.eval",
}

__all__ = list(_EXPORTS)
__version__ = "0.3.0"


def __getattr__(name):
    if name in _EXPORTS:
        return getattr(import_module(_EXPORTS[name]), name)
    raise AttributeError(f"module 'automode' has no attribute {name!r}")
