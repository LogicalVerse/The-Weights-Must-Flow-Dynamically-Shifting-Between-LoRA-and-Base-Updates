"""
AutoMode: Dynamic per-layer switching between LoRA and Full Fine-Tuning.

Entry points
------------
RunConfig          - all hyperparameters for one run
run_experiment     - end-to-end single experiment
run_grid           - resume-safe grid runner
AutoModeController - the switching logic itself (can be reused standalone)
"""

from automode.config import RunConfig, ImportanceSignal, SwitchingMode, METHOD_FIDELITY
from automode.core import AutoModeController, identify_layer_for_param
from automode.train import run_experiment, train_one_run
from automode.grid import run_grid, build_tier_grid
from automode.eval import evaluate_gsm8k, evaluate_math, evaluate_mmlu, evaluate_arc

__all__ = [
    "RunConfig",
    "ImportanceSignal",
    "SwitchingMode",
    "METHOD_FIDELITY",
    "AutoModeController",
    "identify_layer_for_param",
    "run_experiment",
    "train_one_run",
    "run_grid",
    "build_tier_grid",
    "evaluate_gsm8k",
    "evaluate_math",
    "evaluate_mmlu",
    "evaluate_arc",
]

__version__ = "0.1.0"
