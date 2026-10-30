"""
Verified full inference export with staged diagnostics.

Preserves learned base weights, merges only unmerged adapters, removes
PEFT wrappers, saves the full model, and verifies reload behavior.

Diagnostic comparisons isolate:
  - adapter merging,
  - wrapper removal,
  - parameter casting,
  - saving and reloading.

This MUTATES the model. Call only after all live-model evaluations.
The export is inference-only, not a resumable training checkpoint.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List

import torch


PROBE_PROMPTS = [
    "Question: Natalia sold clips to 48 of her friends in April, "
    "and then she sold half as many clips in May. "
    "How many clips did Natalia sell altogether in April and May?"
    "\n\nAnswer:",
    "The capital of France is",
    "def fibonacci(n):",
]


def export_tolerances(dtype: str) -> Dict[str, float]:
    """Existing live-versus-reloaded acceptance thresholds."""
    if dtype == "float32":
        return {
            "max_abs_logit_diff": 1e-3,
            "min_top1_agreement": 1.0,
        }
    return {
        "max_abs_logit_diff": 1.0,
        "min_top1_agreement": 0.98,
    }


@torch.no_grad()
def _probe_logits(model, ids_list, cfg) -> List[torch.Tensor]:
    from automode.train import autocast_ctx

    model.eval()
    device = next(model.parameters()).device
    out = []

    with autocast_ctx(cfg, device):
        for ids in ids_list:
            logits = model(
                input_ids=torch.tensor(
                    [ids], dtype=torch.long, device=device
                ),
                use_cache=False,
            )["logits"]
            out.append(logits.float().cpu())

    return out


def _compare(
    a: List[torch.Tensor],
    b: List[torch.Tensor],
) -> Dict[str, float]:
    diff = max(
        float((x - y).abs().max())
        for x, y in zip(a, b)
    )
    agree = torch.cat([
        (x.argmax(-1) == y.argmax(-1)).flatten()
        for x, y in zip(a, b)
    ]).float().mean()

    return {
        "max_abs_logit_diff": diff,
        "top1_agreement": float(agree),
    }


def export_full_inference_model(
    model,
    tokenizer,
    out_dir: Path,
    cfg,
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    from peft import PeftModel
    from peft.tuners.lora import LoraLayer
    from transformers import AutoModelForCausalLM
    from automode.train import DTYPE_MAP, save_json

    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(
            f"Export directory {out_dir} is not empty"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    model.eval()

    probe_ids = [
        tokenizer(prompt, add_special_tokens=True)["input_ids"]
        for prompt in PROBE_PROMPTS
    ]

    # Stage 1: original live model.
    live = _probe_logits(model, probe_ids, cfg)

    n_already_merged = 0
    n_merged_now = 0

    if isinstance(model, PeftModel):
        for module in model.modules():
            if not isinstance(module, LoraLayer):
                continue

            if list(module.merged_adapters) == ["default"]:
                # Already contains its merged delta and direct base updates.
                n_already_merged += 1
            elif not module.merged_adapters:
                module.merge(
                    safe_merge=True,
                    adapter_names=["default"],
                )
                n_merged_now += 1
            else:
                raise RuntimeError(
                    "Unexpected merged adapters "
                    f"{module.merged_adapters}"
                )

        # Stage 2: merged, with PEFT wrappers still present.
        after_merge = _probe_logits(model, probe_ids, cfg)

        # Remove wrappers without merging again.
        base = model.unload()
    else:
        base = model
        after_merge = live

    # Stage 3: wrappers removed, before export parameter casting.
    before_cast = _probe_logits(base, probe_ids, cfg)

    # Cast weights only. Preserve internal buffer precision.
    # This model will not be trained again after export.
    with torch.no_grad():
        for parameter in base.parameters():
            if parameter.is_floating_point():
                parameter.data = parameter.data.to(
                    DTYPE_MAP[cfg.dtype]
                )

    base.eval()
    base.config.use_cache = True

    # Stage 4: final in-memory model that will be saved.
    # Probes explicitly disable caching for every stage.
    merged = _probe_logits(base, probe_ids, cfg)

    base.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)

    device = next(base.parameters()).device
    reloaded_model = AutoModelForCausalLM.from_pretrained(
        out_dir,
        torch_dtype=DTYPE_MAP[cfg.dtype],
        device_map={"": device},
        attn_implementation=cfg.attn_implementation,
    )

    # Stage 5: freshly reloaded model.
    reloaded = _probe_logits(reloaded_model, probe_ids, cfg)
    del reloaded_model

    tol = export_tolerances(cfg.dtype)
    checks = {
        "merge_vs_live": _compare(after_merge, live),
        "unload_vs_merge": _compare(before_cast, after_merge),
        "cast_vs_unloaded": _compare(merged, before_cast),
        "reload_vs_merged": _compare(reloaded, merged),
        "reload_vs_live": _compare(reloaded, live),
    }

    # Preserve the original acceptance criteria.
    ok = (
        checks["reload_vs_merged"]["max_abs_logit_diff"] <= 1e-4
        and checks["reload_vs_live"]["max_abs_logit_diff"]
        <= tol["max_abs_logit_diff"]
        and checks["reload_vs_live"]["top1_agreement"]
        >= tol["min_top1_agreement"]
    )

    info = {
        # Caller metadata cannot override verification results.
        **metadata,
        "status": "ok" if ok else "failed",
        "kind": "full_inference_export",
        "resumable_training_checkpoint": False,
        "base_model": cfg.model_name,
        "model_revision": cfg.model_revision,
        "saved_dtype": cfg.dtype,
        "n_lora_layers_already_merged": n_already_merged,
        "n_lora_layers_merged_at_export": n_merged_now,
        "probe_prompts": PROBE_PROMPTS,
        "checks": checks,
        "tolerances": tol,
        "export_time_sec": time.perf_counter() - t0,
    }

    # Save diagnostics even when verification fails.
    save_json(info, out_dir / "export_metadata.json")

    if not ok:
        raise RuntimeError(
            f"Export verification failed: {checks}"
        )

    return info