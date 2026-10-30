"""
Verified FULL inference export.

PEFT's adapter-only save_pretrained() drops everything AutoMode learned in base
mode (merged deltas plus direct base updates). This export instead:

  1. records eval-mode logits of the live model on fixed probe prompts,
  2. merges ONLY adapters that are not merged yet, exactly once (layers that
     were promoted already contain their merged delta and base updates),
  3. strips the LoRA wrappers without further merging (PeftModel.unload),
  4. casts to cfg.dtype and saves the full model + tokenizer + config,
  5. reloads from disk and compares logits: reload vs in-memory merged model
     must match (<= 1e-4, i.e. same weights); reload vs the live model within
     export_tolerances().

It MUTATES the model (merge + unload), so it must run after every live-model
evaluation, and the model must not be trained or evaluated afterwards.
The result is an inference-only artefact, NOT a resumable training checkpoint
(no optimizer, scheduler, RNG, controller or data-order state). Interrupted
runs restart from scratch.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List

import torch

PROBE_PROMPTS = [
    "Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. "
    "How many clips did Natalia sell altogether in April and May?\n\nAnswer:",
    "The capital of France is",
    "def fibonacci(n):",
]


def export_tolerances(dtype: str) -> Dict[str, float]:
    """live-vs-reloaded logit tolerance. Merging changes rounding in bf16/fp16,
    so low precision gets a loose absolute bound plus a top-1 agreement floor."""
    if dtype == "float32":
        return {"max_abs_logit_diff": 1e-3, "min_top1_agreement": 1.0}
    return {"max_abs_logit_diff": 1.0, "min_top1_agreement": 0.98}


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


def _compare(a: List[torch.Tensor], b: List[torch.Tensor]) -> Dict[str, float]:
    diff = max(float((x - y).abs().max()) for x, y in zip(a, b))
    agree = torch.cat([(x.argmax(-1) == y.argmax(-1)).flatten() for x, y in zip(a, b)]).float().mean()
    return {"max_abs_logit_diff": diff, "top1_agreement": float(agree)}


def export_full_inference_model(model, tokenizer, out_dir: Path, cfg, metadata: Dict[str, Any]) -> Dict[str, Any]:
    from peft import PeftModel
    from peft.tuners.lora import LoraLayer
    from transformers import AutoModelForCausalLM
    from automode.train import DTYPE_MAP, save_json

    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Export directory {out_dir} is not empty")
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    model.eval()
    probe_ids = [tokenizer(p, add_special_tokens=True)["input_ids"] for p in PROBE_PROMPTS]
    live = _probe_logits(model, probe_ids, cfg)

    n_already_merged = n_merged_now = 0
    if isinstance(model, PeftModel):
        for module in model.modules():
            if not isinstance(module, LoraLayer):
                continue
            if list(module.merged_adapters) == ["default"]:
                n_already_merged += 1
            elif not module.merged_adapters:
                module.merge(safe_merge=True, adapter_names=["default"])
                n_merged_now += 1
            else:
                raise RuntimeError(f"Unexpected merged adapters {module.merged_adapters}")
        base = model.unload()
    else:
        base = model
    # Cast exported weights without rounding internal buffers.
    with torch.no_grad():
        for parameter in base.parameters():
            if parameter.is_floating_point():
                parameter.data = parameter.data.to(DTYPE_MAP[cfg.dtype])
    base.eval()
    base.config.use_cache = True
    merged = _probe_logits(base, probe_ids, cfg)

    base.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)
    device = next(base.parameters()).device
    reloaded_model = AutoModelForCausalLM.from_pretrained(
        out_dir, torch_dtype=DTYPE_MAP[cfg.dtype], device_map={"": device},
        attn_implementation=cfg.attn_implementation)
    reloaded = _probe_logits(reloaded_model, probe_ids, cfg)
    del reloaded_model

    tol = export_tolerances(cfg.dtype)
    checks = {
        "reload_vs_merged": _compare(reloaded, merged),
        "reload_vs_live": _compare(reloaded, live),
    }
    ok = (checks["reload_vs_merged"]["max_abs_logit_diff"] <= 1e-4
          and checks["reload_vs_live"]["max_abs_logit_diff"] <= tol["max_abs_logit_diff"]
          and checks["reload_vs_live"]["top1_agreement"] >= tol["min_top1_agreement"])
    info = {
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
        **metadata,
    }
    save_json(info, out_dir / "export_metadata.json")
    if not ok:
        raise RuntimeError(f"Export verification failed: {checks}")
    return info
