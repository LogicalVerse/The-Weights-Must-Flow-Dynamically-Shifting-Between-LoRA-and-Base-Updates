"""
Verified full inference export.

PEFT models are saved UNMERGED: the learned base (a normal HF checkpoint in the
export root), every adapter (adapter/) and each layer's merged/unmerged state
(layer_state.json). load_unmerged_export() rebuilds the exact evaluated model.
Merging is not used for the artefact because PEFT keeps adapters in fp32 while a
bf16 merge rounds W + delta to the bf16 grid, so it loses part of the learned
delta. Adapter-only saving against the pretrained base would lose base updates.

Two separate checks:
  - serialization fidelity: reloaded artefact versus the live model (gating);
  - merge audit: formula, untouched merged layers (gating), and merge-induced
    drift (reported in bf16, gating in fp32, where a merge should be faithful).

Models without PEFT wrappers are saved as plain checkpoints.

This MUTATES the model. Call only after all live-model evaluations.
The export is inference-only, not a resumable training checkpoint.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn.functional as F

from automode.core import ADAPTER_NAME


ADAPTER_SUBDIR = "adapter"
LAYER_STATE_FILE = "layer_state.json"
RELOAD_MAX_ABS_LOGIT_DIFF = 1e-4

PROBE_PROMPTS = [
    "Question: Natalia sold clips to 48 of her friends in April, "
    "and then she sold half as many clips in May. "
    "How many clips did Natalia sell altogether in April and May?"
    "\n\nAnswer:",
    "The capital of France is",
    "def fibonacci(n):",
]


def export_tolerances(dtype: str) -> Dict[str, float]:
    """Live-versus-merged tolerance. Gating in fp32; informational in bf16/fp16,
    where the merged model is not what gets saved."""
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

    # Per-position output distributions: a tail-token logit can move far without
    # moving any probability that matters.
    kls, tvs = [], []
    for x, y in zip(a, b):
        p, q = x.double().log_softmax(-1), y.double().log_softmax(-1)
        kls.append((p.exp() * (p - q)).sum(-1).flatten())
        tvs.append(0.5 * (p.exp() - q.exp()).abs().sum(-1).flatten())
    kl, tv = torch.cat(kls), torch.cat(tvs)

    return {
        "max_abs_logit_diff": diff,
        "top1_agreement": float(agree),
        "max_total_variation": float(tv.max()),
        "max_kl": float(kl.max()),
        "mean_kl": float(kl.mean()),
    }


def _lora_modules(model) -> List[tuple]:
    from peft.tuners.lora import LoraLayer
    return [(n, m) for n, m in model.named_modules() if isinstance(m, LoraLayer)]


def _spacing(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Distance between adjacent representable values of dtype at each |x| (float64)."""
    info = torch.finfo(dtype)
    exponent = torch.frexp(x)[1]
    return torch.where(x == 0, torch.full_like(x, info.tiny),
                       torch.ldexp(torch.full_like(x, info.eps), exponent - 1))


@torch.no_grad()
def merge_and_audit(model) -> List[Dict[str, Any]]:
    """
    Merge every unmerged adapter, comparing each result with W + scale*(B@A) in
    float64. Raises on a formula violation or a changed already-merged projection;
    returns per-projection precision-loss measurements.
    """
    modules = _lora_modules(model)
    already = {n: m.get_base_layer().weight.detach().clone()
               for n, m in modules if list(m.merged_adapters) == [ADAPTER_NAME]}
    rows: List[Dict[str, Any]] = []

    for name, module in modules:
        if name in already:
            rows.append({"module": name, "state": "already_merged"})
            continue
        if module.merged_adapters:
            raise RuntimeError(f"Unexpected merged adapters {module.merged_adapters} on {name}")

        weight = module.get_base_layer().weight
        original = weight.detach().clone()
        original_dtype, before = original.dtype, original.double()
        A, B = module.lora_A[ADAPTER_NAME].weight.detach(), module.lora_B[ADAPTER_NAME].weight.detach()
        scale = module.scaling[ADAPTER_NAME]
        delta = (B.double() @ A.double()) * scale
        magnitude = (B.double().abs() @ A.double().abs()) * abs(scale)

        module.merge(safe_merge=True, adapter_names=[ADAPTER_NAME])

        merged = weight.detach()
        if merged.dtype != original_dtype:
            raise RuntimeError(f"Merge changed the dtype of {name}: {original_dtype} -> {merged.dtype}")
        merged_d = merged.double()
        # One representable step of the stored dtype, plus the fp32 arithmetic PEFT uses.
        slack = 4 * torch.finfo(A.dtype).eps * (before.abs() + magnitude)
        error = (merged_d - (before + delta)).abs()
        if bool((error > _spacing(merged_d, original_dtype) + slack).any()):
            raise RuntimeError(f"Merge of {name} does not match W + scale*(B@A): "
                               f"max error {float(error.max())}")

        realized = merged_d - before
        norm = float(delta.norm())
        rows.append({
            "module": name,
            "state": "merged_now",
            "weight_rms": float(before.pow(2).mean().sqrt()),
            "delta_rms": float(delta.pow(2).mean().sqrt()),
            "delta_max_abs": float(delta.abs().max()),
            "unchanged_fraction": float((merged == original).double().mean()),
            "delta_error_ratio": float((realized - delta).norm()) / norm if norm else None,
            "delta_cosine": float((realized * delta).sum()) / (float(realized.norm()) * norm)
            if norm and float(realized.norm()) else None,
            "formula_max_error": float(error.max()),
        })

    for name, snapshot in already.items():
        if not torch.equal(dict(modules)[name].get_base_layer().weight, snapshot):
            raise RuntimeError(f"Already-merged projection {name} changed during export")
    return rows


def _save_unmerged(model, tokenizer, out_dir: Path, cfg) -> Dict[str, Any]:
    from automode.train import DTYPE_MAP, save_json

    inner = model.get_base_model()
    parameters = {n for n, _ in inner.named_parameters(remove_duplicate=False)}
    state = {}
    for key, value in inner.state_dict().items():
        if "lora_" in key:
            continue
        # Cast weights only; internal buffers keep their precision.
        if key in parameters and value.is_floating_point():
            value = value.to(DTYPE_MAP[cfg.dtype])
        state[key.replace(".base_layer.", ".")] = value

    inner.config.use_cache = True
    inner.save_pretrained(out_dir, state_dict=state, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)
    model.save_pretrained(out_dir / ADAPTER_SUBDIR, safe_serialization=True, save_embedding_layers=False)

    modules = _lora_modules(model)
    layer_state = {
        "format_version": 1,
        "adapter": ADAPTER_NAME,
        "dtype": cfg.dtype,
        "attn_implementation": cfg.attn_implementation,
        "merged": [n for n, m in modules if list(m.merged_adapters) == [ADAPTER_NAME]],
        "unmerged": [n for n, m in modules if not m.merged_adapters],
    }
    if len(layer_state["merged"]) + len(layer_state["unmerged"]) != len(modules):
        raise RuntimeError("Unexpected merged-adapter state on a LoRA layer")
    save_json(layer_state, out_dir / LAYER_STATE_FILE)
    return layer_state


def load_unmerged_export(out_dir: Path, device):
    """Rebuild an unmerged export: learned base + adapters + per-layer merged state.
    With base_weight_dtype='float32', run it under autocast_ctx like the live model."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    from automode.train import DTYPE_MAP

    out_dir = Path(out_dir)
    state = json.loads((out_dir / LAYER_STATE_FILE).read_text())
    base = AutoModelForCausalLM.from_pretrained(
        out_dir,
        torch_dtype=DTYPE_MAP[state["dtype"]],
        device_map={"": device},
        attn_implementation=state["attn_implementation"],
    )
    model = PeftModel.from_pretrained(base, out_dir / ADAPTER_SUBDIR, is_trainable=False)

    modules = _lora_modules(model)
    if {n for n, _ in modules} != set(state["merged"]) | set(state["unmerged"]):
        raise RuntimeError("layer_state.json does not match the rebuilt LoRA layers")
    merged = set(state["merged"])
    for name, module in modules:
        # Bookkeeping only: the saved base already contains these deltas.
        module.merged_adapters = [state["adapter"]] if name in merged else []
    model.eval()
    return model


def export_full_inference_model(
    model,
    tokenizer,
    out_dir: Path,
    cfg,
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    from peft import PeftModel
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
    device = next(model.parameters()).device

    probe_ids = [
        tokenizer(prompt, add_special_tokens=True)["input_ids"]
        for prompt in PROBE_PROMPTS
    ]

    live = _probe_logits(model, probe_ids, cfg)
    tol = export_tolerances(cfg.dtype)

    if isinstance(model, PeftModel):
        layer_state = _save_unmerged(model, tokenizer, out_dir, cfg)
        reloaded_model = load_unmerged_export(out_dir, device)
        reloaded = _probe_logits(reloaded_model, probe_ids, cfg)
        del reloaded_model

        # Merged forward is measured only, on the live model after the save.
        audit = merge_and_audit(model)
        merged = _probe_logits(model, probe_ids, cfg)

        checks = {
            "reload_vs_live": _compare(reloaded, live),
            "merge_vs_live": _compare(merged, live),
        }
        drift = checks["merge_vs_live"]
        within_tolerance = (drift["max_abs_logit_diff"] <= tol["max_abs_logit_diff"]
                            and drift["top1_agreement"] >= tol["min_top1_agreement"])
        ok = (
            checks["reload_vs_live"]["max_abs_logit_diff"] <= RELOAD_MAX_ABS_LOGIT_DIFF
            and checks["reload_vs_live"]["top1_agreement"] == 1.0
            and (within_tolerance or cfg.dtype != "float32")
        )
        kind = "full_unmerged_inference_export"
        details = {
            "n_lora_layers_already_merged": len(layer_state["merged"]),
            "n_lora_layers_unmerged_in_export": len(layer_state["unmerged"]),
            "merge_audit": audit,
            "merged_model_within_tolerance": within_tolerance,
        }
    else:
        # Cast weights only. Preserve internal buffer precision.
        # This model will not be trained again after export.
        with torch.no_grad():
            for parameter in model.parameters():
                if parameter.is_floating_point():
                    parameter.data = parameter.data.to(DTYPE_MAP[cfg.dtype])
        model.config.use_cache = True
        cast = _probe_logits(model, probe_ids, cfg)

        model.save_pretrained(out_dir, safe_serialization=True)
        tokenizer.save_pretrained(out_dir)
        reloaded_model = AutoModelForCausalLM.from_pretrained(
            out_dir,
            torch_dtype=DTYPE_MAP[cfg.dtype],
            device_map={"": device},
            attn_implementation=cfg.attn_implementation,
        )
        reloaded = _probe_logits(reloaded_model, probe_ids, cfg)
        del reloaded_model

        checks = {
            "cast_vs_live": _compare(cast, live),
            "reload_vs_cast": _compare(reloaded, cast),
            "reload_vs_live": _compare(reloaded, live),
        }
        ok = (
            checks["reload_vs_cast"]["max_abs_logit_diff"] <= RELOAD_MAX_ABS_LOGIT_DIFF
            and checks["reload_vs_live"]["max_abs_logit_diff"] <= tol["max_abs_logit_diff"]
            and checks["reload_vs_live"]["top1_agreement"] >= tol["min_top1_agreement"]
        )
        kind = "full_inference_export"
        details = {}

    info = {
        # Caller metadata cannot override verification results.
        **metadata,
        "status": "ok" if ok else "failed",
        "kind": kind,
        "resumable_training_checkpoint": False,
        "base_model": cfg.model_name,
        "model_revision": cfg.model_revision,
        "saved_dtype": cfg.dtype,
        **details,
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


# ──────────────────────────────────────────────────────────────────────────────
# GPU diagnostic (`rebuttal export-diagnostic`). Never part of a core job.
# ──────────────────────────────────────────────────────────────────────────────

def _with_replaced_outputs(replacements: Dict[Any, Callable], probe: Callable):
    """Run probe() with each module's output replaced by fn(input). Eval only."""
    handles = [m.register_forward_hook(lambda _m, inp, out, fn=fn: fn(inp[0]).to(out.dtype))
               for m, fn in replacements.items()]
    try:
        return probe()
    finally:
        for h in handles:
            h.remove()


@torch.no_grad()
def _greedy_answers(model, tokenizer, ids_list, cfg, batch_size=8, max_new_tokens=64):
    from automode.eval import batched_generate, extract_gsm8k_answer, normalize_number
    model.eval()
    texts: List[str] = []
    for i in range(0, len(ids_list), batch_size):
        decoded, _ = batched_generate(model, tokenizer, ids_list[i:i + batch_size], cfg,
                                      max_new_tokens=max_new_tokens, do_sample=False)
        texts += [t[0] for t in decoded]
    return texts, [normalize_number(extract_gsm8k_answer(t)[0]) for t in texts]


def base_update_assessment(model, batch: Dict[str, torch.Tensor], probe_log: List[dict], cfg) -> Dict[str, Any]:
    """
    Separate evidence on the unchanged base elements in logs/base_update_probe.json:
    exact-zero gradients on one training batch, and for each logged update the
    fraction expected to change if every element's step were exactly the LR.
    Adam's steps are not exactly the LR, so a match is consistent with rounding,
    not proof of it.
    """
    from automode.train import _base_trainable_params, autocast_ctx
    params = _base_trainable_params(model)
    if not params:
        return {"n_trainable_base_elements": 0}
    device = params[0].device

    model.train()
    with autocast_ctx(cfg, device):
        model(**{k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()})["loss"].backward()
    total = sum(p.numel() for p in params)
    zero_grad = sum(int((p.grad == 0).sum()) if p.grad is not None else p.numel() for p in params)
    model.zero_grad(set_to_none=True)
    model.eval()

    half_spacing = [(_spacing(p.detach().double(), p.dtype) / 2) for p in params]
    per_update = [{
        "step": e["step"], "lr": e["lr"], "observed_changed_fraction": e["changed_fraction"],
        "predicted_if_step_equals_lr": sum(int((h <= e["lr"]).sum()) for h in half_spacing) / total,
    } for e in probe_log]
    return {
        "n_trainable_base_elements": total,
        "weight_dtypes": sorted({str(p.dtype) for p in params}),
        "weight_abs_median": float(torch.cat([p.detach().abs().flatten()[::97].float() for p in params]).median()),
        "zero_gradient_fraction_one_batch": zero_grad / total,
        "per_update": per_update,
    }


def export_diagnostic(
    model,
    tokenizer,
    prompt_ids: List[List[int]],
    out_dir: Path,
    cfg,
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Decide what the merge drift is, on a trained model, without test data.
    Compared with the live model:
      exact_merge:      W + delta evaluated in fp32 (no weight rounding);
      adapter_removed:  unmerged adapters skipped (a merge that kept no delta);
      single merges:    each projection's rounded merge alone;
      merged:           the real bf16 merge, including greedy generations.
    Runs the real export in between, so serialization fidelity is checked too.
    MUTATES the model.
    """
    ids = [tokenizer(p, add_special_tokens=True)["input_ids"] for p in PROBE_PROMPTS] + prompt_ids

    def probe():
        return _probe_logits(model, ids, cfg)

    live = probe()
    live_texts, live_answers = _greedy_answers(model, tokenizer, prompt_ids, cfg)

    unmerged = [(n, m) for n, m in _lora_modules(model) if not m.merged_adapters]
    with torch.no_grad():
        exact = {
            m: (lambda x, w=m.get_base_layer().weight.float() + m.get_delta_weight(ADAPTER_NAME).float():
                F.linear(x.float(), w))
            for _, m in unmerged
        }
        rounded = {
            n: (m, lambda x, w=(m.get_base_layer().weight.double()
                                + m.get_delta_weight(ADAPTER_NAME).double()).to(m.get_base_layer().weight.dtype):
                F.linear(x.to(w.dtype), w))
            for n, m in unmerged
        }
        report: Dict[str, Any] = {
            **metadata,
            "n_unmerged_projections": len(unmerged),
            "n_diagnostic_prompts": len(prompt_ids),
            "exact_merge_vs_live": _compare(_with_replaced_outputs(exact, probe), live),
        }
        for _, m in unmerged:
            m._disable_adapters = True
        removed = probe()
        for _, m in unmerged:
            m._disable_adapters = False
        report["adapter_removed_vs_live"] = _compare(removed, live)
        report["single_projection_merge_vs_live"] = {
            n: _compare(_with_replaced_outputs({m: fn}, probe), live) for n, (m, fn) in rounded.items()
        }

    try:
        report["export"] = export_full_inference_model(model, tokenizer, out_dir, cfg, metadata)
    except RuntimeError as e:
        # A failed export leaves the model state unknown; report what was measured.
        report["export_error"] = str(e)
        return report

    merged = probe()
    merged_texts, merged_answers = _greedy_answers(model, tokenizer, prompt_ids, cfg)
    report["merged_vs_live"] = _compare(merged, live)
    report["merged_vs_adapter_removed"] = _compare(merged, removed)
    n = max(len(prompt_ids), 1)
    report["greedy_generation"] = {
        "max_new_tokens": 64,
        "identical_text_fraction": sum(a == b for a, b in zip(live_texts, merged_texts)) / n,
        "same_extracted_answer_fraction": sum(a == b for a, b in zip(live_answers, merged_answers)) / n,
    }
    return report
