"""
Model loading and per-method builders.

Every fine-tuning method is a function that takes a base model and returns a
modified model ready for training. The main entry point is build_model_for_method().

Method builders in this file:
  - apply_full_ft           (all params trainable)
  - apply_lora              (standard LoRA)
  - apply_dora              (DoRA = magnitude + direction LoRA)
  - apply_adalora           (adaptive rank LoRA)
  - apply_bitfit            (bias-only updates)
  - apply_topk_static       (last-k layers fully trainable)
  - apply_topk_deep_block   (fixed deep block trainable)
  - apply_automode          (injects LoRA, returns model — controller handles rest)
  - apply_dyn_full_only     (same as automode — method string switches behavior in controller)
  - apply_lisa              (injects nothing, full-param — controller samples layers)
  - apply_adagradselect     (same as lisa)
  - apply_loraga            (gradient-aligned LoRA init; UNAVAILABLE, see config)
  - apply_fixed_mixed       (rebuttal control: fixed random q/v base layers + LoRA rest)
  - apply_static_qv         (rebuttal control: q/v base weights of every layer)
  select_once uses apply_automode; the controller freezes the allocation after
  its first scheduled selection.
"""

from __future__ import annotations
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import (
    LoraConfig,
    AdaLoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)

import torch.nn as nn

from automode.config import RunConfig, UNAVAILABLE_METHODS
from automode.core import (
    get_transformer_layers,
    freeze_non_layer_params,
    count_trainable_params,
    identify_layer_for_param,
    identify_layer_for_module,
    group_lora_modules_by_layer,
    fixed_layer_allocation,
    promote_to_fft,
    current_layer_mode,
)
from automode.train import DTYPE_MAP


# ──────────────────────────────────────────────────────────────────────────────
# Tokenizer
# ──────────────────────────────────────────────────────────────────────────────

def get_tokenizer(model_name: str, revision: Optional[str] = None):
    tok = AutoTokenizer.from_pretrained(model_name, revision=revision, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if tok.eos_token_id is None or tok.pad_token_id is None:
        raise RuntimeError("Tokenizer needs eos and pad tokens")
    tok.padding_side = "right"  # collators pad explicitly; this is only a default
    return tok


# Methods whose q/v BASE weights may receive optimizer updates.
BASE_TRAINING_METHODS = ("automode", "select_once", "fixed_mixed", "static_qv")


def upcast_target_base_weights(model, target_modules) -> int:
    """Keep q/v base weights in fp32 (cfg.base_weight_dtype='float32')."""
    n = 0
    for name, module in model.named_modules():
        if (isinstance(module, nn.Linear) and name.split(".")[-1] in target_modules
                and identify_layer_for_module(name) is not None):
            module.weight.data = module.weight.data.float()
            n += 1
    if n == 0:
        raise RuntimeError(f"No target modules {target_modules} found to upcast")
    return n


# ──────────────────────────────────────────────────────────────────────────────
# Base model loading
# ──────────────────────────────────────────────────────────────────────────────

def load_base_model(cfg: RunConfig):
    """
    Load the raw pretrained model.

    Key choices:
      - device_map={"": device} pins everything to ONE GPU. Never "auto" —
        that would shard across GPUs and break our gradient-surgery logic.
      - use_cache=False for gradient checkpointing compatibility.
      - enable_input_require_grads for gradient flow through frozen layers.
    """
    dtype = DTYPE_MAP[cfg.dtype]
    device = cfg.device if torch.cuda.is_available() else "cpu"

    kwargs = dict(
        torch_dtype=dtype,
        device_map={"": device},
        low_cpu_mem_usage=cfg.low_cpu_mem_usage,
        revision=cfg.model_revision,
        attn_implementation=cfg.attn_implementation,
    )
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **kwargs)

    # KV-cache off during training (incompatible with gradient checkpointing)
    model.config.use_cache = False

    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        # Required so gradients flow through frozen leading layers
        model.enable_input_require_grads()

    return model


# ──────────────────────────────────────────────────────────────────────────────
# Method-specific builders
# ──────────────────────────────────────────────────────────────────────────────

def apply_full_ft(model):
    for p in model.parameters():
        p.requires_grad = True
    return model


def apply_lora(model, cfg: RunConfig):
    peft_cfg = LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.lora_target_modules),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        init_lora_weights=True,  # A ~ U(-1/sqrt(d_in), 1/sqrt(d_in)), B = 0
    )
    model = get_peft_model(model, peft_cfg)
    freeze_non_layer_params(model)
    return model


def apply_dora(model, cfg: RunConfig):
    peft_cfg = LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.lora_target_modules),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        use_dora=True,
    )
    return get_peft_model(model, peft_cfg)


def apply_adalora(model, cfg: RunConfig, total_steps: int):
    # Clamp AdaLoRA schedule to total_steps
    tinit = min(cfg.adalora_tinit, max(1, total_steps // 4))
    tfinal = min(cfg.adalora_tfinal, max(1, total_steps - tinit - 1))
    deltaT = min(cfg.adalora_deltaT, max(1, total_steps - tinit - tfinal))
    peft_cfg = AdaLoraConfig(
        init_r=cfg.adalora_init_r, target_r=cfg.adalora_target_r,
        lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.lora_target_modules),
        task_type=TaskType.CAUSAL_LM, bias="none",
        total_step=total_steps, tinit=tinit, tfinal=tfinal, deltaT=deltaT,
    )
    return get_peft_model(model, peft_cfg)


def apply_bitfit(model):
    """
    BitFit baseline (Ben Zaken et al. 2021): update ONLY bias vectors.

    Strict implementation — no 1D-parameter fallback. If the target model has
    no bias parameters (e.g., Gemma/LLaMA use bias=False in many linears), we
    raise an explicit error rather than silently including LayerNorm weights
    or other 1D params, because that would inflate trainable count in a way
    that's hard to explain to reviewers.

    The lm_head bias is NOT automatically unfrozen here — we only touch named
    bias parameters, period. If the paper variant you want includes lm_head,
    use a separate method.
    """
    for p in model.parameters():
        p.requires_grad = False
    enabled_any = False
    n_biases = 0
    for name, p in model.named_parameters():
        if name.lower().endswith("bias") or ".bias" in name:
            p.requires_grad = True
            enabled_any = True
            n_biases += p.numel()
    if not enabled_any:
        raise RuntimeError(
            f"BitFit found zero bias parameters in {type(model).__name__}. "
            f"Modern LLaMA-family models often use bias=False in Linear layers, "
            f"making BitFit inapplicable. Use `bitfit` only on models with biases, "
            f"or drop it from your baseline set for this model."
        )
    return model


def apply_topk_static(model, cfg: RunConfig):
    """
    Last-k transformer layers fully trainable, everything else frozen.

    Note on lm_head: following standard practice in the selective-update
    literature (LISA, AdaGradSelect), we keep lm_head trainable because task
    adaptation requires the output head to learn new distributions. If you want
    a stricter "layers-only" baseline, set cfg.keep_lm_head_trainable=False.
    The default is True.
    """
    for p in model.parameters():
        p.requires_grad = False
    layers = get_transformer_layers(model)
    n = len(layers)
    k = min(cfg.topk_k, n)
    selected = set(range(n - k, n))
    for i, layer in enumerate(layers):
        if i in selected:
            for p in layer.parameters():
                p.requires_grad = True
    if getattr(cfg, "keep_lm_head_trainable", True) and hasattr(model, "lm_head"):
        for p in model.lm_head.parameters():
            p.requires_grad = True
    return model


def apply_topk_deep_block(model, cfg: RunConfig):
    """
    Contiguous deep block [start, end] (inclusive) fully trainable.

    lm_head behavior: same as apply_topk_static — trainable by default,
    controllable via cfg.keep_lm_head_trainable.
    """
    for p in model.parameters():
        p.requires_grad = False
    layers = get_transformer_layers(model)
    lo, hi = cfg.deep_block_start, cfg.deep_block_end
    selected = set(range(lo, hi + 1))
    for i, layer in enumerate(layers):
        if i in selected:
            for p in layer.parameters():
                p.requires_grad = True
    if getattr(cfg, "keep_lm_head_trainable", True) and hasattr(model, "lm_head"):
        for p in model.lm_head.parameters():
            p.requires_grad = True
    return model


def apply_automode(model, cfg: RunConfig):
    """
    Initialize the model for AutoMode:
    1. Inject LoRA adapters at cfg.lora_target_modules.
    2. Explicitly freeze lm_head and embeddings (belt-and-suspenders).
    3. All layers start in LoRA mode; the controller handles switches.
    """
    peft_cfg = LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.lora_target_modules),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, peft_cfg)
    freeze_non_layer_params(model, verbose=True)
    return model


def apply_fixed_mixed(model, cfg: RunConfig):
    """
    Rebuttal control: from step 0, a fixed set of cfg.fixed_base_layers layers
    (chosen by stdlib random with cfg.allocation_seed, never by loss or test
    data) trains its q/v BASE weights; every other layer trains q/v LoRA with
    the AutoMode adapter config. The zero-delta adapters of the base layers are
    merged once (adding exactly zero) so promote/export bookkeeping is uniform.
    The chosen layer IDs are stored on model.automode_allocation.
    """
    model = apply_automode(model, cfg)
    groups = group_lora_modules_by_layer(model)
    order = sorted(groups, key=lambda s: int(s.split("_")[1]))
    chosen = fixed_layer_allocation(len(order), cfg.fixed_base_layers, cfg.allocation_seed)
    for i in chosen:
        promote_to_fft(groups[order[i]])
    model.automode_allocation = {
        "base_layer_indices": chosen, "allocation_seed": cfg.allocation_seed,
        "n_layers": len(order), "n_base": len(chosen),
    }
    model.automode_layer_modes = {lid: current_layer_mode(groups[lid]) for lid in order}
    return model


def apply_static_qv(model, cfg: RunConfig):
    """Rebuttal control: q/v base weights of every layer train; nothing else."""
    for p in model.parameters():
        p.requires_grad = False
    n_layers = len(get_transformer_layers(model))
    found = set()
    for name, module in model.named_modules():
        lid = identify_layer_for_module(name)
        if lid is not None and name.split(".")[-1] in cfg.lora_target_modules:
            for p in module.parameters():
                p.requires_grad = True
            found.add((lid, name.split(".")[-1]))
    if len(found) != n_layers * len(cfg.lora_target_modules):
        raise RuntimeError(f"static_qv found {len(found)} target modules, expected "
                           f"{n_layers * len(cfg.lora_target_modules)}")
    model.automode_layer_modes = {f"layer_{i}": "full_ft" for i in range(n_layers)}
    return model


def apply_dyn_full_only(model, cfg: RunConfig):
    """
    Ablation: same as AutoMode, but non-selected layers are FROZEN (not LoRA).
    Implementation-wise identical setup; controller enforces the freeze instead
    of demote_to_lora.
    """
    return apply_automode(model, cfg)


def apply_lisa(model, cfg: RunConfig):
    """
    LISA: no PEFT wrapping. All base params trainable, controller samples
    which ones to actually update per step.
    """
    return apply_full_ft(model)


def apply_adagradselect(model, cfg: RunConfig):
    """AdaGradSelect: same as LISA — full-param, controller selects blocks."""
    return apply_full_ft(model)


def apply_loraga(model, cfg: RunConfig, train_loader):
    """
    LoRA-GA: gradient-aligned LoRA init. Requires a few-step gradient estimation
    pass before get_peft_model(). Uses the custom LoRA-GA PEFT fork.

    UNAVAILABLE: earlier versions silently fell back to plain LoRA when the fork
    was missing. It now raises ImportError instead of substituting a method.
    """
    from peft import LoraGAConfig, get_peft_model as _gpm
    from peft.utils.lora_ga_utils import (
        estimate_gradient, LoraGAContext,
    )

    from itertools import islice
    device = cfg.device
    model = model.to(device)
    model.train()

    # Build a small fixed-size grad-estimation loader
    from torch.utils.data import DataLoader
    max_batches = min(cfg.loraga_steps_for_grad_est, len(train_loader))
    subloader = list(islice(train_loader, max_batches))

    # Move batches to device on the fly
    def _to_device(batch):
        return {k: (v.to(device) if torch.is_tensor(v) else v)
                for k, v in batch.items()}

    device_batches = [_to_device(b) for b in subloader]

    named_grad = estimate_gradient(
        model=model, dataloader=device_batches,
        accelerator=None, quant_flag=False,
    )
    peft_cfg = LoraGAConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.lora_target_modules),
        bias="none",
        bsz=cfg.train_batch_size, iters=max_batches,
        max_length=cfg.max_source_len + cfg.max_target_len,
        dtype="bf16" if cfg.dtype == "bfloat16" else (
            "fp16" if cfg.dtype == "float16" else "fp32"
        ),
        direction="ArB2r", scale="stable", stable_gamma=16,
    )
    with LoraGAContext(model=model, named_grad=named_grad):
        model = _gpm(model=model, peft_config=peft_cfg)
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Dispatcher
# ──────────────────────────────────────────────────────────────────────────────

def build_model_for_method(cfg: RunConfig, tokenizer=None, train_loader=None):
    """
    Load base + apply the method. This is the single entry point train.py calls.
    """
    if cfg.method in UNAVAILABLE_METHODS:
        raise NotImplementedError(
            f"Method {cfg.method!r} is unavailable until revalidated: {UNAVAILABLE_METHODS[cfg.method]}")
    return apply_method(load_base_model(cfg), cfg, train_loader)


def apply_method(model, cfg: RunConfig, train_loader=None):
    """Apply cfg.method to an already-loaded causal LM (also used by CPU tests)."""
    if cfg.method in UNAVAILABLE_METHODS:
        raise NotImplementedError(
            f"Method {cfg.method!r} is unavailable until revalidated: {UNAVAILABLE_METHODS[cfg.method]}")
    if cfg.base_weight_dtype == "float32" and cfg.method in BASE_TRAINING_METHODS:
        upcast_target_base_weights(model, cfg.lora_target_modules)

    # Most methods just apply and return
    dispatch_simple = {
        "full_ft":          apply_full_ft,
        "lora":             lambda m: apply_lora(m, cfg),
        "dora":             lambda m: apply_dora(m, cfg),
        "bitfit":           apply_bitfit,
        "topk_static":      lambda m: apply_topk_static(m, cfg),
        "topk_deep_block":  lambda m: apply_topk_deep_block(m, cfg),
        "automode":         lambda m: apply_automode(m, cfg),
        "select_once":      lambda m: apply_automode(m, cfg),
        "fixed_mixed":      lambda m: apply_fixed_mixed(m, cfg),
        "static_qv":        lambda m: apply_static_qv(m, cfg),
        "dyn_full_only":    lambda m: apply_dyn_full_only(m, cfg),
        "lisa":             lambda m: apply_lisa(m, cfg),
        "adagradselect":    lambda m: apply_adagradselect(m, cfg),
    }

    if cfg.method in dispatch_simple:
        model = dispatch_simple[cfg.method](model)
    elif cfg.method == "adalora":
        # Needs total_steps, passed via train.py awareness — we compute it here
        # from train_loader length and cfg.epochs.
        if train_loader is None:
            raise ValueError("AdaLoRA requires train_loader to compute schedule.")
        import math
        total_steps = (
            math.ceil(len(train_loader) / cfg.grad_accum_steps) * cfg.epochs
        )
        model = apply_adalora(model, cfg, total_steps)
    elif cfg.method == "loraga":
        if train_loader is None:
            raise ValueError("LoRA-GA requires train_loader for grad estimation.")
        model = apply_loraga(model, cfg, train_loader)
    else:
        raise ValueError(f"No builder registered for method: {cfg.method}")

    return model
