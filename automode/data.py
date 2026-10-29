"""
Dataset loading and tokenization.

Supports training corpora:
  - gsm8k        (7.5K examples, #### format)
  - math         (7.5K, \boxed{} format)
  - metamath     (40K, mixed GSM8K + MATH style — the AdaGradSelect corpus)
  - alpaca_gpt4  (52K, instruction-following)

All training pipelines produce a single tokenized DataLoader with input_ids,
attention_mask, and labels (prompt tokens masked to -100).

Sequence construction (TARGET_FORMAT_VERSION):
  prompt_ids = tokenizer(prompt) with special tokens (BOS); if longer than
               max_source_len, keep BOS and the LAST max_source_len-1 tokens,
               so the "Answer:" boundary survives (left truncation).
  target_ids = tokenizer(" " + target) without special tokens, then EOS,
               then cut to max_target_len (a cut target loses its EOS).
  input_ids  = prompt_ids + target_ids;  labels = [-100]*len(prompt) + target_ids.
The prompt template text is the submitted GSM8K template, unchanged. Appending
EOS and tokenizing prompt and target separately are corrected-protocol choices
(see CORRECTIONS.md). Evaluation encodes prompts with the same encode_prompt().
Nothing is dropped: truncation and empty-target counts are logged.

Evaluation datasets (GSM8K-test, MATH-test, MMLU, ARC-C) are handled in eval.py.
"""

from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from datasets import load_dataset, Dataset, DatasetDict
from datasets.utils.logging import disable_progress_bar
import hashlib
import json

from torch.utils.data import DataLoader, Sampler

from automode.config import RunConfig

disable_progress_bar()


# ──────────────────────────────────────────────────────────────────────────────
# Prompt builders for each training track
# ──────────────────────────────────────────────────────────────────────────────

TARGET_FORMAT_VERSION = "separate_tokenization_space_target_eos_v2"
# Fixed seed for choosing a max_train_samples subset: the subset must not depend
# on the training seed or method.
SUBSET_SEED = 42

GSM8K_PROMPT_TEMPLATE = (
    "You are an expert grade-school math tutor. "
    "Solve the problem step by step, then give the final numeric answer on a "
    "separate line as:\n#### <NUMERIC_ANSWER>\n\n"
    "Question: {q}\n\nAnswer:"
)

MATH_PROMPT_TEMPLATE = (
    "Solve the following math problem step by step. "
    "Enclose the final answer in \\boxed{{}}.\n\n"
    "Problem: {q}\n\nSolution:"
)

METAMATH_PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task. Write a response that "
    "appropriately completes the request.\n\n"
    "### Instruction:\n{q}\n\n### Response:"
)


def build_gsm8k_example(ex: dict) -> dict:
    q = ex["question"].strip()
    a = ex["answer"].strip()
    return {
        "prompt": GSM8K_PROMPT_TEMPLATE.format(q=q),
        "target": a,
        "raw_question": q,
        "raw_answer": a,
    }


def build_math_example(ex: dict) -> dict:
    # MATH dataset has fields: problem, solution, answer, level, type
    q = ex.get("problem", ex.get("question", "")).strip()
    # Full solution including \boxed{}
    a = ex.get("solution", ex.get("answer", "")).strip()
    return {
        "prompt": MATH_PROMPT_TEMPLATE.format(q=q),
        "target": a,
        "raw_question": q,
        "raw_answer": a,
    }


def build_metamath_example(ex: dict) -> dict:
    # MetaMathQA has: query, response, type, original_question, original_answer
    q = ex.get("query", ex.get("question", "")).strip()
    a = ex.get("response", ex.get("answer", "")).strip()
    return {
        "prompt": METAMATH_PROMPT_TEMPLATE.format(q=q),
        "target": a,
        "raw_question": q,
        "raw_answer": a,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Raw dataset loaders (with fallback names)
# ──────────────────────────────────────────────────────────────────────────────

def load_gsm8k_raw(revision: Optional[str] = None) -> DatasetDict:
    return load_dataset("openai/gsm8k", "main", revision=revision)


def load_math_raw() -> DatasetDict:
    for name in [
        "DigitalLearningGmbH/MATH-lighteval",
        "lighteval/MATH",
        "hendrycks/competition_math",
    ]:
        try:
            return load_dataset(name)
        except Exception:
            continue
    raise RuntimeError("Could not load any MATH dataset candidate.")


def load_metamath_raw() -> DatasetDict:
    for name in [
        "meta-math/MetaMathQA",
        "meta-math/MetaMathQA-40K",
    ]:
        try:
            return load_dataset(name)
        except Exception:
            continue
    raise RuntimeError("Could not load MetaMathQA.")


def load_alpaca_raw() -> DatasetDict:
    return load_dataset("vicgalle/alpaca-gpt4")


# ──────────────────────────────────────────────────────────────────────────────
# Unified training dataset normalisation
# ──────────────────────────────────────────────────────────────────────────────

def load_and_normalize_training_data(cfg: RunConfig) -> Dataset:
    """
    Return a single flat Dataset with columns: prompt, target, raw_question, raw_answer.

    For multi-split datasets (gsm8k: train+test, math: train+test), we take
    only the TRAIN split — eval is handled separately in eval.py on the test
    split of each benchmark in cfg.eval_benchmarks.
    """
    t = cfg.train_track
    if t == "gsm8k":
        raw = load_gsm8k_raw(cfg.dataset_revision)["train"]
        ds = raw.map(build_gsm8k_example)
    elif t == "math":
        raw = load_math_raw()["train"]
        ds = raw.map(build_math_example)
    elif t == "metamath":
        raw = load_metamath_raw()["train"]
        ds = raw.map(build_metamath_example)
    elif t == "alpaca":
        # Alpaca: treat instruction + input -> output
        raw = load_alpaca_raw()["train"]
        def _mk(ex):
            instr = ex.get("instruction", "").strip()
            inp = ex.get("input", "").strip()
            q = instr if not inp else f"{instr}\n\n{inp}"
            a = ex.get("output", "").strip()
            return {
                "prompt": METAMATH_PROMPT_TEMPLATE.format(q=q),
                "target": a, "raw_question": q, "raw_answer": a,
            }
        ds = raw.map(_mk)
    else:
        raise ValueError(f"Unknown train_track: {t}")

    if cfg.max_train_samples is not None and cfg.max_train_samples < len(ds):
        # Deterministic shuffle-then-select so the 40K subset is reproducible
        # across seeds. We use seed=42 as a fixed "dataset seed" rather than
        # cfg.seed — the subset should be the same across all runs regardless
        # of the training seed, otherwise different methods train on different
        # examples and comparisons aren't apples-to-apples.
        ds = ds.shuffle(seed=SUBSET_SEED).select(range(cfg.max_train_samples))

    # Keep only the unified schema columns
    keep = ["prompt", "target", "raw_question", "raw_answer"]
    drop = [c for c in ds.column_names if c not in keep]
    if drop:
        ds = ds.remove_columns(drop)
    return ds


# ──────────────────────────────────────────────────────────────────────────────
# Tokenization with prompt masking
# ──────────────────────────────────────────────────────────────────────────────

def encode_prompt(tokenizer, prompt: str, max_len: int):
    """Prompt ids with special tokens; left-truncated keeping BOS. -> (ids, truncated)"""
    ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    if len(ids) <= max_len:
        return ids, False
    keep_bos = tokenizer.bos_token_id is not None and ids[0] == tokenizer.bos_token_id
    head = ids[:1] if keep_bos else []
    return head + ids[len(ids) - (max_len - len(head)):], True


def build_sft_example(tokenizer, prompt: str, target: str,
                      max_source_len: int, max_target_len: int) -> Dict[str, Any]:
    """One (prompt, target) pair -> input_ids / labels with an exact prompt mask."""
    prompt_ids, prompt_truncated = encode_prompt(tokenizer, prompt, max_source_len)
    target_ids = tokenizer(" " + target, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
    target_truncated = len(target_ids) > max_target_len
    target_ids = target_ids[:max_target_len]
    return {
        "input_ids": prompt_ids + target_ids,
        "labels": [-100] * len(prompt_ids) + target_ids,
        "prompt_truncated": prompt_truncated,
        "target_truncated": target_truncated,
        "n_target_tokens": len(target_ids),
    }


class CausalLMCollator:
    """
    Right-pads input_ids with pad_token_id, attention_mask with 0, labels with
    -100. Padding never receives loss.
    """
    def __init__(self, tokenizer):
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, features):
        L = max(len(f["input_ids"]) for f in features)
        def pad(seq, v):
            return list(seq) + [v] * (L - len(seq))
        return {
            "input_ids": torch.tensor([pad(f["input_ids"], self.pad_token_id) for f in features]),
            "attention_mask": torch.tensor([pad([1] * len(f["input_ids"]), 0) for f in features]),
            "labels": torch.tensor([pad(f["labels"], -100) for f in features]),
        }


class SeededEpochSampler(Sampler):
    """
    Per-epoch permutations drawn up front from a dedicated torch.Generator
    seeded with data_seed. The order is therefore identical across methods with
    the same data seed, whatever else consumes the global RNG. Iteration k
    yields epoch k's permutation.
    """
    def __init__(self, n: int, n_epochs: int, data_seed: int):
        g = torch.Generator().manual_seed(int(data_seed))
        self.epoch_orders = [torch.randperm(n, generator=g).tolist() for _ in range(n_epochs)]
        self.n = n
        self._epoch = 0

    def __len__(self):
        return self.n

    def __iter__(self):
        if self._epoch >= len(self.epoch_orders):
            raise RuntimeError("SeededEpochSampler iterated more times than n_epochs")
        order = self.epoch_orders[self._epoch]
        self._epoch += 1
        return iter(order)

    def order_sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.epoch_orders).encode()).hexdigest()


def tokenize_examples(ds, tokenizer, cfg: RunConfig):
    """Tokenize every example (none dropped) and return (features, stats)."""
    feats = [build_sft_example(tokenizer, ex["prompt"], ex["target"], cfg.max_source_len, cfg.max_target_len)
             for ex in ds]
    n = len(feats)
    stats = {
        "n_examples": n,
        "n_prompt_truncated": sum(f["prompt_truncated"] for f in feats),
        "n_target_truncated": sum(f["target_truncated"] for f in feats),
        "n_empty_target": sum(f["n_target_tokens"] == 0 for f in feats),
        "n_supervised_tokens": sum(f["n_target_tokens"] for f in feats),
        "target_format_version": TARGET_FORMAT_VERSION,
    }
    for k in ("prompt_truncated", "target_truncated", "empty_target"):
        stats[f"rate_{k}"] = stats[f"n_{k}"] / n if n else None
    content = hashlib.sha256()
    for f in feats:
        content.update(json.dumps([f["input_ids"], f["labels"]]).encode())
    stats["tokenized_sha256"] = content.hexdigest()
    return [{"input_ids": f["input_ids"], "labels": f["labels"]} for f in feats], stats


def make_train_dataloader(features, stats, tokenizer, cfg: RunConfig) -> DataLoader:
    sampler = SeededEpochSampler(len(features), cfg.epochs, cfg.effective_data_seed)
    loader = DataLoader(features, batch_size=cfg.train_batch_size, sampler=sampler,
                        collate_fn=CausalLMCollator(tokenizer), num_workers=0)
    loader.data_stats = {**stats, "data_seed": cfg.effective_data_seed,
                         "data_order_sha256": sampler.order_sha256()}
    return loader


def build_train_dataloader(cfg: RunConfig, tokenizer) -> DataLoader:
    """End-to-end: load raw → normalise → tokenize → seeded DataLoader."""
    ds = load_and_normalize_training_data(cfg)
    features, stats = tokenize_examples(ds, tokenizer, cfg)
    return make_train_dataloader(features, stats, tokenizer, cfg)
