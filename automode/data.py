"""
Dataset loading and tokenization.

Supports training corpora:
  - gsm8k        (7.5K examples, #### format)
  - math         (7.5K, \boxed{} format)
  - metamath     (40K, mixed GSM8K + MATH style — the AdaGradSelect corpus)
  - alpaca_gpt4  (52K, instruction-following)

All training pipelines produce a single tokenized DataLoader with input_ids,
attention_mask, and labels (prompt tokens masked to -100).

Evaluation datasets (GSM8K-test, MATH-test, MMLU, ARC-C) are handled in eval.py.
"""

from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from datasets import load_dataset, Dataset, DatasetDict
from datasets.utils.logging import disable_progress_bar
from torch.utils.data import DataLoader

from automode.config import RunConfig

disable_progress_bar()


# ──────────────────────────────────────────────────────────────────────────────
# Prompt builders for each training track
# ──────────────────────────────────────────────────────────────────────────────

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

def load_gsm8k_raw() -> DatasetDict:
    return load_dataset("openai/gsm8k", "main")


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
        raw = load_gsm8k_raw()["train"]
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
        ds = ds.shuffle(seed=42).select(range(cfg.max_train_samples))

    # Keep only the unified schema columns
    keep = ["prompt", "target", "raw_question", "raw_answer"]
    drop = [c for c in ds.column_names if c not in keep]
    if drop:
        ds = ds.remove_columns(drop)
    return ds


# ──────────────────────────────────────────────────────────────────────────────
# Tokenization with prompt masking
# ──────────────────────────────────────────────────────────────────────────────

def tokenize_sft_batch(batch, tokenizer, max_source_len=384, max_target_len=128):
    """
    Tokenise a batch of (prompt, target) pairs for SFT with prompt-masking.

    The label tensor has:
      - prompt tokens → -100 (ignored by cross-entropy)
      - target tokens → their token IDs
      - padding       → -100 (applied by the collator)

    This ensures loss is computed ONLY on the answer tokens — critical for
    GSM8K where reasoning quality matters. The 2B paper's original bug was
    computing loss over the full sequence including the prompt.
    """
    prompts = batch["prompt"]
    targets = batch["target"]

    # Tokenize prompts alone to get exact prompt lengths
    prompt_only = tokenizer(
        prompts,
        max_length=max_source_len,
        truncation=True, padding=False, add_special_tokens=True,
    )

    # Tokenize full (prompt + " " + target) sequences
    full_texts = [f"{p} {t}" for p, t in zip(prompts, targets)]
    tokenized = tokenizer(
        full_texts,
        max_length=max_source_len + max_target_len,
        truncation=True, padding=False, add_special_tokens=True,
    )

    labels = []
    for i, input_ids in enumerate(tokenized["input_ids"]):
        prompt_len = len(prompt_only["input_ids"][i])
        lbl = list(input_ids)
        lbl[:prompt_len] = [-100] * min(prompt_len, len(lbl))
        labels.append(lbl)
    tokenized["labels"] = labels
    return tokenized


class CausalLMCollator:
    """
    Pads input_ids, attention_mask, and labels to the longest sequence in the
    batch. Label padding is -100 (ignored by loss).
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        # Separate out labels so we can pad them with -100 (not pad_token_id)
        labels_list = [f["labels"] for f in features]
        feat_no_labels = [
            {k: v for k, v in f.items() if k != "labels"} for f in features
        ]
        batch = self.tokenizer.pad(
            feat_no_labels, padding=True, return_tensors="pt",
        )
        # Pad labels with -100
        max_len = batch["input_ids"].size(1)
        padded_labels = []
        for lbl in labels_list:
            needed = max_len - len(lbl)
            if needed > 0:
                # Right pad with -100 to match right-padded input_ids
                lbl = list(lbl) + [-100] * needed
            else:
                lbl = list(lbl)[:max_len]
            padded_labels.append(lbl)
        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch


def build_train_dataloader(cfg: RunConfig, tokenizer) -> DataLoader:
    """
    End-to-end: load raw → normalise → tokenize → DataLoader.

    Called by run_experiment().
    """
    ds = load_and_normalize_training_data(cfg)

    tokenized = ds.map(
        lambda b: tokenize_sft_batch(
            b, tokenizer, cfg.max_source_len, cfg.max_target_len,
        ),
        batched=True,
        remove_columns=ds.column_names,
    )
    collator = CausalLMCollator(tokenizer)
    return DataLoader(
        tokenized,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=2,
        pin_memory=True,
    )
