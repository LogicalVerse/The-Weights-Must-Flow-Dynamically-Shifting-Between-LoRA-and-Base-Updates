"""
Evaluation on GSM8K, MATH, MMLU, ARC-Challenge.

All evaluators share a batched-generation pattern with num_return_sequences=k
for majority voting, which is ~5× faster than k sequential generate() calls.

Output format for every eval:
  {
    "benchmark": "gsm8k",
    "metric_name": "maj@1",
    "metric_value": 0.754,
    "n_correct": 995,
    "n_total": 1319,
    "records": [  # only if cfg.save_eval_predictions
      {"question": ..., "pred": ..., "gold": ..., "correct": 1},
      ...
    ],
  }
"""

from __future__ import annotations
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from automode.config import RunConfig
from automode.train import save_json


# ──────────────────────────────────────────────────────────────────────────────
# Answer extractors
# ──────────────────────────────────────────────────────────────────────────────

def extract_hash_answer(text: str) -> Optional[str]:
    """Extract the number after #### (GSM8K format). Returns None if absent."""
    if text is None:
        return None
    matches = re.findall(r"####\s*([^\n]+)", text)
    if matches:
        return matches[-1].strip().rstrip(".")
    # Fallback: last number in the text
    nums = re.findall(r"[-]?\d[\d,\.]*", text)
    return nums[-1].replace(",", "") if nums else None


def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract content inside \\boxed{...} (MATH format). Returns last match."""
    if text is None:
        return None
    # Handle nested braces in \boxed{...} with a simple bracket counter
    matches = []
    i = 0
    while i < len(text):
        idx = text.find("\\boxed{", i)
        if idx == -1:
            break
        j = idx + len("\\boxed{")
        depth = 1
        content_start = j
        while j < len(text) and depth > 0:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            j += 1
        if depth == 0:
            matches.append(text[content_start:j - 1])
        i = j
    if matches:
        return matches[-1].strip()
    return None


def normalize_numeric(s: Optional[str]) -> str:
    """Standard numeric normalization — strip $, commas, whitespace, lower case."""
    if s is None:
        return ""
    return s.strip().replace("$", "").replace(",", "").replace(" ", "").lower()


def extract_option_letter(text: str, allowed: str = "ABCDE") -> Optional[str]:
    matches = re.findall(rf"\b([{allowed}])\b", text.strip())
    return matches[-1] if matches else None


def majority_vote(items: List[str]) -> Optional[str]:
    items = [x for x in items if x is not None and str(x).strip() != ""]
    if not items:
        return None
    return Counter(items).most_common(1)[0][0]


# ──────────────────────────────────────────────────────────────────────────────
# Shared batched-generation helper
# ──────────────────────────────────────────────────────────────────────────────

def batched_generate(
    model, tokenizer, prompts: List[str], cfg: RunConfig,
    max_new_tokens: Optional[int] = None,
    num_return_sequences: int = 1,
    do_sample: Optional[bool] = None,
) -> List[List[str]]:
    """
    Generate `num_return_sequences` outputs per prompt in a single batched call.

    Returns: list of length len(prompts); each item is a list of k generated
    strings (the model output, prompt stripped).
    """
    # Left-pad for decoder-only batched generation
    tokenizer.padding_side = "left"

    enc = tokenizer(
        prompts, return_tensors="pt",
        padding=True, truncation=True, max_length=cfg.max_source_len,
    )
    enc = {k: v.to(model.device) for k, v in enc.items()}

    gen_kwargs = {
        "max_new_tokens": max_new_tokens or cfg.max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
        "num_return_sequences": num_return_sequences,
    }
    sample = cfg.do_sample if do_sample is None else do_sample
    if sample:
        gen_kwargs.update({
            "do_sample": True,
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
        })
    else:
        gen_kwargs.update({"do_sample": False, "num_beams": cfg.num_beams})

    with torch.inference_mode():
        outputs = model.generate(**enc, **gen_kwargs)

    input_len = enc["input_ids"].shape[1]
    gen_tokens = outputs[:, input_len:]
    decoded = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
    # Reshape: [B*k, ...] -> list of B lists of k strings
    B = len(prompts)
    k = num_return_sequences
    grouped = [decoded[i * k:(i + 1) * k] for i in range(B)]

    tokenizer.padding_side = "right"  # restore
    return grouped


# ──────────────────────────────────────────────────────────────────────────────
# GSM8K
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_gsm8k(model, tokenizer, cfg: RunConfig, paths: Dict[str, Path]) -> Dict[str, Any]:
    """
    maj@1 on GSM8K test (1319 examples).

    For each question, generate k samples with temperature=0.7, top_p=0.95,
    extract the #### answer from each, take majority vote, compare to gold.
    """
    from automode.data import GSM8K_PROMPT_TEMPLATE
    model.eval()
    model.config.use_cache = True

    ds = load_dataset("openai/gsm8k", "main", split="test")
    if cfg.max_eval_samples is not None:
        ds = ds.select(range(min(cfg.max_eval_samples, len(ds))))

    k = max(1, cfg.sampling_k)
    correct = 0
    records = []

    for i in tqdm(range(0, len(ds), cfg.eval_batch_size), desc="GSM8K eval", leave=False):
        batch = ds[i:i + cfg.eval_batch_size]
        prompts = [
            GSM8K_PROMPT_TEMPLATE.format(q=q.strip())
            for q in batch["question"]
        ]
        gold_answers = [extract_hash_answer(a) for a in batch["answer"]]

        generations = batched_generate(
            model, tokenizer, prompts, cfg,
            num_return_sequences=k, do_sample=True,
        )

        for j, prompt in enumerate(prompts):
            preds = [extract_hash_answer(text) for text in generations[j]]
            preds_norm = [normalize_numeric(p) for p in preds]
            maj = majority_vote(preds_norm)
            gold_norm = normalize_numeric(gold_answers[j])
            ok = int(maj is not None and maj == gold_norm)
            correct += ok
            records.append({
                "question": batch["question"][j],
                "pred": maj,
                "gold": gold_norm,
                "samples": preds,
                "correct": ok,
            })

    acc = correct / len(ds) if len(ds) > 0 else 0.0
    result = {
        "benchmark": "gsm8k",
        "metric_name": "maj@1",
        "metric_value": acc,
        "n_correct": correct,
        "n_total": len(ds),
    }
    if cfg.save_eval_predictions:
        result["records"] = records
    save_json(result, paths["evals"] / "gsm8k_eval.json")
    return {"gsm8k_maj1": acc, "gsm8k_n_correct": correct, "gsm8k_n_total": len(ds)}


# ──────────────────────────────────────────────────────────────────────────────
# MATH
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_math(model, tokenizer, cfg: RunConfig, paths: Dict[str, Path]) -> Dict[str, Any]:
    """
    Exact-match accuracy on MATH test set using \\boxed{} extraction.

    Uses greedy decoding (do_sample=False, k=1) by convention — MATH answers
    tend to have stable final expressions that don't benefit from sampling.
    """
    from automode.data import MATH_PROMPT_TEMPLATE, load_math_raw
    model.eval()
    model.config.use_cache = True

    raw = load_math_raw()
    ds = raw["test"] if "test" in raw else raw[list(raw.keys())[0]]
    if cfg.max_eval_samples is not None:
        ds = ds.select(range(min(cfg.max_eval_samples, len(ds))))

    correct = 0
    records = []

    for i in tqdm(range(0, len(ds), cfg.eval_batch_size), desc="MATH eval", leave=False):
        batch = ds[i:i + cfg.eval_batch_size]
        prompts = [
            MATH_PROMPT_TEMPLATE.format(q=(q or "").strip())
            for q in batch.get("problem", batch.get("question", [""] * len(batch.get("solution", []))))
        ]
        # Gold answer from either "answer" field or extracted from solution
        gold_raw = batch.get("answer", batch.get("solution", [""] * len(prompts)))
        gold_answers = [
            normalize_numeric(extract_boxed_answer(a) or a)
            for a in gold_raw
        ]

        generations = batched_generate(
            model, tokenizer, prompts, cfg,
            num_return_sequences=1,
            do_sample=False,
            max_new_tokens=max(cfg.max_new_tokens, 512),
        )

        for j, prompt in enumerate(prompts):
            text = generations[j][0]
            pred = normalize_numeric(extract_boxed_answer(text))
            ok = int(pred != "" and pred == gold_answers[j])
            correct += ok
            records.append({
                "question": prompts[j],
                "pred": pred,
                "gold": gold_answers[j],
                "generated": text,
                "correct": ok,
            })

    acc = correct / len(ds) if len(ds) > 0 else 0.0
    result = {
        "benchmark": "math",
        "metric_name": "exact_match",
        "metric_value": acc,
        "n_correct": correct,
        "n_total": len(ds),
    }
    if cfg.save_eval_predictions:
        result["records"] = records
    save_json(result, paths["evals"] / "math_eval.json")
    return {"math_acc": acc, "math_n_correct": correct, "math_n_total": len(ds)}


# ──────────────────────────────────────────────────────────────────────────────
# MMLU (5-shot)
# ──────────────────────────────────────────────────────────────────────────────

MMLU_HEADER = (
    "The following are multiple-choice questions about {subject}. "
    "Answer with a single letter (A, B, C, or D).\n\n"
)


def _format_mmlu_example(ex, include_answer: bool = False) -> str:
    letters = "ABCD"
    out = ex["question"].strip() + "\n"
    for i, c in enumerate(ex["choices"]):
        out += f"{letters[i]}. {c}\n"
    out += "Answer:"
    if include_answer:
        out += f" {letters[ex['answer']]}\n\n"
    return out


def evaluate_mmlu(model, tokenizer, cfg: RunConfig, paths: Dict[str, Path],
                  n_shot: int = 5, max_samples: Optional[int] = None) -> Dict[str, Any]:
    """
    5-shot MMLU accuracy across all 57 subjects.

    Uses the auxiliary_train/dev split for few-shot exemplars (standard convention).
    """
    model.eval()
    model.config.use_cache = True

    ds = load_dataset("cais/mmlu", "all")
    test_ds = ds["test"]
    dev_ds = ds["dev"] if "dev" in ds else ds.get("validation")

    # Cap samples if requested (for smoke tests)
    if max_samples is not None and max_samples < len(test_ds):
        test_ds = test_ds.shuffle(seed=cfg.seed).select(range(max_samples))

    # Group dev by subject for few-shot exemplars
    subject_to_dev: Dict[str, List] = {}
    for ex in dev_ds:
        subj = ex["subject"]
        subject_to_dev.setdefault(subj, []).append(ex)

    correct = 0
    records = []

    for i in tqdm(range(0, len(test_ds), cfg.eval_batch_size),
                  desc="MMLU 5-shot eval", leave=False):
        batch = test_ds[i:i + cfg.eval_batch_size]
        prompts = []
        gold_letters = []
        subjects = batch["subject"]
        for j in range(len(subjects)):
            subj = subjects[j]
            shots = subject_to_dev.get(subj, [])[:n_shot]
            shot_text = "".join(_format_mmlu_example(s, include_answer=True) for s in shots)
            cur = {
                "question": batch["question"][j],
                "choices": batch["choices"][j],
                "answer": batch["answer"][j],
            }
            prompt = MMLU_HEADER.format(subject=subj.replace("_", " ")) + shot_text + _format_mmlu_example(cur, include_answer=False)
            prompts.append(prompt)
            gold_letters.append("ABCD"[cur["answer"]])

        generations = batched_generate(
            model, tokenizer, prompts, cfg,
            num_return_sequences=1, do_sample=False,
            max_new_tokens=8,
        )

        for j in range(len(prompts)):
            pred = extract_option_letter(generations[j][0], allowed="ABCD")
            ok = int(pred == gold_letters[j])
            correct += ok
            records.append({
                "subject": subjects[j],
                "pred": pred, "gold": gold_letters[j], "correct": ok,
            })

    acc = correct / len(test_ds) if len(test_ds) > 0 else 0.0
    result = {
        "benchmark": "mmlu", "metric_name": "accuracy_5shot",
        "metric_value": acc, "n_correct": correct, "n_total": len(test_ds),
    }
    if cfg.save_eval_predictions:
        result["records"] = records[:5000]  # MMLU has 14K, cap the file size
    save_json(result, paths["evals"] / "mmlu_eval.json")
    return {"mmlu_5shot_acc": acc}


# ──────────────────────────────────────────────────────────────────────────────
# ARC-Challenge
# ──────────────────────────────────────────────────────────────────────────────

ARC_PROMPT = (
    "Answer the science multiple-choice question. "
    "Respond with only the option letter.\n\n"
    "Question: {q}\n{choices}\nAnswer:"
)


def evaluate_arc(model, tokenizer, cfg: RunConfig, paths: Dict[str, Path]) -> Dict[str, Any]:
    model.eval()
    model.config.use_cache = True
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    if cfg.max_eval_samples is not None:
        ds = ds.select(range(min(cfg.max_eval_samples, len(ds))))

    correct = 0
    records = []
    letters_all = "ABCDE"

    for i in tqdm(range(0, len(ds), cfg.eval_batch_size), desc="ARC-C eval", leave=False):
        batch = ds[i:i + cfg.eval_batch_size]
        prompts = []
        gold_letters = []
        for j in range(len(batch["question"])):
            ch = batch["choices"][j]
            labels = ch["label"]  # these are "A"/"B"/... or "1"/"2"/...
            texts = ch["text"]
            # Normalize to ABCDE
            if any(L.isdigit() for L in labels):
                disp_labels = letters_all[:len(labels)]
                gold_idx = labels.index(batch["answerKey"][j])
                gold = disp_labels[gold_idx]
            else:
                disp_labels = labels
                gold = batch["answerKey"][j]
            choices_block = "\n".join(
                f"{L}. {T}" for L, T in zip(disp_labels, texts)
            )
            prompts.append(ARC_PROMPT.format(q=batch["question"][j], choices=choices_block))
            gold_letters.append(gold)

        generations = batched_generate(
            model, tokenizer, prompts, cfg,
            num_return_sequences=1, do_sample=False, max_new_tokens=8,
        )
        for j in range(len(prompts)):
            pred = extract_option_letter(generations[j][0], allowed=letters_all)
            ok = int(pred == gold_letters[j])
            correct += ok
            records.append({"pred": pred, "gold": gold_letters[j], "correct": ok})

    acc = correct / len(ds) if len(ds) > 0 else 0.0
    result = {
        "benchmark": "arc_c", "metric_name": "accuracy",
        "metric_value": acc, "n_correct": correct, "n_total": len(ds),
    }
    if cfg.save_eval_predictions:
        result["records"] = records
    save_json(result, paths["evals"] / "arc_c_eval.json")
    return {"arc_c_acc": acc}


# ──────────────────────────────────────────────────────────────────────────────
# Orchestrator — called from train.py
# ──────────────────────────────────────────────────────────────────────────────

def run_evals(cfg: RunConfig, model, tokenizer, paths: Dict[str, Path]) -> Dict[str, Any]:
    """Run all benchmarks listed in cfg.eval_benchmarks. Return merged metrics dict."""
    out: Dict[str, Any] = {}
    benches = cfg.eval_benchmarks or ()

    if "gsm8k" in benches:
        try:
            out.update(evaluate_gsm8k(model, tokenizer, cfg, paths))
        except Exception as e:
            print(f"[eval] gsm8k failed: {e}")
    if "math" in benches:
        try:
            out.update(evaluate_math(model, tokenizer, cfg, paths))
        except Exception as e:
            print(f"[eval] math failed: {e}")
    if "mmlu" in benches:
        try:
            out.update(evaluate_mmlu(model, tokenizer, cfg, paths))
        except Exception as e:
            print(f"[eval] mmlu failed: {e}")
    if "arc_c" in benches:
        try:
            out.update(evaluate_arc(model, tokenizer, cfg, paths))
        except Exception as e:
            print(f"[eval] arc_c failed: {e}")
    return out
