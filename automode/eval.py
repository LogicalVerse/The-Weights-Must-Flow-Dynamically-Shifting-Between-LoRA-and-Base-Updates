"""
Evaluation on GSM8K, MATH, MMLU, ARC-Challenge.

GSM8K protocol (GSM8K_EXTRACTION_VERSION):
  * k samples per question (sampling_k; core: k=3, T=0.7, top_p=0.95,
    max_new_tokens=256), prompt = submitted template, encoded exactly like
    training prompts (data.encode_prompt), left-padded for batched generation.
  * Sampling RNG: torch.manual_seed(evaluation_seed + first_item_index) before
    every batch, so it does not depend on how much RNG training consumed.
  * Extraction: text after the FIRST "####", first number in it. If there is
    no "####", the LAST number in the response ("last_number_fallback"). No
    number at all -> invalid sample.
  * Normalisation: drop "," and "$", strip a trailing ".", parse as Decimal,
    canonical plain notation without trailing zeros ("18.00" -> "18").
    Unparseable -> invalid sample.
  * Vote: plurality over all k samples where every invalid sample is a vote
    for a single INVALID option; ties go to the option that occurs first in
    sample order. If INVALID wins, the item is incorrect. No votes are removed.
  * Metric: gsm8k_maj{k} = n_correct / n_total; every response, extraction,
    gold, item id and correctness is stored.

Every evaluator raises on failure; run_evals does not swallow exceptions.
"""

from __future__ import annotations
import hashlib
import re
import time
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import load_dataset
from tqdm import tqdm

from automode.config import RunConfig
from automode.train import save_json, autocast_ctx

GSM8K_EXTRACTION_VERSION = "gsm8k_first_hash_then_last_number_v2"
# Fixed seed for the MMLU retention subset: identical across models and seeds.
MC_SUBSET_SEED = 1234
INVALID = "<invalid>"

_NUMBER = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?")


# ──────────────────────────────────────────────────────────────────────────────
# Answer extractors
# ──────────────────────────────────────────────────────────────────────────────

def extract_gsm8k_answer(text: Optional[str]) -> Tuple[Optional[str], str]:
    """-> (raw number string or None, extraction method)."""
    if not text:
        return None, "none"
    m = re.search(r"####([^\n]*)", text)
    if m:
        n = _NUMBER.search(m.group(1))
        return (n.group(0), "hash") if n else (None, "hash_without_number")
    nums = _NUMBER.findall(text)
    return (nums[-1], "last_number_fallback") if nums else (None, "none")


def normalize_number(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    s = s.replace(",", "").replace("$", "").strip().rstrip(".")
    try:
        d = Decimal(s)
    except InvalidOperation:
        return None
    if not d.is_finite():
        return None
    out = format(d.normalize(), "f")
    if "." in out:
        out = out.rstrip("0").rstrip(".")
    return "0" if out in ("-0", "") else out


def plurality_vote(answers: List[Optional[str]]) -> Optional[str]:
    """Plurality with invalid (None) as one option; ties -> first occurrence."""
    votes = [INVALID if a is None else a for a in answers]
    if not votes:
        return None
    counts = Counter(votes)
    best = max(counts.values())
    winner = next(v for v in votes if counts[v] == best)
    return None if winner == INVALID else winner


def extract_hash_answer(text: str) -> Optional[str]:
    """Back-compat wrapper: normalised GSM8K answer or None."""
    return normalize_number(extract_gsm8k_answer(text)[0])


def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract content inside \\boxed{...} (MATH format). Returns last match."""
    if text is None:
        return None
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
    """String normalisation used by the MATH evaluator."""
    if s is None:
        return ""
    return s.strip().replace("$", "").replace(",", "").replace(" ", "").lower()


def extract_option_letter(text: str, allowed: str = "ABCDE") -> Optional[str]:
    matches = re.findall(rf"\b([{allowed}])\b", text.strip())
    return matches[-1] if matches else None


# ──────────────────────────────────────────────────────────────────────────────
# Shared batched-generation helper
# ──────────────────────────────────────────────────────────────────────────────

def batched_generate(
    model, tokenizer, prompt_ids: List[List[int]], cfg: RunConfig,
    max_new_tokens: Optional[int] = None,
    num_return_sequences: int = 1,
    do_sample: Optional[bool] = None,
) -> Tuple[List[List[str]], List[List[int]]]:
    """
    Generate from already-encoded prompts (left-padded here). Returns
    (texts, n_generated_tokens), each a list per prompt of k entries.
    """
    device = next(model.parameters()).device
    L = max(len(x) for x in prompt_ids)
    pad = tokenizer.pad_token_id
    input_ids = torch.tensor([[pad] * (L - len(x)) + list(x) for x in prompt_ids], device=device)
    attention_mask = torch.tensor([[0] * (L - len(x)) + [1] * len(x) for x in prompt_ids], device=device)

    gen_kwargs = {
        "max_new_tokens": max_new_tokens or cfg.max_new_tokens,
        "pad_token_id": pad,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
        "num_return_sequences": num_return_sequences,
    }
    sample = cfg.do_sample if do_sample is None else do_sample
    if sample:
        gen_kwargs.update({"do_sample": True, "temperature": cfg.temperature, "top_p": cfg.top_p})
    else:
        gen_kwargs.update({"do_sample": False, "num_beams": cfg.num_beams})

    with torch.inference_mode(), autocast_ctx(cfg, device):
        outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs)

    gen_tokens = outputs[:, L:].tolist()
    lengths = []
    for seq in gen_tokens:
        n = len(seq)
        for i, t in enumerate(seq):
            if t == tokenizer.eos_token_id or t == pad:
                n = i + (t == tokenizer.eos_token_id)
                break
        lengths.append(n)
    decoded = tokenizer.batch_decode(outputs[:, L:], skip_special_tokens=True)
    k = num_return_sequences
    B = len(prompt_ids)
    return ([decoded[i * k:(i + 1) * k] for i in range(B)],
            [lengths[i * k:(i + 1) * k] for i in range(B)])


def _prepare_eval(model, cfg: RunConfig):
    model.eval()
    model.config.use_cache = True


# ──────────────────────────────────────────────────────────────────────────────
# GSM8K
# ──────────────────────────────────────────────────────────────────────────────

def load_gsm8k_test(cfg: RunConfig):
    ds = load_dataset("openai/gsm8k", "main", split="test", revision=cfg.dataset_revision)
    if cfg.max_eval_samples is not None:
        ds = ds.select(range(min(cfg.max_eval_samples, len(ds))))
    return ds


def evaluate_gsm8k(model, tokenizer, cfg: RunConfig, paths: Dict[str, Path],
                   dataset=None) -> Dict[str, Any]:
    """maj@k on the GSM8K test split (first max_eval_samples items if set)."""
    from automode.data import GSM8K_PROMPT_TEMPLATE, encode_prompt
    _prepare_eval(model, cfg)
    ds = load_gsm8k_test(cfg) if dataset is None else dataset
    k = cfg.sampling_k
    if k > 1 and not cfg.do_sample:
        raise ValueError("sampling_k > 1 requires do_sample=True")

    records = []
    n_correct = n_invalid_samples = n_prompt_truncated = n_hit_cap = gen_tokens = 0
    t0 = time.perf_counter()
    for start in tqdm(range(0, len(ds), cfg.eval_batch_size), desc="GSM8K eval", leave=False):
        batch = ds[start:start + cfg.eval_batch_size]
        encoded = [encode_prompt(tokenizer, GSM8K_PROMPT_TEMPLATE.format(q=q.strip()), cfg.max_source_len)
                   for q in batch["question"]]
        n_prompt_truncated += sum(t for _, t in encoded)
        torch.manual_seed(cfg.evaluation_seed + start)
        texts, lengths = batched_generate(model, tokenizer, [ids for ids, _ in encoded], cfg,
                                          num_return_sequences=k, do_sample=cfg.do_sample)
        for j, question in enumerate(batch["question"]):
            idx = start + j
            gold = normalize_number(extract_gsm8k_answer(batch["answer"][j])[0])
            if gold is None:
                raise ValueError(f"Could not parse GSM8K gold answer for item {idx}")
            samples = []
            for text, n_tok in zip(texts[j], lengths[j]):
                raw, method = extract_gsm8k_answer(text)
                norm = normalize_number(raw)
                n_invalid_samples += norm is None
                n_hit_cap += n_tok >= cfg.max_new_tokens
                gen_tokens += n_tok
                samples.append({"text": text, "extracted": raw, "normalized": norm,
                                "valid": norm is not None, "extraction": method,
                                "n_generated_tokens": n_tok})
            pred = plurality_vote([s["normalized"] for s in samples])
            ok = int(pred is not None and pred == gold)
            n_correct += ok
            records.append({
                "item_id": f"gsm8k_test_{idx:04d}",
                "question_sha256": hashlib.sha256(question.encode()).hexdigest()[:16],
                "question": question, "gold_raw": batch["answer"][j], "gold": gold,
                "samples": samples, "prediction": pred, "correct": ok,
            })
    eval_time = time.perf_counter() - t0

    n_total = len(records)
    if n_total != len(ds):
        raise RuntimeError(f"Evaluated {n_total} of {len(ds)} items")
    metric = f"gsm8k_maj{k}"
    summary = {
        metric: n_correct / n_total if n_total else None,
        "gsm8k_n_correct": n_correct,
        "gsm8k_n_total": n_total,
        "gsm8k_sampling_k": k,
        "gsm8k_n_invalid_samples": n_invalid_samples,
        "gsm8k_n_items_invalid_won": sum(r["prediction"] is None for r in records),
        "gsm8k_n_prompt_truncated": n_prompt_truncated,
        "gsm8k_n_samples_hit_max_new_tokens": n_hit_cap,
        "gsm8k_generated_tokens": gen_tokens,
        "gsm8k_eval_time_sec": eval_time,
        "gsm8k_extraction_version": GSM8K_EXTRACTION_VERSION,
    }
    result = {"benchmark": "gsm8k", "metric_name": metric, "metric_value": summary[metric],
              **summary, "decoding": {"do_sample": cfg.do_sample, "temperature": cfg.temperature,
                                      "top_p": cfg.top_p, "max_new_tokens": cfg.max_new_tokens,
                                      "evaluation_seed": cfg.evaluation_seed,
                                      "eval_batch_size": cfg.eval_batch_size}}
    if cfg.save_eval_predictions:
        result["records"] = records
    save_json(result, paths["evals"] / "gsm8k_eval.json")
    return summary


# ──────────────────────────────────────────────────────────────────────────────
# MATH
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_math(model, tokenizer, cfg: RunConfig, paths: Dict[str, Path]) -> Dict[str, Any]:
    """
    Exact-match accuracy on MATH test set using \\boxed{} extraction.
    Greedy decoding (do_sample=False, k=1).
    """
    from automode.data import MATH_PROMPT_TEMPLATE, load_math_raw, encode_prompt
    _prepare_eval(model, cfg)

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
        gold_raw = batch.get("answer", batch.get("solution", [""] * len(prompts)))
        gold_answers = [normalize_numeric(extract_boxed_answer(a) or a) for a in gold_raw]
        ids = [encode_prompt(tokenizer, p, cfg.max_source_len)[0] for p in prompts]
        generations, _ = batched_generate(
            model, tokenizer, ids, cfg, num_return_sequences=1, do_sample=False,
            max_new_tokens=max(cfg.max_new_tokens, 512),
        )
        for j in range(len(prompts)):
            text = generations[j][0]
            pred = normalize_numeric(extract_boxed_answer(text))
            ok = int(pred != "" and pred == gold_answers[j])
            correct += ok
            records.append({"item_id": f"math_test_{i + j:05d}", "question": prompts[j], "pred": pred,
                            "gold": gold_answers[j], "generated": text, "correct": ok})

    acc = correct / len(ds) if len(ds) > 0 else None
    result = {"benchmark": "math", "metric_name": "exact_match", "metric_value": acc,
              "n_correct": correct, "n_total": len(ds)}
    if cfg.save_eval_predictions:
        result["records"] = records
    save_json(result, paths["evals"] / "math_eval.json")
    return {"math_acc": acc, "math_n_correct": correct, "math_n_total": len(ds)}


# ──────────────────────────────────────────────────────────────────────────────
# MMLU (few-shot) and ARC-Challenge — retention evals with their own context
# budget (cfg.mc_max_source_len). The test question and all its choices are
# always kept whole; if the prompt is too long, shots are dropped one at a time
# (deliberately, logged per item). A question that does not fit even 0-shot
# raises instead of being truncated.
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


def build_mmlu_prompt_ids(tokenizer, subject: str, shots: List[dict], cur: dict,
                          max_len: int) -> Tuple[List[int], int]:
    """-> (prompt ids, number of shots actually used)."""
    header = MMLU_HEADER.format(subject=subject.replace("_", " "))
    question = _format_mmlu_example(cur, include_answer=False)
    for n in range(len(shots), -1, -1):
        text = header + "".join(_format_mmlu_example(s, include_answer=True) for s in shots[:n]) + question
        ids = tokenizer(text, add_special_tokens=True)["input_ids"]
        if len(ids) <= max_len:
            return ids, n
    raise ValueError(f"MMLU question does not fit in mc_max_source_len={max_len} even 0-shot")


def evaluate_mmlu(model, tokenizer, cfg: RunConfig, paths: Dict[str, Path],
                  n_shot: Optional[int] = None, max_samples: Optional[int] = None) -> Dict[str, Any]:
    """
    Few-shot MMLU accuracy (dev split exemplars). max_samples defaults to
    cfg.max_eval_samples; the subset is a fixed shuffle (MC_SUBSET_SEED), the
    same for every model and seed.
    """
    _prepare_eval(model, cfg)
    n_shot = cfg.mmlu_n_shot if n_shot is None else n_shot
    max_samples = cfg.max_eval_samples if max_samples is None else max_samples

    ds = load_dataset("cais/mmlu", "all")
    test_ds = ds["test"]
    dev_ds = ds["dev"] if "dev" in ds else ds.get("validation")
    item_index = list(range(len(test_ds)))
    if max_samples is not None and max_samples < len(test_ds):
        test_ds = test_ds.add_column("orig_index", item_index).shuffle(seed=MC_SUBSET_SEED).select(range(max_samples))
        item_index = test_ds["orig_index"]

    subject_to_dev: Dict[str, List] = {}
    for ex in dev_ds:
        subject_to_dev.setdefault(ex["subject"], []).append(ex)

    correct = 0
    records = []
    for i in tqdm(range(0, len(test_ds), cfg.eval_batch_size), desc="MMLU eval", leave=False):
        batch = test_ds[i:i + cfg.eval_batch_size]
        prompt_ids, gold_letters, shots_used = [], [], []
        for j, subj in enumerate(batch["subject"]):
            cur = {"question": batch["question"][j], "choices": batch["choices"][j], "answer": batch["answer"][j]}
            ids, used = build_mmlu_prompt_ids(tokenizer, subj, subject_to_dev.get(subj, [])[:n_shot], cur,
                                              cfg.mc_max_source_len)
            prompt_ids.append(ids)
            shots_used.append(used)
            gold_letters.append("ABCD"[cur["answer"]])
        generations, _ = batched_generate(model, tokenizer, prompt_ids, cfg, num_return_sequences=1,
                                          do_sample=False, max_new_tokens=8)
        for j in range(len(prompt_ids)):
            pred = extract_option_letter(generations[j][0], allowed="ABCD")
            ok = int(pred == gold_letters[j])
            correct += ok
            records.append({"item_id": f"mmlu_test_{item_index[i + j]:05d}", "subject": batch["subject"][j],
                            "n_shots": shots_used[j], "pred": pred, "gold": gold_letters[j], "correct": ok})

    n = len(test_ds)
    acc = correct / n if n else None
    shots = [r["n_shots"] for r in records]
    summary = {"mmlu_acc": acc, "mmlu_n_correct": correct, "mmlu_n_total": n,
               "mmlu_requested_shots": n_shot,
               "mmlu_n_items_reduced_shots": sum(s < n_shot for s in shots),
               "mmlu_mean_shots": sum(shots) / n if n else None}
    result = {"benchmark": "mmlu", "metric_name": "accuracy_fewshot", "metric_value": acc,
              **summary, "mc_max_source_len": cfg.mc_max_source_len, "subset_seed": MC_SUBSET_SEED}
    if cfg.save_eval_predictions:
        result["records"] = records
    save_json(result, paths["evals"] / "mmlu_eval.json")
    return summary


ARC_PROMPT = (
    "Answer the science multiple-choice question. "
    "Respond with only the option letter.\n\n"
    "Question: {q}\n{choices}\nAnswer:"
)


def evaluate_arc(model, tokenizer, cfg: RunConfig, paths: Dict[str, Path]) -> Dict[str, Any]:
    _prepare_eval(model, cfg)
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    if cfg.max_eval_samples is not None:
        ds = ds.select(range(min(cfg.max_eval_samples, len(ds))))

    correct = 0
    records = []
    letters_all = "ABCDE"
    for i in tqdm(range(0, len(ds), cfg.eval_batch_size), desc="ARC-C eval", leave=False):
        batch = ds[i:i + cfg.eval_batch_size]
        prompt_ids, gold_letters = [], []
        for j in range(len(batch["question"])):
            ch = batch["choices"][j]
            labels, texts = ch["label"], ch["text"]
            if any(L.isdigit() for L in labels):
                disp_labels = letters_all[:len(labels)]
                gold = disp_labels[labels.index(batch["answerKey"][j])]
            else:
                disp_labels = labels
                gold = batch["answerKey"][j]
            choices_block = "\n".join(f"{L}. {T}" for L, T in zip(disp_labels, texts))
            ids = tokenizer(ARC_PROMPT.format(q=batch["question"][j], choices=choices_block),
                            add_special_tokens=True)["input_ids"]
            if len(ids) > cfg.mc_max_source_len:
                raise ValueError(f"ARC item {i + j} exceeds mc_max_source_len={cfg.mc_max_source_len}")
            prompt_ids.append(ids)
            gold_letters.append(gold)
        generations, _ = batched_generate(model, tokenizer, prompt_ids, cfg, num_return_sequences=1,
                                          do_sample=False, max_new_tokens=8)
        for j in range(len(prompt_ids)):
            pred = extract_option_letter(generations[j][0], allowed=letters_all)
            ok = int(pred == gold_letters[j])
            correct += ok
            records.append({"item_id": batch["id"][j], "pred": pred, "gold": gold_letters[j], "correct": ok})

    n = len(ds)
    acc = correct / n if n else None
    result = {"benchmark": "arc_c", "metric_name": "accuracy", "metric_value": acc,
              "n_correct": correct, "n_total": n, "mc_max_source_len": cfg.mc_max_source_len}
    if cfg.save_eval_predictions:
        result["records"] = records
    save_json(result, paths["evals"] / "arc_c_eval.json")
    return {"arc_c_acc": acc, "arc_c_n_correct": correct, "arc_c_n_total": n}


# ──────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ──────────────────────────────────────────────────────────────────────────────

EVALUATORS = {"gsm8k": evaluate_gsm8k, "math": evaluate_math, "mmlu": evaluate_mmlu, "arc_c": evaluate_arc}


def run_evals(cfg: RunConfig, model, tokenizer, paths: Dict[str, Path]) -> Dict[str, Any]:
    """Run every benchmark in cfg.eval_benchmarks. Any failure propagates."""
    unknown = set(cfg.eval_benchmarks) - set(EVALUATORS)
    if unknown:
        raise ValueError(f"Unknown eval benchmarks: {sorted(unknown)}")
    out: Dict[str, Any] = {}
    for name in cfg.eval_benchmarks:
        out.update(EVALUATORS[name](model, tokenizer, cfg, paths))
    return out
