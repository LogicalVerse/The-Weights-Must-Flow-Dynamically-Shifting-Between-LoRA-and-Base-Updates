"""
Post-hoc analysis from saved run artefacts (no torch).

switch_diagnostics() answers the reviewers' permanence / re-promotion questions
from dynamic/switch_history.json. Time is measured in optimizer UPDATES: the
allocation chosen at decision step s trains updates s+1 ... next decision, and
the initial allocation trains updates 1 ... first decision. Occupancy is
therefore update-weighted, never an average over irregular snapshots.

The summary helpers group runs by exact (variant, train_track, protocol_version)
and pair comparisons by seed or item_id, never by array position. With three
seeds, report the seed standard deviation descriptively; it is not a
significance test.
"""

from __future__ import annotations

import statistics
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


def _lid(name: str) -> int:
    return int(name.split("_")[1])


def _dist(values: List[float]) -> Optional[Dict[str, float]]:
    if not values:
        return None
    a = np.asarray(values, dtype=np.float64)
    q = np.percentile(a, [0, 25, 50, 75, 100])
    return {"n": int(a.size), "mean": float(a.mean()), "min": float(q[0]), "q25": float(q[1]),
            "median": float(q[2]), "q75": float(q[3]), "max": float(q[4])}


def _jaccard(a: set, b: set) -> Optional[float]:
    return len(a & b) / len(a | b) if (a | b) else None


def switch_diagnostics(history: List[dict], initial_modes: Dict[str, str], total_steps: int,
                       trainable_per_update: Optional[List[int]] = None) -> Dict[str, Any]:
    layers = sorted(initial_modes, key=_lid)
    events = sorted(history, key=lambda h: h["step"])
    current = dict(initial_modes)
    updates = {l: {"full_ft": 0, "lora": 0} for l in layers}
    per_layer = {l: {"initial_mode": initial_modes[l], "promotions": 0, "demotions": 0,
                     "re_promotions": 0} for l in layers}
    demoted_before = set()
    per_event = []
    prev_step, prev_selected = 0, None

    for h in events:
        for l in layers:
            updates[l][current[l]] += h["step"] - prev_step
        selected = {l for l, m in h["target_modes"].items() if m == "full_ft"}
        promos = demos = 0
        for t in h["transitions"]:
            l = t["layer"]
            if t["to"] == "full_ft":
                promos += 1
                per_layer[l]["promotions"] += 1
                if l in demoted_before:
                    per_layer[l]["re_promotions"] += 1
            else:
                demos += 1
                per_layer[l]["demotions"] += 1
                demoted_before.add(l)
        by_mode: Dict[str, List[float]] = {}
        for l, s in h["scores"].items():
            by_mode.setdefault(h["modes_before"][l], []).append(s)
        per_event.append({
            "step": h["step"],
            "n_selected_base": len(selected),
            "n_tied_at_threshold": h.get("n_tied_at_threshold"),
            "threshold": h.get("threshold"),
            "n_promotions": promos,
            "n_demotions": demos,
            "jaccard_with_previous_selected": None if prev_selected is None else _jaccard(selected, prev_selected),
            "score_distribution_by_mode_before": {m: _dist(v) for m, v in sorted(by_mode.items())},
        })
        current = dict(h["layer_modes"])
        prev_step, prev_selected = h["step"], selected
    for l in layers:
        updates[l][current[l]] += total_steps - prev_step

    out: Dict[str, Any] = {
        "total_optimizer_updates": total_steps,
        "n_layers": len(layers),
        "n_decision_events": len(events),
        "initial_modes_count": {m: sum(1 for l in layers if initial_modes[l] == m) for m in ("full_ft", "lora")},
        "per_event": per_event,
        "per_layer": {l: {**per_layer[l], "updates_in_base": updates[l]["full_ft"],
                          "updates_in_lora": updates[l]["lora"],
                          "base_occupancy_fraction": updates[l]["full_ft"] / total_steps}
                      for l in layers},
        "mean_base_occupancy_fraction": sum(updates[l]["full_ft"] for l in layers) / (total_steps * len(layers)),
        "totals": {k: sum(per_layer[l][k] for l in layers) for k in ("promotions", "demotions", "re_promotions")},
        "optimizer_step_weighted_trainable_params": (sum(trainable_per_update) / len(trainable_per_update))
        if trainable_per_update else None,
    }
    js = [e["jaccard_with_previous_selected"] for e in per_event if e["jaccard_with_previous_selected"] is not None]
    out["mean_successive_jaccard"] = {"value": sum(js) / len(js) if js else None, "n_pairs": len(js)}
    n_dem = out["totals"]["demotions"]
    out["layers_ever_demoted"] = {"count": len(demoted_before), "denominator": len(layers)}
    out["re_promotion"] = {
        "layers_re_promoted": sum(1 for l in layers if per_layer[l]["re_promotions"]),
        "denominator_layers_ever_demoted": len(demoted_before),
        "rate": (sum(1 for l in demoted_before if per_layer[l]["re_promotions"]) / len(demoted_before))
        if demoted_before else None,
        "note": "no demotions occurred, so re-promotion is undefined" if n_dem == 0 else None,
    }
    if events:
        first = events[0]
        first_set = sorted((l for l, m in first["target_modes"].items() if m == "full_ft"), key=_lid)
        stayed = [l for l in first_set
                  if all(h["layer_modes"][l] == "full_ft" for h in events)]
        out["first_selection"] = {"step": first["step"], "layers": [_lid(l) for l in first_set],
                                  "size": len(first_set)}
        out["first_selected_remaining_base_rest_of_training"] = {
            "count": len(stayed), "denominator": len(first_set),
            "fraction": len(stayed) / len(first_set) if first_set else None,
            "decision_events_after_first": len(events) - 1,
        }
    else:
        out["first_selection"] = None
        out["first_selected_remaining_base_rest_of_training"] = None
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Run summaries for figures/tables
# ──────────────────────────────────────────────────────────────────────────────

GROUP_KEYS = ("variant", "train_track", "protocol_version")


def summarize_runs(rows: Iterable[dict], metric: str = "eval_gsm8k_maj3") -> List[dict]:
    """Mean and seed SD (ddof=1) per exact (variant, track, protocol). Duplicate
    seeds inside a group are an error, not something to average away."""
    groups: Dict[Tuple, Dict[int, float]] = {}
    for r in rows:
        key = tuple(r[k] for k in GROUP_KEYS)
        seeds = groups.setdefault(key, {})
        if r["seed"] in seeds:
            raise ValueError(f"Duplicate seed {r['seed']} in group {key}")
        seeds[r["seed"]] = r[metric]
    out = []
    for key, by_seed in sorted(groups.items()):
        vals = [by_seed[s] for s in sorted(by_seed)]
        out.append({**dict(zip(GROUP_KEYS, key)), "metric": metric, "n_seeds": len(vals),
                    "seeds": sorted(by_seed), "values_by_seed": dict(sorted(by_seed.items())),
                    "mean": statistics.mean(vals),
                    "seed_sd": statistics.stdev(vals) if len(vals) > 1 else None})
    return out


def paired_seed_differences(rows: Iterable[dict], metric: str, variant_a: str, variant_b: str,
                            train_track: str, protocol_version: str) -> dict:
    """a - b per seed, using only seeds present for both variants."""
    rows = list(rows)

    def pick(v):
        return {r["seed"]: r[metric] for r in rows
                if r["variant"] == v and r["train_track"] == train_track and r["protocol_version"] == protocol_version}
    a, b = pick(variant_a), pick(variant_b)
    seeds = sorted(set(a) & set(b))
    diffs = {s: a[s] - b[s] for s in seeds}
    vals = list(diffs.values())
    return {"variant_a": variant_a, "variant_b": variant_b, "metric": metric, "seeds": seeds,
            "unpaired_seeds": sorted(set(a) ^ set(b)), "diff_by_seed": diffs,
            "mean_diff": statistics.mean(vals) if vals else None,
            "seed_sd_of_diff": statistics.stdev(vals) if len(vals) > 1 else None}


def paired_item_table(records_a: List[dict], records_b: List[dict]) -> dict:
    """2x2 correctness table matched by item_id. Item sets must be identical."""
    a = {r["item_id"]: r["correct"] for r in records_a}
    b = {r["item_id"]: r["correct"] for r in records_b}
    if set(a) != set(b) or len(a) != len(records_a) or len(b) != len(records_b):
        raise ValueError("Item sets differ or contain duplicate item_ids; cannot pair")
    t = {"both_correct": 0, "a_only": 0, "b_only": 0, "neither": 0}
    for i in a:
        t[{(1, 1): "both_correct", (1, 0): "a_only", (0, 1): "b_only", (0, 0): "neither"}[(a[i], b[i])]] += 1
    return {**t, "n_items": len(a)}
