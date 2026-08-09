#!/usr/bin/env python3
"""Compute E7a--E7c aggregate diagnostics from private text-free scores."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from common import read_jsonl, utc_now, write_json


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    positive = int(labels.sum())
    negative = len(labels) - positive
    if not positive or not negative:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    index = 0
    while index < len(scores):
        end = index + 1
        while end < len(scores) and scores[order[end]] == scores[order[index]]:
            end += 1
        ranks[order[index:end]] = (index + 1 + end) / 2
        index = end
    return float((ranks[labels.astype(bool)].sum() - positive * (positive + 1) / 2) / (positive * negative))


def distribution(values: np.ndarray) -> dict:
    return {
        "n": int(len(values)),
        "mean": float(np.mean(values)) if len(values) else None,
        "median": float(np.median(values)) if len(values) else None,
        "q25": float(np.quantile(values, 0.25)) if len(values) else None,
        "q75": float(np.quantile(values, 0.75)) if len(values) else None,
    }


def risk_scores(records: list[dict], direction: str) -> np.ndarray:
    values = np.array([float(record["raw_cost_score"]) for record in records])
    return values if direction == "higher_is_unsafe" else -values


def bootstrap_pair_intervals(
    records: list[dict],
    direction: str,
    repetitions: int,
    seed: int,
) -> dict:
    by_pair: defaultdict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_pair[record["pair_id"]].append(record)
    pairs = list(by_pair.values())
    rng = np.random.default_rng(seed)
    auc_values: list[float] = []
    for _ in range(repetitions):
        sampled = [pairs[index] for index in rng.integers(0, len(pairs), len(pairs))]
        flat = [record for pair in sampled for record in pair]
        labels = np.array([not bool(record["is_safe"]) for record in flat])
        current_auc = auc(labels, risk_scores(flat, direction))
        if current_auc is not None:
            auc_values.append(current_auc)
    return {
        "repetitions": repetitions,
        "auc_q025": float(np.quantile(auc_values, 0.025)),
        "auc_q975": float(np.quantile(auc_values, 0.975)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--private-run", type=Path, required=True)
    parser.add_argument("--aggregate-dir", type=Path, required=True)
    parser.add_argument(
        "--score-direction", choices=("higher_is_unsafe", "lower_is_unsafe"), required=True
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260813)
    arguments = parser.parse_args()
    all_records = [
        record
        for path in arguments.private_run.glob("*_scores.jsonl")
        for record in read_jsonl(path)
        if record.get("record_schema") == "pku-saferlhf.e7-score.v1"
    ]
    records = [record for record in all_records if record.get("status") == "ok"]
    failed_records = [
        record for record in all_records if record.get("status") != "ok"
    ]
    if not records:
        raise ValueError("No completed E7 score records found.")
    direction = arguments.score_direction
    risks = risk_scores(records, direction)
    labels = np.array([not bool(record["is_safe"]) for record in records])
    boundary = {
        "risk_score_definition": direction,
        "auroc_unsafe_vs_safe": auc(labels, risks),
        "unsafe": distribution(risks[labels]),
        "safe": distribution(risks[~labels]),
    }
    threshold_predictions = risks >= 0
    boundary["zero_threshold_diagnostic"] = {
        "true_unsafe_predicted_unsafe": int(np.sum(labels & threshold_predictions)),
        "true_unsafe_predicted_safe": int(np.sum(labels & ~threshold_predictions)),
        "true_safe_predicted_unsafe": int(np.sum(~labels & threshold_predictions)),
        "true_safe_predicted_safe": int(np.sum(~labels & ~threshold_predictions)),
        "boundary": "Scale-specific diagnostic only; no calibrated or author-designated threshold is assumed.",
    }
    category_names = sorted(
        {category for record in records for category in record["harm_categories"]}
    )
    unsafe_indices = np.array([not bool(record["is_safe"]) for record in records])
    unsafe_records = [record for record, keep in zip(records, unsafe_indices) if keep]
    unsafe_risks = risks[unsafe_indices]
    design = np.column_stack(
        [
            np.ones(len(unsafe_records)),
            np.array([record["severity_level"] for record in unsafe_records]),
            np.array(
                [
                    [int(category in record["harm_categories"]) for category in category_names]
                    for record in unsafe_records
                ]
            ),
            np.log1p(
                np.array(
                    [record["response_character_length"] for record in unsafe_records]
                )
            ),
        ]
    )
    coefficients, _, _, _ = np.linalg.lstsq(design, unsafe_risks, rcond=None)
    severity_model = {
        "n_unsafe_positions": len(unsafe_records),
        "risk_score_definition": direction,
        "severity_coefficient_adjusted_for_categories_and_log_character_length": float(
            coefficients[1]
        ),
        "category_order": category_names,
        "coefficient_vector": [float(value) for value in coefficients],
        "boundary": "Associational description; severity was not assumed to be a direct cost-loss target.",
    }
    category_distribution_rows = [
        {
            "harm_category": category,
            **distribution(
                np.array(
                    [
                        risk
                        for record, risk in zip(unsafe_records, unsafe_risks)
                        if category in record["harm_categories"]
                    ]
                )
            ),
        }
        for category in category_names
    ]
    by_pair: defaultdict[str, list[tuple[dict, float]]] = defaultdict(list)
    for record, risk in zip(records, risks):
        by_pair[record["pair_id"]].append((record, float(risk)))
    gap_rows: list[dict] = []
    lower_severity_total = lower_severity_ordered = 0
    for pair_id, entries in by_pair.items():
        if len(entries) != 2:
            continue
        entries.sort(key=lambda item: item[0]["response_position"])
        first, second = entries
        records_pair = [first[0], second[0]]
        risk_by_position = {record["response_position"]: risk for record, risk in entries}
        winner = int(records_pair[0]["safer_response_id"])
        gap = risk_by_position[1 - winner] - risk_by_position[winner]
        gap_rows.append(
            {
                "pair_id": pair_id,
                "native_stratum": records_pair[0]["native_stratum"],
                "signed_risk_gap_less_safe_minus_safer": gap,
                "absolute_risk_gap": abs(gap),
            }
        )
        both_unsafe = not records_pair[0]["is_safe"] and not records_pair[1]["is_safe"]
        same_categories = (
            records_pair[0]["harm_categories"] == records_pair[1]["harm_categories"]
        )
        unequal_severity = (
            records_pair[0]["severity_level"] != records_pair[1]["severity_level"]
        )
        if both_unsafe and same_categories and unequal_severity:
            lower_severity_total += 1
            lower = min(records_pair, key=lambda record: record["severity_level"])[
                "response_position"
            ]
            other = 1 - lower
            lower_severity_ordered += int(risk_by_position[lower] < risk_by_position[other])
    gap_by_stratum = []
    for stratum in sorted({row["native_stratum"] for row in gap_rows}):
        selected = np.array(
            [
                row["signed_risk_gap_less_safe_minus_safer"]
                for row in gap_rows
                if row["native_stratum"] == stratum
            ]
        )
        gap_by_stratum.append({"native_stratum": stratum, **distribution(selected)})
    summary = {
        "result_schema": "pku-saferlhf.e7-aggregate.v1",
        "created_at_utc": utc_now(),
        "successful_response_positions": len(records),
        "failed_response_positions": len(failed_records),
        "score_direction": direction,
        "e7a_binary_boundary": boundary,
        "e7b_severity_trace": {
            **severity_model,
            "both_unsafe_same_category_unequal_severity_pairs": lower_severity_total,
            "share_lower_severity_receives_lower_risk": (
                lower_severity_ordered / lower_severity_total
                if lower_severity_total
                else None
            ),
        },
        "e7c_gap_by_stratum": gap_by_stratum,
        "bootstrap": bootstrap_pair_intervals(
            records,
            direction,
            arguments.bootstrap_replicates,
            arguments.bootstrap_seed,
        ),
    }
    write_json(arguments.aggregate_dir / "e7_summary.json", summary)
    write_csv(
        arguments.aggregate_dir / "e7_unsafe_category_score_distributions.csv",
        category_distribution_rows,
    )
    write_csv(arguments.aggregate_dir / "e7_pair_gaps.csv", gap_rows)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
