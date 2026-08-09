#!/usr/bin/env python3
"""Report E7 pilot score orientation without selecting it automatically."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import read_jsonl, utc_now, write_json


def auc(labels: list[bool], scores: list[float]) -> float | None:
    positive = sum(labels)
    negative = len(labels) - positive
    if not positive or not negative:
        return None
    indexed = sorted(enumerate(scores), key=lambda item: item[1])
    ranks = [0.0] * len(scores)
    start = 0
    while start < len(indexed):
        end = start + 1
        while end < len(indexed) and indexed[end][1] == indexed[start][1]:
            end += 1
        rank = (start + 1 + end) / 2
        for index, _ in indexed[start:end]:
            ranks[index] = rank
        start = end
    positive_rank_sum = sum(rank for rank, label in zip(ranks, labels) if label)
    return (positive_rank_sum - positive * (positive + 1) / 2) / (positive * negative)


def describe(values: list[float]) -> dict:
    ordered = sorted(values)
    if not ordered:
        return {"n": 0}
    return {
        "n": len(ordered),
        "mean": sum(ordered) / len(ordered),
        "minimum": ordered[0],
        "median": ordered[len(ordered) // 2],
        "maximum": ordered[-1],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--private-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    records = [
        record
        for path in arguments.private_run.glob("pilot_scores.jsonl")
        for record in read_jsonl(path)
        if record.get("record_schema") == "pku-saferlhf.e7-score.v1"
        and record.get("status") == "ok"
    ]
    if not records:
        raise ValueError("No completed E7 pilot scores found.")
    unsafe = [not bool(record["is_safe"]) for record in records]
    raw = [float(record["raw_cost_score"]) for record in records]
    result = {
        "result_schema": "pku-saferlhf.e7-pilot-orientation.v1",
        "created_at_utc": utc_now(),
        "scored_positions": len(records),
        "raw_score_by_released_state": {
            "released_unsafe": describe(
                [score for score, label in zip(raw, unsafe) if label]
            ),
            "released_safe": describe(
                [score for score, label in zip(raw, unsafe) if not label]
            ),
        },
        "auroc_if_higher_score_means_unsafe": auc(unsafe, raw),
        "auroc_if_lower_score_means_unsafe": auc(unsafe, [-score for score in raw]),
        "decision_required": (
            "Select a score direction using the model-card convention and the pilot "
            "distribution, record the justification, then supply it explicitly to "
            "the full E7 command. This report never chooses a direction itself."
        ),
    }
    write_json(arguments.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
