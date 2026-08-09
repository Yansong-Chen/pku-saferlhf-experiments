#!/usr/bin/env python3
"""Aggregate E6 judgements into CCAI states with orientation diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from common import read_jsonl, utc_now, write_json


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def native_vote(record: dict) -> int | None:
    if record["label"] == "No preference":
        return None
    return record["response_a_id"] if record["label"] == "A" else record["response_b_id"]


def reconcile(records: list[dict]) -> tuple[str, int | None]:
    """Return orientation status and an optional native response preference."""
    if len(records) != 2:
        return "missing", None
    votes = [native_vote(record) for record in records]
    if votes[0] is None and votes[1] is None:
        return "stable_indifference", None
    if votes[0] is not None and votes[1] is not None:
        if votes[0] == votes[1]:
            return "stable_substantive", votes[0]
        return "position_unstable", None
    return "orientation_inconclusive", None


def state(votes: list[int | None]) -> str:
    retained = [vote for vote in votes if vote is not None]
    if not retained:
        return "indifference"
    return "consensus" if len(set(retained)) == 1 else "conflict"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--private-run", type=Path, required=True)
    parser.add_argument("--aggregate-dir", type=Path, required=True)
    parser.add_argument(
        "--phase", choices=("pilot", "primary", "repeat"), default="primary"
    )
    arguments = parser.parse_args()
    all_phase_records = [
        record
        for path in arguments.private_run.glob("*_judgements.jsonl")
        for record in read_jsonl(path)
        if record.get("record_schema") == "pku-saferlhf.e6-principle-order.v1"
        and record.get("phase") == arguments.phase
    ]
    raw_records = [
        record for record in all_phase_records if record.get("status") == "ok"
    ]
    failed_records = [
        record for record in all_phase_records if record.get("status") != "ok"
    ]
    # A resumed job may contain an earlier successful record only when a
    # private log was deliberately merged. Keep the last completion per
    # request identifier so every pair-principle-order has one vote.
    latest: dict[str, dict] = {}
    for record in raw_records:
        latest[record["request_id"]] = record
    records = list(latest.values())
    grouped: defaultdict[tuple[str, int, int], list[dict]] = defaultdict(list)
    for record in records:
        grouped[
            (f"{record['source_file']}:{record['source_line']}", record["principle_index"], 0)
        ].append(record)
    pair_votes: defaultdict[str, list[tuple[str, int | None]]] = defaultdict(list)
    pair_metadata: dict[str, dict] = {}
    all_pair_ids: set[str] = set()
    orientation_counts: Counter[str] = Counter()
    agreement_rows: list[dict] = []
    for (pair_id, principle_index, _), current in grouped.items():
        orientation, vote = reconcile(current)
        orientation_counts[orientation] += 1
        metadata = current[0]
        pair_metadata[pair_id] = metadata
        all_pair_ids.add(pair_id)
        if orientation in {"stable_substantive", "stable_indifference"}:
            pair_votes[pair_id].append((orientation, vote))
        agreement_rows.append(
            {
                "pair_id": pair_id,
                "principle_index": principle_index,
                "orientation_status": orientation,
                "preferred_native_response": vote,
                "native_stratum": metadata["native_stratum"],
            }
        )
    by_stratum: Counter[tuple[str, str]] = Counter()
    weighted_by_stratum: Counter[tuple[str, str]] = Counter()
    direction_counts: Counter[tuple[str, str]] = Counter()
    unclassifiable = 0
    for pair_id, votes in pair_votes.items():
        metadata = pair_metadata[pair_id]
        ccai_state = state([vote for _, vote in votes])
        key = (metadata["native_stratum"], ccai_state)
        by_stratum[key] += 1
        weighted_by_stratum[key] += float(metadata["design_weight_N_h_over_n_h"])
        for orientation, vote in votes:
            if vote is not None:
                direction_counts[(ccai_state, "agrees_with_safer" if vote == metadata["safer_response_id"] else "disagrees_with_safer")] += 1
    unclassifiable = len(all_pair_ids - set(pair_votes))
    table_rows = []
    for key in sorted(set(by_stratum) | set(weighted_by_stratum)):
        stratum, ccai_state = key
        table_rows.append(
            {
                "native_stratum": stratum,
                "ccai_state": ccai_state,
                "unweighted_pairs": by_stratum[key],
                "design_weighted_pair_estimate": weighted_by_stratum[key],
            }
        )
    repeat_agreement = None
    if arguments.phase == "repeat":
        primary_latest: dict[tuple[str, int, int, int], dict] = {}
        for path in arguments.private_run.glob("primary_judgements.jsonl"):
            for record in read_jsonl(path):
                if (
                    record.get("record_schema") == "pku-saferlhf.e6-principle-order.v1"
                    and record.get("status") == "ok"
                ):
                    primary_latest[
                        (
                            record["source_file"],
                            int(record["source_line"]),
                            int(record["principle_index"]),
                            int(record["order"]),
                        )
                    ] = record
        matches = disagreements = 0
        for record in records:
            key = (
                record["source_file"],
                int(record["source_line"]),
                int(record["principle_index"]),
                int(record["order"]),
            )
            if key in primary_latest:
                matches += int(record["label"] == primary_latest[key]["label"])
                disagreements += int(record["label"] != primary_latest[key]["label"])
        repeat_agreement = {
            "matched_order_judgements": matches + disagreements,
            "exact_label_agreement": (
                matches / (matches + disagreements)
                if matches + disagreements
                else None
            ),
        }
    summary = {
        "result_schema": "pku-saferlhf.e6-aggregate.v1",
        "created_at_utc": utc_now(),
        "phase": arguments.phase,
        "successful_order_judgements": len(records),
        "failed_order_judgements": len(failed_records),
        "pair_principle_orientation_counts": dict(orientation_counts),
        "pairs_with_retained_principles": len(pair_votes),
        "unclassifiable_pairs": unclassifiable,
        "direction_counts_within_retained_principles": {
            "|".join(key): value for key, value in direction_counts.items()
        },
        "repeat_vs_primary_exact_label_agreement": repeat_agreement,
        "boundary": (
            "This table relates a fixed CCAI reference system to PKU native strata. "
            "It is neither a safety truth table nor a direct reproduction of Buyl et al."
        ),
    }
    write_json(arguments.aggregate_dir / f"e6_{arguments.phase}_summary.json", summary)
    write_csv(arguments.aggregate_dir / f"e6_{arguments.phase}_by_stratum.csv", table_rows)
    write_csv(arguments.aggregate_dir / f"e6_{arguments.phase}_orientation.csv", agreement_rows)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
