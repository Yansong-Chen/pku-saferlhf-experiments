#!/usr/bin/env python3
"""Aggregate E6 judgements into CCAI states with orientation diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from random import Random

from common import load_rows_for_manifest, native_stratum, read_jsonl, utc_now, write_json


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


CCAI_STATES = ("conflict", "consensus", "indifference")


def percentile(values: list[float], probability: float) -> float:
    """Linear-interpolated percentile without a numerical-library dependency."""
    if not values:
        raise ValueError("A percentile requires at least one value.")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def bootstrap_state_intervals(
    ccai_state_by_pair: dict[str, str],
    pair_metadata: dict[str, dict],
    replicates: int,
    seed: int,
) -> dict | None:
    """Bootstrap CCAI-state shares by resampling pairs within native strata.

    The oracle, response-order reconciliation, and fixed principle battery remain
    fixed.  These intervals therefore describe sampling uncertainty only.
    """
    if not ccai_state_by_pair or replicates <= 0:
        return None
    states_by_stratum: defaultdict[str, list[str]] = defaultdict(list)
    population_by_stratum: dict[str, float] = {}
    for pair_id, ccai_state in ccai_state_by_pair.items():
        metadata = pair_metadata[pair_id]
        stratum = metadata["native_stratum"]
        states_by_stratum[stratum].append(ccai_state)
        population_by_stratum[stratum] = float(metadata["stratum_population_N_h"])
    strata = sorted(states_by_stratum)
    if not strata:
        return None
    random = Random(seed)
    stratum_draws: defaultdict[tuple[str, str], list[float]] = defaultdict(list)
    weighted_draws: defaultdict[str, list[float]] = defaultdict(list)
    l4_difference_draws: defaultdict[str, list[float]] = defaultdict(list)
    total_population = sum(population_by_stratum.values())
    for _ in range(replicates):
        shares: dict[str, dict[str, float]] = {}
        for stratum in strata:
            values = states_by_stratum[stratum]
            counts = Counter(random.choice(values) for _ in range(len(values)))
            shares[stratum] = {
                ccai_state: counts[ccai_state] / len(values)
                for ccai_state in CCAI_STATES
            }
            for ccai_state in CCAI_STATES:
                stratum_draws[(stratum, ccai_state)].append(shares[stratum][ccai_state])
        for ccai_state in CCAI_STATES:
            weighted_draws[ccai_state].append(
                sum(
                    population_by_stratum[stratum] * shares[stratum][ccai_state]
                    for stratum in strata
                )
                / total_population
            )
        l4 = "L4_safety_boundary_difference"
        if l4 in shares:
            for comparator in ("L1_no_recorded_safety_field_difference",
                               "L2_category_difference_tied_safety_and_severity",
                               "L3_severity_difference_same_safety_state"):
                if comparator in shares:
                    l4_difference_draws[comparator].append(
                        shares[l4]["conflict"] - shares[comparator]["conflict"]
                    )

    def interval(values: list[float], estimate: float) -> dict:
        return {
            "estimate": estimate,
            "lower_95": percentile(values, 0.025),
            "upper_95": percentile(values, 0.975),
        }

    stratum_state = {
        f"{stratum}|{ccai_state}": interval(
            stratum_draws[(stratum, ccai_state)],
            states_by_stratum[stratum].count(ccai_state)
            / len(states_by_stratum[stratum]),
        )
        for stratum in strata
        for ccai_state in CCAI_STATES
    }
    weighted_state = {
        ccai_state: interval(
            weighted_draws[ccai_state],
            sum(
                population_by_stratum[stratum]
                * states_by_stratum[stratum].count(ccai_state)
                / len(states_by_stratum[stratum])
                for stratum in strata
            )
            / total_population,
        )
        for ccai_state in CCAI_STATES
    }
    l4_contrasts = {
        f"L4_minus_{comparator.split('_', 1)[0]}": interval(
            draws,
            states_by_stratum["L4_safety_boundary_difference"].count("conflict")
            / len(states_by_stratum["L4_safety_boundary_difference"])
            - states_by_stratum[comparator].count("conflict")
            / len(states_by_stratum[comparator]),
        )
        for comparator, draws in l4_difference_draws.items()
    }
    return {
        "method": "stratified nonparametric pair bootstrap",
        "interval_level": 0.95,
        "replicates": replicates,
        "seed": seed,
        "stratum_state_shares": stratum_state,
        "release_weighted_state_shares": weighted_state,
        "l4_conflict_rate_differences": l4_contrasts,
        "boundary": (
            "Intervals resample released pairs within native strata while holding "
            "the oracle, prompt, response-order rule, and principle battery fixed. "
            "They do not quantify variation across models, prompts, or principles."
        ),
    }


def native_l4_selection_check(sample_manifest: Path, pair_ids: set[str]) -> dict:
    """Record whether the named safer response is the released-safe response in L4.

    This derives only text-free counts from the pinned release.  L4's definition
    establishes that the two response-level safety states differ; this separate
    check establishes the direction selected by ``safer_response_id``.
    """
    l4_pairs = 0
    selects_safe = 0
    selects_unsafe = 0
    for manifest_record, row in load_rows_for_manifest(sample_manifest):
        pair_id = f"{manifest_record['source_file']}:{manifest_record['source_line']}"
        if pair_id not in pair_ids:
            continue
        if native_stratum(row) != "L4_safety_boundary_difference":
            continue
        l4_pairs += 1
        selected_is_safe = bool(
            row[f"is_response_{int(row['safer_response_id'])}_safe"]
        )
        selects_safe += int(selected_is_safe)
        selects_unsafe += int(not selected_is_safe)
    return {
        "l4_pairs_with_ccai_state": l4_pairs,
        "safer_selects_released_safe": selects_safe,
        "safer_selects_released_unsafe": selects_unsafe,
        "boundary": (
            "This is a release-record check for the analysed L4 pairs. It does not "
            "validate the released safe/unsafe state."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--private-run", type=Path, required=True)
    parser.add_argument("--aggregate-dir", type=Path, required=True)
    parser.add_argument(
        "--phase", choices=("pilot", "primary", "repeat"), default="primary"
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260811)
    parser.add_argument(
        "--sample-manifest",
        type=Path,
        help="Optional E6 manifest for text-free native L4 direction checks.",
    )
    arguments = parser.parse_args()
    all_phase_records = [
        record
        for path in arguments.private_run.glob("*_judgements.jsonl")
        for record in read_jsonl(path)
        if record.get("record_schema") == "pku-saferlhf.e6-principle-order.v1"
        and record.get("phase") == arguments.phase
    ]
    # Retain the last record for each planned judgement before separating
    # successful labels from failures.  A failure is a missing measurement
    # state, never an abstention or an implicit ``No preference`` vote.
    latest_all: dict[str, dict] = {}
    for record in all_phase_records:
        latest_all[record["request_id"]] = record
    raw_records = [
        record for record in latest_all.values() if record.get("status") == "ok"
    ]
    failed_records = [
        record for record in latest_all.values() if record.get("status") != "ok"
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
    for record in latest_all.values():
        pair_id = f"{record['source_file']}:{record['source_line']}"
        pair_metadata[pair_id] = record
        all_pair_ids.add(pair_id)
    for (pair_id, principle_index, _), current in grouped.items():
        orientation, vote = reconcile(current)
        orientation_counts[orientation] += 1
        metadata = current[0]
        pair_metadata[pair_id] = metadata
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
    direction_counts_by_stratum: Counter[tuple[str, str, str]] = Counter()
    ccai_state_by_pair: dict[str, str] = {}
    unclassifiable = 0
    for pair_id, votes in pair_votes.items():
        metadata = pair_metadata[pair_id]
        ccai_state = state([vote for _, vote in votes])
        ccai_state_by_pair[pair_id] = ccai_state
        key = (metadata["native_stratum"], ccai_state)
        by_stratum[key] += 1
        weighted_by_stratum[key] += float(metadata["design_weight_N_h_over_n_h"])
        for orientation, vote in votes:
            if vote is not None:
                direction_counts[(ccai_state, "agrees_with_safer" if vote == metadata["safer_response_id"] else "disagrees_with_safer")] += 1
                direction_counts_by_stratum[
                    (
                        metadata["native_stratum"],
                        ccai_state,
                        "agrees_with_safer"
                        if vote == metadata["safer_response_id"]
                        else "disagrees_with_safer",
                    )
                ] += 1
    unclassifiable = len(all_pair_ids - set(pair_votes))
    failure_counts = Counter(
        str(record.get("error_type", "UnknownFailure")) for record in failed_records
    )
    unparseable_finish_reasons = Counter(
        str(record.get("last_finish_reason", "not_recorded"))
        for record in failed_records
        if record.get("error_type") == "UnparseableModelOutput"
    )
    pairs_with_failed_orders = len(
        {
            f"{record['source_file']}:{record['source_line']}"
            for record in failed_records
        }
    )
    bootstrap = (
        bootstrap_state_intervals(
            ccai_state_by_pair,
            pair_metadata,
            arguments.bootstrap_replicates,
            arguments.bootstrap_seed,
        )
        if arguments.phase == "primary"
        else None
    )
    native_l4_check = (
        native_l4_selection_check(arguments.sample_manifest, set(ccai_state_by_pair))
        if arguments.sample_manifest
        else None
    )
    strata = sorted({metadata["native_stratum"] for metadata in pair_metadata.values()})
    total_pairs_by_stratum = Counter(
        pair_metadata[pair_id]["native_stratum"] for pair_id in ccai_state_by_pair
    )
    table_rows = []
    for stratum in strata:
        for ccai_state in CCAI_STATES:
            key = (stratum, ccai_state)
            uncertainty = (
                bootstrap["stratum_state_shares"].get(f"{stratum}|{ccai_state}")
                if bootstrap
                else None
            )
            n_pairs = total_pairs_by_stratum[stratum]
            table_rows.append(
                {
                    "native_stratum": stratum,
                    "ccai_state": ccai_state,
                    "unweighted_pairs": by_stratum[key],
                    "unweighted_pair_share": by_stratum[key] / n_pairs if n_pairs else None,
                    "design_weighted_pair_estimate": weighted_by_stratum[key],
                    "bootstrap_lower_95": uncertainty["lower_95"] if uncertainty else None,
                    "bootstrap_upper_95": uncertainty["upper_95"] if uncertainty else None,
                }
            )
    direction_rows = []
    for stratum, ccai_state in sorted(
        {(stratum, ccai_state) for stratum, ccai_state, _ in direction_counts_by_stratum}
    ):
        agrees = direction_counts_by_stratum[(stratum, ccai_state, "agrees_with_safer")]
        disagrees = direction_counts_by_stratum[(stratum, ccai_state, "disagrees_with_safer")]
        direction_rows.append(
            {
                "native_stratum": stratum,
                "ccai_state": ccai_state,
                "agrees_with_safer_votes": agrees,
                "disagrees_with_safer_votes": disagrees,
                "substantive_principle_votes": agrees + disagrees,
                "share_agreeing_with_safer": agrees / (agrees + disagrees),
            }
        )
    repeat_agreement = None
    repeat_state_agreement = None
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
        primary_grouped: defaultdict[tuple[str, int], list[dict]] = defaultdict(list)
        for record in primary_latest.values():
            primary_grouped[
                (f"{record['source_file']}:{record['source_line']}", record["principle_index"])
            ].append(record)
        primary_pair_votes: defaultdict[str, list[int | None]] = defaultdict(list)
        for (pair_id, _), current in primary_grouped.items():
            orientation, vote = reconcile(current)
            if orientation in {"stable_substantive", "stable_indifference"}:
                primary_pair_votes[pair_id].append(vote)
        primary_states = {
            pair_id: state(votes) for pair_id, votes in primary_pair_votes.items()
        }
        transitions: Counter[tuple[str, str]] = Counter()
        matches = comparisons = 0
        for pair_id, repeat_state in ccai_state_by_pair.items():
            primary_state = primary_states.get(pair_id, "unclassifiable")
            transitions[(primary_state, repeat_state)] += 1
            comparisons += 1
            matches += int(primary_state == repeat_state)
        repeat_state_agreement = {
            "matched_pairs": comparisons,
            "exact_ccai_state_agreement": matches / comparisons if comparisons else None,
            "state_transition_counts": {
                "|".join(key): value for key, value in sorted(transitions.items())
            },
        }
    summary = {
        "result_schema": "pku-saferlhf.e6-aggregate.v1",
        "created_at_utc": utc_now(),
        "phase": arguments.phase,
        "successful_order_judgements": len(records),
        "failed_order_judgements": len(failed_records),
        "failed_order_judgements_by_error_type": dict(failure_counts),
        "unparseable_output_finish_reasons": dict(unparseable_finish_reasons),
        "pairs_with_one_or_more_failed_order_judgements": pairs_with_failed_orders,
        "pair_principle_orientation_counts": dict(orientation_counts),
        "pairs_with_retained_principles": len(pair_votes),
        "unclassifiable_pairs": unclassifiable,
        "direction_counts_within_retained_principles": {
            "|".join(key): value for key, value in direction_counts.items()
        },
        "direction_counts_by_native_stratum": {
            "|".join(key): value
            for key, value in sorted(direction_counts_by_stratum.items())
        },
        "stratum_pair_totals": dict(sorted(total_pairs_by_stratum.items())),
        "bootstrap_sampling_uncertainty": bootstrap,
        "native_l4_selection_check": native_l4_check,
        "repeat_vs_primary_exact_label_agreement": repeat_agreement,
        "repeat_vs_primary_ccai_state_agreement": repeat_state_agreement,
        "boundary": (
            "This table relates a fixed CCAI reference system to PKU native strata. "
            "It is neither a safety truth table nor a direct reproduction of Buyl et al. "
            "Failed or unparseable requests remain a separate missing-measurement state."
        ),
    }
    write_json(arguments.aggregate_dir / f"e6_{arguments.phase}_summary.json", summary)
    write_csv(arguments.aggregate_dir / f"e6_{arguments.phase}_by_stratum.csv", table_rows)
    write_csv(
        arguments.aggregate_dir / f"e6_{arguments.phase}_direction_by_stratum.csv",
        direction_rows,
    )
    write_csv(arguments.aggregate_dir / f"e6_{arguments.phase}_orientation.csv", agreement_rows)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
