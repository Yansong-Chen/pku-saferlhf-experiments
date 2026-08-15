#!/usr/bin/env python3
"""Aggregate text-free E5 private records into reviewed result tables."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from common import append_jsonl, read_jsonl, utc_now, write_json


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def agreement_statistics(
    rows: list[dict], external_key: str, *, weighted: bool = False
) -> dict:
    cells: Counter[tuple[bool, bool]] = Counter()
    for row in rows:
        weight = float(row.get("pair_design_weight", 1.0)) if weighted else 1.0
        cells[(not bool(row["is_safe"]), bool(row[external_key]))] += weight
    total = sum(cells.values())
    if not total:
        return {"n": 0}
    observed = (cells[(True, True)] + cells[(False, False)]) / total
    p_native = (cells[(True, True)] + cells[(True, False)]) / total
    p_external = (cells[(True, True)] + cells[(False, True)]) / total
    expected = p_native * p_external + (1 - p_native) * (1 - p_external)
    kappa = None if expected == 1 else (observed - expected) / (1 - expected)
    positive_denominator = 2 * cells[(True, True)] + cells[(True, False)] + cells[(False, True)]
    negative_denominator = 2 * cells[(False, False)] + cells[(True, False)] + cells[(False, True)]
    return {
        "n": total,
        "pku_unsafe_and_external_unsafe": cells[(True, True)],
        "pku_unsafe_and_external_safe": cells[(True, False)],
        "pku_safe_and_external_unsafe": cells[(False, True)],
        "pku_safe_and_external_safe": cells[(False, False)],
        "observed_agreement": observed,
        "positive_proportional_agreement": (
            2 * cells[(True, True)] / positive_denominator if positive_denominator else None
        ),
        "negative_proportional_agreement": (
            2 * cells[(False, False)] / negative_denominator if negative_denominator else None
        ),
        "cohens_kappa": kappa,
        "external_unsafe_given_pku_unsafe": (
            cells[(True, True)] / (cells[(True, True)] + cells[(True, False)])
            if cells[(True, True)] + cells[(True, False)]
            else None
        ),
        "external_safe_given_pku_safe": (
            cells[(False, False)] / (cells[(False, False)] + cells[(False, True)])
            if cells[(False, False)] + cells[(False, True)]
            else None
        ),
    }


def design_aware_agreement(rows: list[dict], external_key: str) -> dict:
    """Return raw sample facts beside an inverse-probability population estimate."""

    sample = agreement_statistics(rows, external_key)
    if not any("pair_design_weight" in row for row in rows):
        return {"sample": sample, "design_weighted_population_estimate": None}
    return {
        "sample": sample,
        "design_weighted_population_estimate": agreement_statistics(
            rows, external_key, weighted=True
        ),
    }


def weighted_count(rows: list[dict]) -> float:
    return sum(float(row.get("pair_design_weight", 1.0)) for row in rows)


def external_pair_relation(pair: dict[int, dict]) -> str:
    """Describe the external state of the PKU-selected safer response and peer."""
    safer_response_id = int(pair[0]["safer_response_id"])
    alternative_response_id = 1 - safer_response_id
    selected_unsafe = bool(pair[safer_response_id]["external_unsafe"])
    alternative_unsafe = bool(pair[alternative_response_id]["external_unsafe"])
    if not selected_unsafe and alternative_unsafe:
        return "safer_external_safe_alternative_unsafe"
    if selected_unsafe and not alternative_unsafe:
        return "safer_external_unsafe_alternative_safe"
    return "both_external_unsafe" if selected_unsafe else "both_external_safe"


def safer_selection_statistics(rows: list[dict]) -> dict:
    """Calculate the pair-level relation between `safer` and an external state."""

    pair_groups: defaultdict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        pair_groups[(row["source_file"], int(row["source_line"]))].append(row)

    relation_groups: defaultdict[str, list[dict]] = defaultdict(list)
    incomplete_pairs = 0
    for pair_records in pair_groups.values():
        by_position = {
            int(record["response_position"]): record for record in pair_records
        }
        if set(by_position) != {0, 1}:
            incomplete_pairs += 1
            continue
        relation_groups[external_pair_relation(by_position)].append(
            by_position[int(by_position[0]["safer_response_id"])]
        )

    externally_discriminated = (
        relation_groups["safer_external_safe_alternative_unsafe"]
        + relation_groups["safer_external_unsafe_alternative_safe"]
    )
    selected_external_safe = relation_groups[
        "safer_external_safe_alternative_unsafe"
    ]
    return {
        "complete_pairs": len(pair_groups) - incomplete_pairs,
        "incomplete_pairs": incomplete_pairs,
        "externally_discriminated_pairs": len(externally_discriminated),
        "safer_selects_external_safe_pairs": len(selected_external_safe),
        "share_safer_selects_external_safe_given_external_difference": (
            len(selected_external_safe) / len(externally_discriminated)
            if externally_discriminated
            else None
        ),
        "design_weighted_externally_discriminated_pair_estimate": weighted_count(
            externally_discriminated
        ),
        "design_weighted_safer_selects_external_safe_pair_estimate": weighted_count(
            selected_external_safe
        ),
        "design_weighted_share_safer_selects_external_safe_given_external_difference": (
            weighted_count(selected_external_safe) / weighted_count(externally_discriminated)
            if externally_discriminated
            else None
        ),
    }


def strata_complete_pairs(rows: list[dict]) -> dict[str, list[list[dict]]]:
    """Return complete response pairs grouped by their sampled native stratum."""

    by_pair: defaultdict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for row in rows:
        by_pair[(row["native_stratum"], row["source_file"], int(row["source_line"]))].append(
            row
        )
    grouped: defaultdict[str, list[list[dict]]] = defaultdict(list)
    for (stratum, _, _), pair_records in by_pair.items():
        if {int(row["response_position"]) for row in pair_records} == {0, 1}:
            grouped[stratum].append(pair_records)
    return dict(grouped)


def percentile_interval(values: list[float | None]) -> list[float | None]:
    valid = [value for value in values if value is not None]
    if not valid:
        return [None, None]
    return [float(np.quantile(valid, 0.025)), float(np.quantile(valid, 0.975))]


def bootstrap_external_relation(
    rows: list[dict], external_key: str, repetitions: int, seed: int
) -> dict:
    """Bootstrap design-weighted E5 summaries by resampling pairs within L1--L4."""

    pairs_by_stratum = strata_complete_pairs(rows)
    if not pairs_by_stratum:
        return {"repetitions": 0, "seed": seed, "method": "not available"}

    rng = np.random.default_rng(seed)
    agreement_metrics = {
        "observed_agreement": [],
        "positive_proportional_agreement": [],
        "negative_proportional_agreement": [],
        "cohens_kappa": [],
        "external_unsafe_given_pku_unsafe": [],
        "external_safe_given_pku_safe": [],
    }
    safer_rates: list[float | None] = []
    for _ in range(repetitions):
        sampled_rows: list[dict] = []
        for stratum in sorted(pairs_by_stratum):
            stratum_pairs = pairs_by_stratum[stratum]
            sampled_indices = rng.integers(0, len(stratum_pairs), size=len(stratum_pairs))
            for index in sampled_indices:
                sampled_rows.extend(stratum_pairs[int(index)])
        agreement = agreement_statistics(sampled_rows, external_key, weighted=True)
        for metric in agreement_metrics:
            agreement_metrics[metric].append(agreement.get(metric))
        safer_rates.append(
            safer_selection_statistics(sampled_rows)[
                "design_weighted_share_safer_selects_external_safe_given_external_difference"
            ]
        )

    return {
        "repetitions": repetitions,
        "seed": seed,
        "method": (
            "Stratified non-parametric pair bootstrap: resample complete pairs within "
            "each native L1--L4 stratum, retain pair design weights, and recompute "
            "each design-weighted statistic."
        ),
        "pair_counts_by_stratum": {
            stratum: len(pair_rows) for stratum, pair_rows in sorted(pairs_by_stratum.items())
        },
        "design_weighted_agreement_95_intervals": {
            metric: percentile_interval(values)
            for metric, values in agreement_metrics.items()
        },
        "design_weighted_safer_selection_95_interval": percentile_interval(safer_rates),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--private-run", type=Path, required=True)
    parser.add_argument("--aggregate-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260814)
    arguments = parser.parse_args()
    records = [
        record
        for path in arguments.private_run.glob("*.jsonl")
        for record in read_jsonl(path)
        if record.get("record_schema") == "pku-saferlhf.e5-position.v1"
    ]
    completed = [record for record in records if record.get("status") == "ok"]
    failures = [record for record in records if record.get("status") != "ok"]
    grouped: defaultdict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in completed:
        grouped[(record["site"], record["rendering"])].append(record)
    summary = {
        "result_schema": "pku-saferlhf.e5-aggregate.v2",
        "created_at_utc": utc_now(),
        "private_run": str(arguments.private_run),
        "completed_records": len(completed),
        "failed_records": len(failures),
        "site_rendering_summaries": {},
    }
    three_way_rows: list[dict] = []
    safer_pair_rows: list[dict] = []
    policy_rows: list[dict] = []
    severity_rows: list[dict] = []
    category_set_rows: list[dict] = []
    for (site, rendering), rows in sorted(grouped.items()):
        key = f"{site}:{rendering}"
        summary["site_rendering_summaries"][key] = design_aware_agreement(
            rows, "external_unsafe"
        )
        counts: defaultdict[tuple, list[dict]] = defaultdict(list)
        for row in rows:
            counts[
                (
                    row["native_stratum"],
                    not bool(row["is_safe"]),
                    bool(row["external_unsafe"]),
                    bool(row["response_position"] == row["safer_response_id"]),
                )
            ].append(row)
        for (stratum, pku_unsafe, external_unsafe, winner), current in sorted(counts.items()):
            three_way_rows.append(
                {
                    "site": site,
                    "rendering": rendering,
                    "native_stratum": stratum,
                    "pku_unsafe": bool_text(pku_unsafe),
                    "external_unsafe": bool_text(external_unsafe),
                    "is_safer_winner": bool_text(winner),
                    "sample_response_positions": len(current),
                    "design_weighted_position_estimate": weighted_count(current),
                }
            )
        pair_groups: defaultdict[tuple[str, int], list[dict]] = defaultdict(list)
        for row in rows:
            pair_groups[(row["source_file"], int(row["source_line"]))].append(row)
        pair_counts: defaultdict[tuple[str, str, bool], list[dict]] = defaultdict(list)
        for _, pair_records in pair_groups.items():
            by_position = {
                int(record["response_position"]): record for record in pair_records
            }
            if set(by_position) != {0, 1}:
                continue
            selected = by_position[int(by_position[0]["safer_response_id"])]
            relation = external_pair_relation(by_position)
            pair_counts[
                (
                    str(selected["native_stratum"]),
                    relation,
                    bool(selected["is_safe"]),
                )
            ].append(selected)
        for (
            native_stratum,
            relation,
            pku_safer_winner_is_safe,
        ), current in sorted(pair_counts.items()):
            safer_pair_rows.append(
                {
                    "site": site,
                    "rendering": rendering,
                    "native_stratum": native_stratum,
                    "external_pair_relation": relation,
                    "pku_safer_winner_is_safe": bool_text(pku_safer_winner_is_safe),
                    "sample_pairs": len(current),
                    "design_weighted_pair_estimate": weighted_count(current),
                }
            )
        summary["site_rendering_summaries"][key]["e5b_safer_selection"] = {
            **safer_selection_statistics(rows),
            "boundary": (
                "This pair-level comparison relates PKU's safer selection to an "
                "external binary state. It does not establish that either selection "
                "is safety ground truth."
            ),
        }
        summary["site_rendering_summaries"][key]["bootstrap_sampling_uncertainty"] = (
            bootstrap_external_relation(
                rows,
                "external_unsafe",
                arguments.bootstrap_replicates,
                arguments.bootstrap_seed,
            )
        )
        severity_counts: defaultdict[tuple, list[dict]] = defaultdict(list)
        for row in rows:
            severity_counts[
                (
                    row["severity_level"],
                    not bool(row["is_safe"]),
                    bool(row["external_unsafe"]),
                )
            ].append(row)
        for (severity, pku_unsafe, external_unsafe), current in sorted(severity_counts.items()):
            severity_rows.append(
                {
                    "site": site,
                    "rendering": rendering,
                    "severity_level": severity,
                    "pku_unsafe": bool_text(pku_unsafe),
                    "external_unsafe": bool_text(external_unsafe),
                    "sample_response_positions": len(current),
                    "design_weighted_position_estimate": weighted_count(current),
                }
            )
        category_counts: defaultdict[tuple, list[dict]] = defaultdict(list)
        for row in rows:
            category_counts[
                (
                    " | ".join(row["harm_categories"]) if row["harm_categories"] else "<empty>",
                    not bool(row["is_safe"]),
                    bool(row["external_unsafe"]),
                )
            ].append(row)
        for (category_set, pku_unsafe, external_unsafe), current in sorted(category_counts.items()):
            category_set_rows.append(
                {
                    "site": site,
                    "rendering": rendering,
                    "category_set": category_set,
                    "pku_unsafe": bool_text(pku_unsafe),
                    "external_unsafe": bool_text(external_unsafe),
                    "sample_response_positions": len(current),
                    "design_weighted_position_estimate": weighted_count(current),
                }
            )
        if site == "shieldgemma_9b":
            policy_ids = sorted(
                {
                    policy
                    for row in rows
                    for policy in row.get("policy_flags", {})
                }
            )
            for policy in policy_ids:
                projected = [
                    {**row, "policy_external_unsafe": bool(row["policy_flags"][policy])}
                    for row in rows
                    if policy in row.get("policy_flags", {})
                ]
                stats = design_aware_agreement(projected, "policy_external_unsafe")
                policy_rows.append(
                    {
                        "site": site,
                        "rendering": rendering,
                        "policy": policy,
                        "sample_response_positions": stats["sample"]["n"],
                        "sample_observed_agreement": stats["sample"]["observed_agreement"],
                        "sample_cohens_kappa": stats["sample"]["cohens_kappa"],
                        "weighted_position_estimate": (
                            stats["design_weighted_population_estimate"]["n"]
                            if stats["design_weighted_population_estimate"]
                            else None
                        ),
                        "weighted_observed_agreement": (
                            stats["design_weighted_population_estimate"]["observed_agreement"]
                            if stats["design_weighted_population_estimate"]
                            else None
                        ),
                        "weighted_cohens_kappa": (
                            stats["design_weighted_population_estimate"]["cohens_kappa"]
                            if stats["design_weighted_population_estimate"]
                            else None
                        ),
                    }
                )
    write_json(arguments.aggregate_dir / "e5_summary.json", summary)
    write_csv(arguments.aggregate_dir / "e5_three_way_by_stratum.csv", three_way_rows)
    write_csv(
        arguments.aggregate_dir / "e5_safer_external_pair_relation.csv", safer_pair_rows
    )
    write_csv(arguments.aggregate_dir / "e5_by_severity.csv", severity_rows)
    write_csv(arguments.aggregate_dir / "e5_by_category_set.csv", category_set_rows)
    write_csv(arguments.aggregate_dir / "e5_shieldgemma_policy_agreement.csv", policy_rows)
    write_json(
        arguments.aggregate_dir / "e5_failure_summary.json",
        {
            "failed_records": len(failures),
            "by_site_rendering_error": {
                "|".join(str(part) for part in key): value
                for key, value in Counter(
                    (record.get("site"), record.get("rendering"), record.get("error_type"))
                    for record in failures
                ).items()
            },
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
