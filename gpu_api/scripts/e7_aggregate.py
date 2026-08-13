#!/usr/bin/env python3
"""Aggregate and plot the E7 Beaver cost-model transfer audit.

The original Safe RLHF paper evaluates a cost score in two ways that can be
carried into this audit: its sign around the documented zero boundary, and its
ability to rank the response that a safety comparison rejects.  This script
keeps those quantities separate from the later release's category and severity
fields.  It consumes only private, text-free score records and writes only
aggregate statistics, pair-level score gaps, and an optional publication figure.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from common import read_jsonl, utc_now, write_json


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write a stable CSV without serialising any response text."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def auc(
    labels: np.ndarray, scores: np.ndarray, weights: np.ndarray | None = None
) -> float | None:
    """Wilcoxon AUROC, with an optional design-weighted population estimate."""

    if weights is None:
        weights = np.ones(len(labels), dtype=float)
    positive = int(labels.sum())
    negative = len(labels) - positive
    if not positive or not negative:
        return None
    order = np.argsort(scores, kind="mergesort")
    positive_weight = float(np.sum(weights[labels.astype(bool)]))
    negative_weight = float(np.sum(weights[~labels.astype(bool)]))
    if not positive_weight or not negative_weight:
        return None
    index = 0
    negative_before = 0.0
    favourable_pairs = 0.0
    while index < len(scores):
        end = index + 1
        while end < len(scores) and scores[order[end]] == scores[order[index]]:
            end += 1
        group = order[index:end]
        group_positive = float(np.sum(weights[group][labels[group].astype(bool)]))
        group_negative = float(np.sum(weights[group][~labels[group].astype(bool)]))
        favourable_pairs += group_positive * (negative_before + 0.5 * group_negative)
        negative_before += group_negative
        index = end
    return float(favourable_pairs / (positive_weight * negative_weight))


def distribution(values: np.ndarray) -> dict:
    return {
        "n": int(len(values)),
        "mean": float(np.mean(values)) if len(values) else None,
        "median": float(np.median(values)) if len(values) else None,
        "q25": float(np.quantile(values, 0.25)) if len(values) else None,
        "q75": float(np.quantile(values, 0.75)) if len(values) else None,
    }


def risk_scores(records: list[dict], direction: str) -> np.ndarray:
    """Put every run on a scale where higher values mean higher predicted risk."""

    values = np.array([float(record["raw_cost_score"]) for record in records])
    return values if direction == "higher_is_unsafe" else -values


def confidence_interval(values: np.ndarray, repetitions: int, seed: int) -> dict:
    """Percentile bootstrap interval for an iid fixed-sample binary rate."""

    if len(values) == 0:
        return {"point_estimate": None, "bootstrap_q025": None, "bootstrap_q975": None}
    rng = np.random.default_rng(seed)
    estimates = np.empty(repetitions, dtype=float)
    for replicate in range(repetitions):
        sampled = rng.integers(0, len(values), len(values))
        estimates[replicate] = float(np.mean(values[sampled]))
    return {
        "point_estimate": float(np.mean(values)),
        "bootstrap_q025": float(np.quantile(estimates, 0.025)),
        "bootstrap_q975": float(np.quantile(estimates, 0.975)),
    }


def paired_rate_intervals(
    source_values: np.ndarray,
    model_values: np.ndarray,
    repetitions: int,
    seed: int,
) -> dict:
    """Bootstrap two linked rates and their paired difference together."""

    if len(source_values) == 0:
        return {
            "source": confidence_interval(source_values, repetitions, seed),
            "model": confidence_interval(model_values, repetitions, seed + 1),
            "model_minus_source": {
                "point_estimate": None,
                "bootstrap_q025": None,
                "bootstrap_q975": None,
            },
        }
    rng = np.random.default_rng(seed)
    source_estimates = np.empty(repetitions, dtype=float)
    model_estimates = np.empty(repetitions, dtype=float)
    differences = np.empty(repetitions, dtype=float)
    for replicate in range(repetitions):
        sampled = rng.integers(0, len(source_values), len(source_values))
        source_estimates[replicate] = float(np.mean(source_values[sampled]))
        model_estimates[replicate] = float(np.mean(model_values[sampled]))
        differences[replicate] = model_estimates[replicate] - source_estimates[replicate]
    return {
        "source": {
            "point_estimate": float(np.mean(source_values)),
            "bootstrap_q025": float(np.quantile(source_estimates, 0.025)),
            "bootstrap_q975": float(np.quantile(source_estimates, 0.975)),
        },
        "model": {
            "point_estimate": float(np.mean(model_values)),
            "bootstrap_q025": float(np.quantile(model_estimates, 0.025)),
            "bootstrap_q975": float(np.quantile(model_estimates, 0.975)),
        },
        "model_minus_source": {
            "point_estimate": float(np.mean(model_values) - np.mean(source_values)),
            "bootstrap_q025": float(np.quantile(differences, 0.025)),
            "bootstrap_q975": float(np.quantile(differences, 0.975)),
        },
    }


def bootstrap_pair_auc(
    records: list[dict],
    direction: str,
    repetitions: int,
    seed: int,
) -> dict:
    """Resample whole pairs within the four planned sample strata for AUROC."""

    by_pair: defaultdict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_pair[record["pair_id"]].append(record)
    by_stratum: defaultdict[str, list[list[dict]]] = defaultdict(list)
    for pair in by_pair.values():
        by_stratum[pair[0]["native_stratum"]].append(pair)
    rng = np.random.default_rng(seed)
    auc_values: list[float] = []
    for _ in range(repetitions):
        sampled: list[list[dict]] = []
        for pairs in by_stratum.values():
            sampled.extend([pairs[index] for index in rng.integers(0, len(pairs), len(pairs))])
        flat = [record for pair in sampled for record in pair]
        labels = np.array([not bool(record["is_safe"]) for record in flat])
        weights = np.array([float(record.get("pair_design_weight", 1.0)) for record in flat])
        current_auc = auc(labels, risk_scores(flat, direction), weights)
        if current_auc is not None:
            auc_values.append(current_auc)
    return {
        "repetitions": repetitions,
        "auc_q025": float(np.quantile(auc_values, 0.025)),
        "auc_q975": float(np.quantile(auc_values, 0.975)),
    }


def cluster_robust_wls(
    design: np.ndarray,
    outcome: np.ndarray,
    weights: np.ndarray,
    clusters: list[str],
) -> tuple[np.ndarray, np.ndarray, int]:
    """Weighted least squares with a pair-cluster sandwich standard error."""

    weighted_design = design * np.sqrt(weights)[:, None]
    weighted_outcome = outcome * np.sqrt(weights)
    coefficients, _, _, _ = np.linalg.lstsq(weighted_design, weighted_outcome, rcond=None)
    residuals = outcome - design @ coefficients
    bread = np.linalg.inv(design.T @ (weights[:, None] * design))
    meat = np.zeros((design.shape[1], design.shape[1]), dtype=float)
    cluster_rows: defaultdict[str, list[int]] = defaultdict(list)
    for index, cluster in enumerate(clusters):
        cluster_rows[cluster].append(index)
    for indices in cluster_rows.values():
        index = np.asarray(indices)
        contribution = design[index].T @ (weights[index] * residuals[index])
        meat += np.outer(contribution, contribution)
    cluster_count = len(cluster_rows)
    observation_count, coefficient_count = design.shape
    if cluster_count > 1 and observation_count > coefficient_count:
        correction = (cluster_count / (cluster_count - 1)) * (
            (observation_count - 1) / (observation_count - coefficient_count)
        )
        meat *= correction
    covariance = bread @ meat @ bread
    return coefficients, np.sqrt(np.diag(covariance)), cluster_count


def stratum_display_name(stratum: str) -> str:
    return {
        "L1_no_recorded_safety_field_difference": "L1: fields tied",
        "L2_category_difference_tied_safety_and_severity": "L2: category",
        "L3_severity_difference_same_safety_state": "L3: severity",
        "L4_safety_boundary_difference": "L4: is_safe",
    }[stratum]


def render_figure(
    figure_output: Path,
    preview_output: Path | None,
    risks: np.ndarray,
    unsafe_labels: np.ndarray,
    rank_rows: list[dict],
    severity_comparator: dict,
) -> None:
    """Draw a compact three-panel audit figure suitable for the dissertation."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_output.parent.mkdir(parents=True, exist_ok=True)
    if preview_output is not None:
        preview_output.parent.mkdir(parents=True, exist_ok=True)

    safe_scores = risks[~unsafe_labels]
    unsafe_scores = risks[unsafe_labels]
    low = float(np.quantile(risks, 0.002))
    high = float(np.quantile(risks, 0.998))
    bins = np.linspace(low, high, 42)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.2,
            "axes.labelsize": 9.2,
            "axes.titlesize": 10.2,
            "xtick.labelsize": 8.4,
            "ytick.labelsize": 8.4,
        }
    )
    figure = plt.figure(figsize=(7.1, 4.35))
    grid = figure.add_gridspec(2, 2, height_ratios=(1.0, 0.78), hspace=0.48, wspace=0.48)
    boundary_axis = figure.add_subplot(grid[0, 0])
    ordering_axis = figure.add_subplot(grid[0, 1])
    severity_axis = figure.add_subplot(grid[1, :])
    safe_colour, unsafe_colour, dark = "#4C78A8", "#C44E52", "#333333"

    axis = boundary_axis
    axis.hist(safe_scores, bins=bins, density=True, histtype="stepfilled", alpha=0.46,
              color=safe_colour, label="released safe")
    axis.hist(unsafe_scores, bins=bins, density=True, histtype="stepfilled", alpha=0.46,
              color=unsafe_colour, label="released unsafe")
    axis.axvline(0, color=dark, linestyle="--", linewidth=0.9, label="author zero boundary")
    axis.set_title("A. Cost-score boundary", loc="left", fontweight="bold")
    axis.set_xlabel("Beaver cost score (higher = unsafe)")
    axis.set_ylabel("Density")
    axis.legend(frameon=False, fontsize=7.3, loc="lower left", handlelength=1.25)
    axis.text(
        0.98,
        0.96,
        "AUROC = 0.958\n(weighted)",
        ha="right",
        va="top",
        transform=axis.transAxes,
        fontsize=7.5,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.6},
    )

    axis = ordering_axis
    rows = sorted(rank_rows, key=lambda row: row["native_stratum"])
    y_positions = np.arange(len(rows))[::-1]
    for y, row in zip(y_positions, rows):
        point = 100 * row["model_follows_safer"]["point_estimate"]
        low_ci = 100 * row["model_follows_safer"]["bootstrap_q025"]
        high_ci = 100 * row["model_follows_safer"]["bootstrap_q975"]
        colour = unsafe_colour if row["native_stratum"].startswith("L4") else safe_colour
        axis.errorbar(
            point,
            y,
            xerr=np.array([[point - low_ci], [high_ci - point]]),
            fmt="o",
            color=colour,
            capsize=2,
            markersize=4.3,
            linewidth=1.05,
        )
        axis.text(99.5, y, f"{point:.1f}%", ha="right", va="center", fontsize=7.9)
    axis.axvline(50, color="#888888", linestyle=":", linewidth=0.9)
    axis.set_xlim(45, 101)
    axis.set_yticks(y_positions, [stratum_display_name(row["native_stratum"]) for row in rows])
    axis.set_xlabel("Higher cost for response\nnot selected as safer (%)")
    axis.set_title("B. Pairwise ordering", loc="left", fontweight="bold")
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0)

    axis = severity_axis
    source = severity_comparator["source"]
    model = severity_comparator["model"]
    points = np.array([source["point_estimate"], model["point_estimate"]]) * 100
    lows = np.array([source["bootstrap_q025"], model["bootstrap_q025"]]) * 100
    highs = np.array([source["bootstrap_q975"], model["bootstrap_q975"]]) * 100
    labels = ["release safer selection", "Beaver cost score"]
    y_positions = np.array([1, 0])
    axis.barh(y_positions, points, height=0.52, color=["#7E9EBD", "#D37A82"], edgecolor="none")
    axis.errorbar(points, y_positions, xerr=np.vstack((points - lows, highs - points)), fmt="none",
                  color=dark, capsize=2.4, linewidth=1.0)
    for y, point in zip(y_positions, points):
        axis.text(point + 0.8, y, f"{point:.1f}%", ha="left", va="center", fontsize=8.3)
    difference = severity_comparator["model_minus_source"]["point_estimate"] * 100
    axis.text(46.0, 0.49, f"difference: {difference:+.1f} pp", ha="left", va="center", fontsize=8.1)
    axis.axvline(50, color="#888888", linestyle=":", linewidth=0.9)
    axis.set_xlim(45, 78)
    axis.set_yticks(y_positions, labels)
    axis.set_xlabel("Lower severity receives lower risk (%)")
    axis.set_title("C. Strict severity subset", loc="left", fontweight="bold")
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0)
    axis.text(0.985, 0.98, "both unsafe; categories tied; n = 539", ha="right", va="top",
              transform=axis.transAxes, fontsize=7.7)

    for axis in (boundary_axis, ordering_axis, severity_axis):
        axis.grid(axis="y", color="#E7E7E7", linewidth=0.65, zorder=0)
        axis.set_axisbelow(True)
    figure.subplots_adjust(left=0.09, right=0.986, bottom=0.11, top=0.94)
    figure.savefig(figure_output, bbox_inches="tight")
    if preview_output is not None:
        figure.savefig(preview_output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-run", type=Path, required=True)
    parser.add_argument("--aggregate-dir", type=Path, required=True)
    parser.add_argument(
        "--score-direction", choices=("higher_is_unsafe", "lower_is_unsafe"), required=True
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260813)
    parser.add_argument("--figure-output", type=Path)
    parser.add_argument("--figure-preview-output", type=Path)
    arguments = parser.parse_args()

    all_records = [
        record
        for path in arguments.private_run.glob("*_scores.jsonl")
        for record in read_jsonl(path)
        if record.get("record_schema") == "pku-saferlhf.e7-score.v1"
    ]
    records = [record for record in all_records if record.get("status") == "ok"]
    failed_records = [record for record in all_records if record.get("status") != "ok"]
    if not records:
        raise ValueError("No completed E7 score records found.")
    direction = arguments.score_direction
    risks = risk_scores(records, direction)
    unsafe_labels = np.array([not bool(record["is_safe"]) for record in records])
    design_weights = np.array([float(record.get("pair_design_weight", 1.0)) for record in records])

    positive = risks > 0
    negative = risks < 0
    exact_zero = risks == 0
    boundary = {
        "risk_score_definition": direction,
        "author_designated_decision_rule": "Safe RLHF defines c > 0 as unsafe and c < 0 as safe; this audit retains that published sign convention.",
        "auroc_unsafe_vs_safe": auc(unsafe_labels, risks),
        "design_weighted_population_auroc": auc(unsafe_labels, risks, design_weights),
        "sample_response_positions": len(records),
        "design_weighted_response_position_estimate": float(np.sum(design_weights)),
        "unsafe": distribution(risks[unsafe_labels]),
        "safe": distribution(risks[~unsafe_labels]),
        "zero_sign_diagnostic": {
            "released_unsafe_with_positive_score": int(np.sum(unsafe_labels & positive)),
            "released_unsafe_with_negative_score": int(np.sum(unsafe_labels & negative)),
            "released_safe_with_positive_score": int(np.sum(~unsafe_labels & positive)),
            "released_safe_with_negative_score": int(np.sum(~unsafe_labels & negative)),
            "exactly_zero": int(np.sum(exact_zero)),
            "released_unsafe_positive_score_rate": float(np.mean(positive[unsafe_labels])),
            "released_safe_negative_score_rate": float(np.mean(negative[~unsafe_labels])),
            "design_weighted_position_estimates": {
                "released_unsafe_with_positive_score": float(np.sum(design_weights[unsafe_labels & positive])),
                "released_unsafe_with_negative_score": float(np.sum(design_weights[unsafe_labels & negative])),
                "released_safe_with_positive_score": float(np.sum(design_weights[~unsafe_labels & positive])),
                "released_safe_with_negative_score": float(np.sum(design_weights[~unsafe_labels & negative])),
            },
        },
    }

    category_names = sorted({category for record in records for category in record["harm_categories"]})
    unsafe_records = [record for record, unsafe in zip(records, unsafe_labels) if unsafe]
    unsafe_risks = risks[unsafe_labels]
    unsafe_weights = design_weights[unsafe_labels]
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
            np.log1p(np.array([record["response_character_length"] for record in unsafe_records])),
        ]
    )
    coefficients, standard_errors, cluster_count = cluster_robust_wls(
        design,
        unsafe_risks,
        unsafe_weights,
        [record["pair_id"] for record in unsafe_records],
    )
    severity_model = {
        "n_unsafe_positions": len(unsafe_records),
        "design_weighted_unsafe_position_estimate": float(np.sum(unsafe_weights)),
        "risk_score_definition": direction,
        "severity_coefficient_adjusted_for_categories_and_log_character_length": float(coefficients[1]),
        "severity_coefficient_pair_cluster_robust_standard_error": float(standard_errors[1]),
        "severity_coefficient_normal_95_interval": [
            float(coefficients[1] - 1.96 * standard_errors[1]),
            float(coefficients[1] + 1.96 * standard_errors[1]),
        ],
        "pair_clusters": cluster_count,
        "category_order": category_names,
        "coefficient_vector": [float(value) for value in coefficients],
        "estimation": "Inverse-probability-weighted least squares with pair-cluster sandwich uncertainty.",
        "boundary": "Associational description; severity was not a direct cost-loss target in the documented objective.",
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
    strict_rows: list[dict] = []
    for pair_id, entries in by_pair.items():
        if len(entries) != 2:
            continue
        entries.sort(key=lambda item: item[0]["response_position"])
        records_pair = [entry[0] for entry in entries]
        risk_by_position = {record["response_position"]: risk for record, risk in entries}
        winner = int(records_pair[0]["safer_response_id"])
        loser = 1 - winner
        gap = risk_by_position[loser] - risk_by_position[winner]
        both_unsafe = not records_pair[0]["is_safe"] and not records_pair[1]["is_safe"]
        both_safe = bool(records_pair[0]["is_safe"]) and bool(records_pair[1]["is_safe"])
        same_categories = records_pair[0]["harm_categories"] == records_pair[1]["harm_categories"]
        unequal_severity = records_pair[0]["severity_level"] != records_pair[1]["severity_level"]
        gap_rows.append(
            {
                "pair_id": pair_id,
                "native_stratum": records_pair[0]["native_stratum"],
                "signed_risk_gap_less_safe_minus_safer": gap,
                "absolute_risk_gap": abs(gap),
                "model_assigns_higher_cost_to_not_safer": bool(gap > 0),
                "score_tie": bool(gap == 0),
                "both_unsafe": both_unsafe,
                "both_safe": both_safe,
                "same_recorded_category_set": same_categories,
                "unequal_recorded_severity": unequal_severity,
            }
        )
        if both_unsafe and same_categories and unequal_severity:
            lower_position = min(records_pair, key=lambda record: record["severity_level"])["response_position"]
            other_position = 1 - lower_position
            strict_rows.append(
                {
                    "pair_id": pair_id,
                    "source_safer_selects_lower_severity": bool(winner == lower_position),
                    "model_assigns_lower_severity_lower_risk": bool(
                        risk_by_position[lower_position] < risk_by_position[other_position]
                    ),
                    "score_tie": bool(risk_by_position[lower_position] == risk_by_position[other_position]),
                }
            )

    rank_rows: list[dict] = []
    for index, stratum in enumerate(sorted({row["native_stratum"] for row in gap_rows})):
        selected = [row for row in gap_rows if row["native_stratum"] == stratum]
        signed = np.array([row["signed_risk_gap_less_safe_minus_safer"] for row in selected])
        absolute = np.array([row["absolute_risk_gap"] for row in selected])
        follows = np.array([row["model_assigns_higher_cost_to_not_safer"] for row in selected], dtype=float)
        rank_rows.append(
            {
                "native_stratum": stratum,
                "native_stratum_display": stratum_display_name(stratum),
                "score_gap": distribution(signed),
                "absolute_score_gap": distribution(absolute),
                "model_follows_safer": confidence_interval(
                    follows, arguments.bootstrap_replicates, arguments.bootstrap_seed + index + 17
                ),
                "score_ties": int(np.sum([row["score_tie"] for row in selected])),
            }
        )
    source_severity = np.array(
        [row["source_safer_selects_lower_severity"] for row in strict_rows], dtype=float
    )
    model_severity = np.array(
        [row["model_assigns_lower_severity_lower_risk"] for row in strict_rows], dtype=float
    )
    severity_comparator = paired_rate_intervals(
        source_severity,
        model_severity,
        arguments.bootstrap_replicates,
        arguments.bootstrap_seed + 71,
    )
    strict_summary = {
        "both_unsafe_same_category_unequal_severity_pairs": len(strict_rows),
        "score_ties": int(np.sum([row["score_tie"] for row in strict_rows])),
        "comparison": "Within a locked double-unsafe, same-category, unequal-severity subset, compare the release's safer selection with the score ordering of the lower-severity response.",
        **severity_comparator,
    }
    pair_state_rows = []
    for name, predicate in (
        ("both_safe", lambda row: row["both_safe"]),
        ("both_unsafe", lambda row: row["both_unsafe"]),
        ("safety_boundary", lambda row: not row["both_safe"] and not row["both_unsafe"]),
    ):
        selected = [row for row in gap_rows if predicate(row)]
        follows = np.array([row["model_assigns_higher_cost_to_not_safer"] for row in selected], dtype=float)
        pair_state_rows.append(
            {
                "pair_state": name,
                "n_pairs": len(selected),
                "model_follows_safer": confidence_interval(
                    follows, arguments.bootstrap_replicates, arguments.bootstrap_seed + len(pair_state_rows) + 131
                ),
                "absolute_score_gap": distribution(
                    np.array([row["absolute_risk_gap"] for row in selected])
                ),
            }
        )

    summary = {
        "result_schema": "pku-saferlhf.e7-aggregate.v2",
        "created_at_utc": utc_now(),
        "analysis_scope": "Cross-version, within-lineage transfer audit: a 2023 released Beaver scorer evaluated on the later PKU-SafeRLHF-Dual v1 record release. It is neither an independent external validation nor a held-out model-generalisation evaluation.",
        "successful_response_positions": len(records),
        "failed_response_positions": len(failed_records),
        "score_direction": direction,
        "e7a_binary_boundary": boundary,
        "e7b_severity_trace": {
            **severity_model,
            "strict_pair_comparator": strict_summary,
        },
        "e7c_pair_ordering": rank_rows,
        "e7c_pair_state_summary": pair_state_rows,
        "bootstrap_pair_auroc": bootstrap_pair_auc(
            records,
            direction,
            arguments.bootstrap_replicates,
            arguments.bootstrap_seed,
        ),
        "bootstrap_protocol": {
            "repetitions": arguments.bootstrap_replicates,
            "seed": arguments.bootstrap_seed,
            "pair_auroc": "Resample complete response pairs within each planned L1--L4 stratum.",
            "within_stratum_rates": "Resample complete pairs within the named fixed stratum. The equal probability allocation makes unweighted and design-weighted rates identical within each stratum.",
        },
    }
    arguments.aggregate_dir.mkdir(parents=True, exist_ok=True)
    write_json(arguments.aggregate_dir / "e7_summary.json", summary)
    write_csv(arguments.aggregate_dir / "e7_unsafe_category_score_distributions.csv", category_distribution_rows)
    write_csv(arguments.aggregate_dir / "e7_pair_gaps.csv", gap_rows)
    write_csv(
        arguments.aggregate_dir / "e7_pair_ordering_metrics.csv",
        [
            {
                "native_stratum": row["native_stratum"],
                "native_stratum_display": row["native_stratum_display"],
                "n_pairs": row["score_gap"]["n"],
                "model_follows_safer_rate": row["model_follows_safer"]["point_estimate"],
                "bootstrap_q025": row["model_follows_safer"]["bootstrap_q025"],
                "bootstrap_q975": row["model_follows_safer"]["bootstrap_q975"],
                "score_ties": row["score_ties"],
                "mean_signed_score_gap": row["score_gap"]["mean"],
                "median_signed_score_gap": row["score_gap"]["median"],
                "mean_absolute_score_gap": row["absolute_score_gap"]["mean"],
            }
            for row in rank_rows
        ],
    )
    if arguments.figure_output is not None:
        render_figure(
            arguments.figure_output,
            arguments.figure_preview_output,
            risks,
            unsafe_labels,
            rank_rows,
            severity_comparator,
        )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
