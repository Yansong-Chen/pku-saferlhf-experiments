#!/usr/bin/env python3
"""Create text-free, category-level diagnostics for the completed E5 ShieldGemma run.

The outputs describe association between PKU categories and ShieldGemma policy
flags.  PKU categories are multi-label, so category rows overlap and cannot be
summed into a partition of either agreement or disagreement.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


POLICIES = (
    "dangerous_content",
    "harassment",
    "hate_speech",
    "sexually_explicit_information",
)
RENDERINGS = ("prompt_response", "response_only")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"No rows to write: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rate(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def category_policy_rows(rows: list[dict[str, Any]], rendering: str) -> list[dict[str, Any]]:
    counts: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for row in rows:
        if bool(row["is_safe"]):
            continue
        weight = float(row["pair_design_weight"])
        for category in row["harm_categories"]:
            current = counts[category]
            current["sample_positions"] += 1
            current["weighted_positions"] += weight
            current["weighted_any_policy_unsafe"] += weight * bool(row["external_unsafe"])
            for policy in POLICIES:
                current[f"weighted_{policy}"] += weight * bool(row["policy_flags"][policy])
    output = []
    for category, current in sorted(counts.items(), key=lambda item: -item[1]["weighted_positions"]):
        denominator = current["weighted_positions"]
        output.append(
            {
                "site": "shieldgemma_9b",
                "rendering": rendering,
                "category": category,
                "sample_unsafe_positions_with_category": int(current["sample_positions"]),
                "design_weighted_unsafe_position_estimate": denominator,
                "weighted_any_policy_unsafe_rate": rate(current["weighted_any_policy_unsafe"], denominator),
                **{
                    f"weighted_{policy}_flag_rate": rate(current[f"weighted_{policy}"], denominator)
                    for policy in POLICIES
                },
            }
        )
    return output


def severity_rows(rows: list[dict[str, Any]], rendering: str) -> list[dict[str, Any]]:
    counts: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for row in rows:
        if bool(row["is_safe"]):
            continue
        current = counts[int(row["severity_level"])]
        weight = float(row["pair_design_weight"])
        current["sample_positions"] += 1
        current["weighted_positions"] += weight
        current["weighted_any_policy_unsafe"] += weight * bool(row["external_unsafe"])
    output = []
    for severity, current in sorted(counts.items()):
        output.append(
            {
                "site": "shieldgemma_9b",
                "rendering": rendering,
                "severity_level": severity,
                "sample_unsafe_positions": int(current["sample_positions"]),
                "design_weighted_unsafe_position_estimate": current["weighted_positions"],
                "weighted_any_policy_unsafe_rate": rate(
                    current["weighted_any_policy_unsafe"], current["weighted_positions"]
                ),
            }
        )
    return output


def policy_state_rows(rows: list[dict[str, Any]], rendering: str) -> list[dict[str, Any]]:
    counts: dict[tuple[str, bool], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for row in rows:
        weight = float(row["pair_design_weight"])
        pku_unsafe = not bool(row["is_safe"])
        for policy in POLICIES:
            current = counts[(policy, pku_unsafe)]
            current["sample_positions"] += 1
            current["weighted_positions"] += weight
            current["weighted_flagged"] += weight * bool(row["policy_flags"][policy])
    output = []
    for (policy, pku_unsafe), current in sorted(counts.items()):
        output.append(
            {
                "site": "shieldgemma_9b",
                "rendering": rendering,
                "policy": policy,
                "pku_unsafe": bool_text(pku_unsafe),
                "sample_response_positions": int(current["sample_positions"]),
                "design_weighted_position_estimate": current["weighted_positions"],
                "weighted_policy_flag_rate": rate(current["weighted_flagged"], current["weighted_positions"]),
            }
        )
    return output


def transition_rows(by_rendering: dict[str, dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    left = by_rendering["prompt_response"]
    right = by_rendering["response_only"]
    if set(left) != set(right):
        raise RuntimeError("The two renderings do not contain the same response positions.")
    counts: dict[tuple[bool, bool, bool], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for key in left:
        first, second = left[key], right[key]
        if first["is_safe"] != second["is_safe"] or first["pair_design_weight"] != second["pair_design_weight"]:
            raise RuntimeError(f"Native record mismatch across renderings at {key}.")
        current = counts[(not bool(first["is_safe"]), bool(first["external_unsafe"]), bool(second["external_unsafe"]))]
        weight = float(first["pair_design_weight"])
        current["sample_positions"] += 1
        current["weighted_positions"] += weight
    output = []
    for (pku_unsafe, primary_unsafe, sensitivity_unsafe), current in sorted(counts.items()):
        output.append(
            {
                "site": "shieldgemma_9b",
                "pku_unsafe": bool_text(pku_unsafe),
                "prompt_response_external_unsafe": bool_text(primary_unsafe),
                "response_only_external_unsafe": bool_text(sensitivity_unsafe),
                "sample_response_positions": int(current["sample_positions"]),
                "design_weighted_position_estimate": current["weighted_positions"],
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--private-run", type=Path, required=True)
    parser.add_argument("--aggregate-dir", type=Path, required=True)
    arguments = parser.parse_args()

    arguments.aggregate_dir.mkdir(parents=True, exist_ok=True)
    by_rendering: dict[str, list[dict[str, Any]]] = {}
    keyed_rows: dict[str, dict[str, dict[str, Any]]] = {}
    for rendering in RENDERINGS:
        path = arguments.private_run / f"shieldgemma_9b_{rendering}.jsonl"
        rows = [row for row in read_jsonl(path) if row.get("status") == "ok"]
        if not rows:
            raise RuntimeError(f"No successful rows in {path}")
        if any(set(row.get("policy_flags", {})) != set(POLICIES) for row in rows):
            raise RuntimeError(f"Missing or unexpected ShieldGemma policy fields in {path}")
        by_rendering[rendering] = rows
        keyed_rows[rendering] = {
            f"{row['source_file']}:{row['source_line']}:{row['response_position']}": row
            for row in rows
        }

    category_output = []
    severity_output = []
    policy_output = []
    for rendering, rows in by_rendering.items():
        category_output.extend(category_policy_rows(rows, rendering))
        severity_output.extend(severity_rows(rows, rendering))
        policy_output.extend(policy_state_rows(rows, rendering))
    transitions = transition_rows(keyed_rows)

    write_csv(arguments.aggregate_dir / "e5_shieldgemma_category_policy_relation.csv", category_output)
    write_csv(arguments.aggregate_dir / "e5_shieldgemma_severity_relation.csv", severity_output)
    write_csv(arguments.aggregate_dir / "e5_shieldgemma_policy_by_pku_state.csv", policy_output)
    write_csv(arguments.aggregate_dir / "e5_shieldgemma_rendering_transitions.csv", transitions)
    summary = {
        "result_schema": "pku-saferlhf.e5-shieldgemma-detail.v1",
        "private_run": str(arguments.private_run),
        "renderings": list(RENDERINGS),
        "policy_ids": list(POLICIES),
        "records_per_rendering": {name: len(rows) for name, rows in by_rendering.items()},
        "category_analysis_boundary": (
            "Each category row conditions on membership in a multi-label PKU category. "
            "Category rows overlap, so their weighted counts and disagreement rates are "
            "descriptive associations and do not partition the ShieldGemma disagreement."
        ),
        "policy_scope_boundary": (
            "The four ShieldGemma guideline fields are retained separately. Their relation "
            "to PKU categories does not validate or invalidate either taxonomy."
        ),
    }
    (arguments.aggregate_dir / "e5_shieldgemma_detail_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
