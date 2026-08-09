#!/usr/bin/env python3
"""Create the fixed, text-free E6 stratified sample manifest.

The manifest identifies raw rows by source file and one-indexed JSONL line.
The API runner can re-read those rows when authorised; prompts and responses
never enter the tracked sampling manifest.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


SCRIPT = Path(__file__).resolve()
EXPERIMENTS = SCRIPT.parents[2]
WORKSPACE = EXPERIMENTS.parent
RAW_DUAL = WORKSPACE / "data" / "raw" / "dual"
CONFIG_DIR = EXPERIMENTS / "gpu_api" / "config"
CONFIG_PATH = CONFIG_DIR / "e6_ccai.json"
P0_PATH = EXPERIMENTS / "cpu" / "results" / "p0_snapshot.json"
NATIVE_PATH = EXPERIMENTS / "cpu" / "results" / "native_audit.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def category_set(row: dict, position: int) -> tuple[str, ...]:
    flags = row[f"response_{position}_harm_category"]
    return tuple(category for category, present in flags.items() if present)


def stratum(row: dict) -> str:
    safe0 = bool(row["is_response_0_safe"])
    safe1 = bool(row["is_response_1_safe"])
    severity0 = int(row["response_0_severity_level"])
    severity1 = int(row["response_1_severity_level"])
    if safe0 != safe1:
        return "L4_safety_boundary_difference"
    if severity0 != severity1:
        return "L3_severity_difference_same_safety_state"
    if category_set(row, 0) != category_set(row, 1):
        return "L2_category_difference_tied_safety_and_severity"
    return "L1_no_recorded_safety_field_difference"


def allocation(populations: dict[str, int], minimum: int, total: int, order: list[str]) -> dict[str, int]:
    if total < minimum * len(order):
        raise ValueError("Total sample is smaller than its fixed stratum minima")
    if any(populations[name] < minimum for name in order):
        raise ValueError("At least one stratum cannot satisfy the fixed minimum")
    allocation_result = {name: minimum for name in order}
    remaining = total - minimum * len(order)
    capacity = {name: populations[name] - minimum for name in order}
    capacity_total = sum(capacity.values())
    raw = {name: remaining * capacity[name] / capacity_total for name in order}
    extras = {name: int(raw[name]) for name in order}
    unassigned = remaining - sum(extras.values())
    rank = sorted(
        order,
        key=lambda name: (raw[name] - extras[name], -order.index(name)),
        reverse=True,
    )
    for name in rank[:unassigned]:
        extras[name] += 1
    for name in order:
        allocation_result[name] += extras[name]
        if allocation_result[name] > populations[name]:
            raise ValueError("Stratum allocation exceeds population")
    return allocation_result


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if not P0_PATH.exists() or not NATIVE_PATH.exists():
        raise FileNotFoundError("Run P0 and the native CPU audit before fixing the E6 sample")
    p0 = json.loads(P0_PATH.read_text(encoding="utf-8"))
    if p0["releases"]["dual"]["rows"] != 73_907:
        raise ValueError("P0 does not describe the expected dual release")

    strata_order = config["sample"]["strata"]
    grouped: defaultdict[str, list[dict]] = defaultdict(list)
    row_counter = 0
    for path in sorted(RAW_DUAL.glob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                row = json.loads(line)
                row_counter += 1
                grouped[stratum(row)].append(
                    {
                        "source_file": path.name,
                        "source_line": line_number,
                        "response_0_sha256": row["response_0_sha256"],
                        "response_1_sha256": row["response_1_sha256"],
                    }
                )
    if row_counter != 73_907 or set(grouped) != set(strata_order):
        raise ValueError("The frozen E6 stratum definition does not partition the P0 population")

    populations = {name: len(grouped[name]) for name in strata_order}
    selected_counts = allocation(
        populations,
        config["sample"]["minimum_per_stratum"],
        config["sample"]["pairs"],
        strata_order,
    )
    rng = random.Random(config["sample"]["seed"])
    selected: list[dict] = []
    for name in strata_order:
        sample = rng.sample(grouped[name], selected_counts[name])
        for sequence, record in enumerate(sample, start=1):
            selected.append(
                {
                    **record,
                    "stratum": name,
                    "stratum_population_N_h": populations[name],
                    "stratum_sample_n_h": selected_counts[name],
                    "inclusion_probability": selected_counts[name] / populations[name],
                    "design_weight_N_h_over_n_h": populations[name] / selected_counts[name],
                    "within_stratum_draw_order": sequence,
                }
            )
    if len(selected) != config["sample"]["pairs"]:
        raise RuntimeError("Incorrect final E6 sample size")

    manifest = CONFIG_DIR / "e6_sample_manifest.csv"
    fields = [
        "source_file",
        "source_line",
        "response_0_sha256",
        "response_1_sha256",
        "stratum",
        "stratum_population_N_h",
        "stratum_sample_n_h",
        "inclusion_probability",
        "design_weight_N_h_over_n_h",
        "within_stratum_draw_order",
    ]
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(selected)
    sample_digest = sha256(manifest)
    summary = {
        "result_schema": "pku-saferlhf.e6-sample-manifest.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha256(CONFIG_PATH),
        "script_sha256": sha256(SCRIPT),
        "p0_manifest_sha256": sha256(P0_PATH),
        "native_audit_sha256": sha256(NATIVE_PATH),
        "population_rows": row_counter,
        "sample_rows": len(selected),
        "sample_manifest_sha256": sample_digest,
        "strata": [
            {
                "stratum": name,
                "N_h": populations[name],
                "n_h": selected_counts[name],
                "inclusion_probability": selected_counts[name] / populations[name],
                "design_weight": populations[name] / selected_counts[name],
            }
            for name in strata_order
        ],
    }
    (CONFIG_DIR / "e6_sample_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
