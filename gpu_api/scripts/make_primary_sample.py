#!/usr/bin/env python3
"""Freeze the shared, text-free primary pair sample for E5 and E7.

The design takes 1,000 pairs independently and uniformly from each of the
four native strata.  Its paired structure serves E5's response-level boundary
tables and E7's within-pair score-gap analysis.  E6 is drawn as a nested
subsample by e6_make_sample.py.
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
CONFIG_PATH = CONFIG_DIR / "primary_pair_sample.json"
P0_PATH = EXPERIMENTS / "cpu" / "results" / "p0_snapshot.json"
NATIVE_PATH = EXPERIMENTS / "cpu" / "results" / "native_audit.json"
MANIFEST_PATH = CONFIG_DIR / "primary_pair_sample_manifest.csv"
SUMMARY_PATH = CONFIG_DIR / "primary_pair_sample_summary.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def category_set(row: dict, position: int) -> tuple[str, ...]:
    return tuple(
        category
        for category, present in row[f"response_{position}_harm_category"].items()
        if present
    )


def stratum(row: dict) -> str:
    if bool(row["is_response_0_safe"]) != bool(row["is_response_1_safe"]):
        return "L4_safety_boundary_difference"
    if int(row["response_0_severity_level"]) != int(row["response_1_severity_level"]):
        return "L3_severity_difference_same_safety_state"
    if category_set(row, 0) != category_set(row, 1):
        return "L2_category_difference_tied_safety_and_severity"
    return "L1_no_recorded_safety_field_difference"


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    for required in (P0_PATH, NATIVE_PATH):
        if not required.exists():
            raise FileNotFoundError(f"Required CPU provenance is absent: {required}")
    p0 = json.loads(P0_PATH.read_text(encoding="utf-8"))
    if p0["releases"]["dual"]["rows"] != 73_907:
        raise ValueError("P0 does not describe the expected 73,907-row dual release")

    sample = config["sample"]
    strata = sample["strata"]
    per_stratum = int(sample["pairs_per_stratum"])
    grouped: defaultdict[str, list[dict]] = defaultdict(list)
    rows = 0
    for path in sorted(RAW_DUAL.glob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                row = json.loads(line)
                rows += 1
                grouped[stratum(row)].append(
                    {
                        "source_file": path.name,
                        "source_line": line_number,
                        "response_0_sha256": row["response_0_sha256"],
                        "response_1_sha256": row["response_1_sha256"],
                    }
                )
    if rows != 73_907 or set(grouped) != set(strata):
        raise ValueError("The primary sampling strata do not partition P0")
    if any(len(grouped[name]) < per_stratum for name in strata):
        raise ValueError("A stratum is smaller than its requested sample")

    rng = random.Random(int(sample["seed"]))
    selected: list[dict] = []
    for name in strata:
        population = len(grouped[name])
        for draw_order, record in enumerate(rng.sample(grouped[name], per_stratum), start=1):
            selected.append(
                {
                    **record,
                    "stratum": name,
                    "stratum_population_N_h": population,
                    "stratum_sample_n_h": per_stratum,
                    "inclusion_probability": per_stratum / population,
                    "design_weight_N_h_over_n_h": population / per_stratum,
                    "within_stratum_draw_order": draw_order,
                }
            )
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
    with MANIFEST_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(selected)
    summary = {
        "result_schema": "pku-saferlhf.primary-pair-sample.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha256(CONFIG_PATH),
        "script_sha256": sha256(SCRIPT),
        "p0_manifest_sha256": sha256(P0_PATH),
        "native_audit_sha256": sha256(NATIVE_PATH),
        "population_rows": rows,
        "sample_rows": len(selected),
        "sample_manifest_sha256": sha256(MANIFEST_PATH),
        "strata": [
            {
                "stratum": name,
                "N_h": len(grouped[name]),
                "n_h": per_stratum,
                "inclusion_probability": per_stratum / len(grouped[name]),
                "design_weight": len(grouped[name]) / per_stratum,
            }
            for name in strata
        ],
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
