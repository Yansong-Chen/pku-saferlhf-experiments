#!/usr/bin/env python3
"""Create E6's nested, text-free CCAI sample manifest.

E6 draws 50 pairs within each native stratum from the frozen 4,000-pair E5/E7
sample.  Nesting retains a common set of observations for cross-experiment
comparison while preserving an inclusion probability of 50 / N_h for each
population pair in stratum h.  Prompts and responses never enter the manifest.
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


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if not P0_PATH.exists() or not NATIVE_PATH.exists():
        raise FileNotFoundError("Run P0 and the native CPU audit before fixing the E6 sample")
    p0 = json.loads(P0_PATH.read_text(encoding="utf-8"))
    if p0["releases"]["dual"]["rows"] != 73_907:
        raise ValueError("P0 does not describe the expected dual release")

    sample_config = config["sample"]
    strata_order = sample_config["strata"]
    base_manifest = CONFIG_DIR / sample_config["nested_base_manifest"]
    if not base_manifest.exists():
        raise FileNotFoundError(
            f"Create the shared primary sample before E6: {base_manifest}"
        )
    with base_manifest.open(newline="", encoding="utf-8") as handle:
        base_records = list(csv.DictReader(handle))
    grouped: defaultdict[str, list[dict]] = defaultdict(list)
    for record in base_records:
        grouped[record["stratum"]].append(record)
    expected_base = int(sample_config["nested_base_pairs_per_stratum"])
    selected_per_stratum = int(sample_config["pairs_per_stratum"])
    if set(grouped) != set(strata_order) or any(
        len(grouped[name]) != expected_base for name in strata_order
    ):
        raise ValueError("The shared primary manifest does not match E6's frozen nesting design")
    if selected_per_stratum > expected_base:
        raise ValueError("E6 requests more pairs than the shared primary sample contains")
    populations = {
        name: int(grouped[name][0]["stratum_population_N_h"]) for name in strata_order
    }
    rng = random.Random(config["sample"]["seed"])
    selected: list[dict] = []
    for name in strata_order:
        sample = rng.sample(grouped[name], selected_per_stratum)
        for sequence, record in enumerate(sample, start=1):
            selected.append(
                {
                    "source_file": record["source_file"],
                    "source_line": record["source_line"],
                    "response_0_sha256": record["response_0_sha256"],
                    "response_1_sha256": record["response_1_sha256"],
                    "stratum": name,
                    "stratum_population_N_h": populations[name],
                    "stratum_sample_n_h": selected_per_stratum,
                    "inclusion_probability": selected_per_stratum / populations[name],
                    "design_weight_N_h_over_n_h": populations[name] / selected_per_stratum,
                    "within_stratum_draw_order": sequence,
                }
            )
    if len(selected) != sample_config["pairs"]:
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
        "nested_base_manifest": str(base_manifest.relative_to(CONFIG_DIR.parent.parent)),
        "nested_base_manifest_sha256": sha256(base_manifest),
        "population_rows": p0["releases"]["dual"]["rows"],
        "sample_rows": len(selected),
        "sample_manifest_sha256": sample_digest,
        "strata": [
            {
                "stratum": name,
                "N_h": populations[name],
                "n_h": selected_per_stratum,
                "inclusion_probability": selected_per_stratum / populations[name],
                "design_weight": populations[name] / selected_per_stratum,
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
