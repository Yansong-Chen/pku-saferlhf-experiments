#!/usr/bin/env python3
"""Verify the scope-critical invariants of the completed CPU evidence package."""

from __future__ import annotations

import csv
import json
from pathlib import Path


SCRIPT = Path(__file__).resolve()
EXPERIMENTS = SCRIPT.parents[1]
RESULTS = EXPERIMENTS / "cpu" / "results"


def read_json(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def main() -> None:
    native = read_json("native_audit.json")
    e1 = read_json("e1_response_process.json")
    e2 = read_json("e2_efa.json")
    e4 = read_json("e4_field_routing.json")
    dual_single = read_json("dual_single_contrast.json")
    assert native["result_schema"] == "pku-saferlhf.native-audit.v1"
    assert e1["result_schema"] == "pku-saferlhf.e1-response-process-result.v1"
    assert len(e1["sources"]) == 4
    assert native["analysis_population"]["rows"] == 73_907
    assert native["analysis_population"]["response_positions"] == 147_814
    assert native["schema"]["schema_identity"]["safe_equals_severity_zero"] == {
        "numerator": 147_806,
        "denominator": 147_814,
    }
    decomposition = {
        (row["safer_winner_is_safe"], row["alternative_is_safe"]): row["rows"]
        for row in native["absolute_state_decomposition"]
    }
    assert decomposition[("false", "false")] == 32_656
    assert decomposition[("false", "true")] == 0
    assert sum(row["rows"] for row in native["native_strata"]["rows"]) == 73_907
    with (RESULTS / "category_frequency.csv").open(encoding="utf-8") as handle:
        categories = list(csv.DictReader(handle))
    assert len(categories) == 19
    assert e2["result_schema"] == "pku-saferlhf.e2-efa.v1"
    assert e2["primary_unsafe_positions"]["n_obs"] == 76_125
    assert e2["primary_unsafe_positions"]["excluded_safe_response_positions"] == 71_689
    assert e2["one_response_per_pair_unsafe_sensitivity"]["n_obs"] > 0
    assert e2["primary_unsafe_positions"]["parallel_analysis"]["simulations"] == 1_000
    assert e2["one_response_per_pair_unsafe_sensitivity"]["parallel_analysis"]["simulations"] == 1_000
    assert len(e4["fields"]) == 5
    assert e4["protocol_schema"] == "pku-saferlhf.e4-field-routing.v1"
    assert dual_single["result_schema"] == "pku-saferlhf.dual-single-contrast.v1"
    assert dual_single["matching"]["one_to_one_matched_pairs"] == 64_640
    assert dual_single["provenance"]["matched_prompt_mismatches"] == 0
    assert dual_single["dual_preference_relation"]["agree"]["rows"] == 49_050
    assert dual_single["dual_preference_relation"]["conflict"]["rows"] == 15_590
    assert dual_single["dual_preference_relation"]["conflict"]["single_follows_dual_safer"] == {
        "rows": 7_899,
        "share_given_conflict": 7_899 / 15_590,
    }
    assert dual_single["l4_binary_boundary_contrast"] == {
        "definition": (
            "Subset of matched pairs where Dual better and safer select different responses "
            "and Dual records different is_safe states. In every such matched row, the Dual "
            "safer selection is the released-safe response."
        ),
        "rows": 1_604,
        "single_selects_dual_released_safe": 1_187,
        "share": 1_187 / 1_604,
    }
    print(
        json.dumps(
            {
                "cpu_verification": "passed",
                "native_rows": native["analysis_population"]["rows"],
                "e2_parallel_simulations_per_analysis": 1_000,
                "e4_routed_fields": len(e4["fields"]),
                "dual_single_one_to_one_pairs": dual_single["matching"][
                    "one_to_one_matched_pairs"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
