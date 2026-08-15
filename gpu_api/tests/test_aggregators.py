#!/usr/bin/env python3
"""Smoke tests for aggregate scripts using text-free synthetic records."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "gpu_api" / "scripts"


def write_records(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def run(script: str, *arguments: str) -> None:
    subprocess.run(
        [sys.executable, str(SCRIPTS / script), *arguments],
        check=True,
        cwd=ROOT,
    )


def e5_record(position: int, safe: bool, external_unsafe: bool) -> dict:
    return {
        "record_schema": "pku-saferlhf.e5-position.v1",
        "request_id": f"e5:{position}",
        "status": "ok",
        "site": "shieldgemma_9b",
        "rendering": "prompt_response",
        "pair_id": "fixture:1",
        "source_file": "fixture.jsonl",
        "source_line": 1,
        "response_position": position,
        "response_sha256": f"hash{position}",
        "is_safe": safe,
        "severity_level": 0 if safe else 2,
        "harm_categories": [] if safe else ["Cybercrime"],
        "response_character_length": 10,
        "safer_response_id": 0,
        "native_stratum": "L4_safety_boundary_difference",
        "pair_design_weight": 5.0,
        "external_unsafe": external_unsafe,
        "policy_flags": {"dangerous_content": external_unsafe},
        "policy_yes_probabilities": {"dangerous_content": 0.9 if external_unsafe else 0.1},
    }


def e6_record(order: int, label: str) -> dict:
    response_a_id, response_b_id = ((0, 1) if order == 0 else (1, 0))
    return {
        "record_schema": "pku-saferlhf.e6-principle-order.v1",
        "request_id": f"primary:fixture:1:0:{order}",
        "status": "ok",
        "phase": "primary",
        "source_file": "fixture.jsonl",
        "source_line": 1,
        "response_0_sha256": "hash0",
        "response_1_sha256": "hash1",
        "native_stratum": "L4_safety_boundary_difference",
        "stratum_population_N_h": 10,
        "stratum_sample_n_h": 2,
        "design_weight_N_h_over_n_h": 5.0,
        "safer_response_id": 0,
        "principle_index": 0,
        "order": order,
        "response_a_id": response_a_id,
        "response_b_id": response_b_id,
        "label": label,
    }


def e7_record(pair: int, position: int, safe: bool, score: float, safer: int) -> dict:
    return {
        "record_schema": "pku-saferlhf.e7-score.v1",
        "request_id": f"fixture:{pair}:{position}",
        "status": "ok",
        "pair_id": f"fixture:{pair}",
        "source_file": "fixture.jsonl",
        "source_line": pair,
        "response_position": position,
        "response_sha256": f"hash{pair}{position}",
        "is_safe": safe,
        "severity_level": 0 if safe else 2,
        "harm_categories": [] if safe else ["Cybercrime"],
        "response_character_length": 10 + position,
        "safer_response_id": safer,
        "native_stratum": "L4_safety_boundary_difference",
        "pair_design_weight": 5.0,
        "raw_cost_score": score,
    }


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        private = root / "private"
        output = root / "aggregate"
        private.mkdir()
        output.mkdir()

        write_records(
            private / "e5.jsonl",
            [e5_record(0, True, False), e5_record(1, False, True)],
        )
        run(
            "e5_aggregate.py",
            "--private-run",
            str(private),
            "--aggregate-dir",
            str(output / "e5"),
        )
        e5 = json.loads((output / "e5" / "e5_summary.json").read_text())
        assert e5["completed_records"] == 2
        assert (
            e5["site_rendering_summaries"]["shieldgemma_9b:prompt_response"]
            ["design_weighted_population_estimate"]["n"]
            == 10.0
        )
        e5b = e5["site_rendering_summaries"]["shieldgemma_9b:prompt_response"][
            "e5b_safer_selection"
        ]
        assert e5b["complete_pairs"] == 1
        assert e5b["externally_discriminated_pairs"] == 1
        assert e5b["safer_selects_external_safe_pairs"] == 1
        e5_bootstrap = e5["site_rendering_summaries"]["shieldgemma_9b:prompt_response"][
            "bootstrap_sampling_uncertainty"
        ]
        assert e5_bootstrap["repetitions"] == 2_000
        assert e5_bootstrap["pair_counts_by_stratum"]["L4_safety_boundary_difference"] == 1
        assert e5_bootstrap["design_weighted_agreement_95_intervals"][
            "observed_agreement"
        ] == [1.0, 1.0]

        write_records(
            private / "primary_judgements.jsonl",
            [e6_record(0, "A"), e6_record(1, "B")],
        )
        run(
            "e6_aggregate.py",
            "--private-run",
            str(private),
            "--aggregate-dir",
            str(output / "e6"),
            "--phase",
            "primary",
        )
        e6 = json.loads((output / "e6" / "e6_primary_summary.json").read_text())
        assert e6["pair_principle_orientation_counts"]["stable_substantive"] == 1
        assert e6["bootstrap_sampling_uncertainty"]["replicates"] == 10_000
        assert (
            e6["bootstrap_sampling_uncertainty"]["stratum_state_shares"]
            ["L4_safety_boundary_difference|consensus"]["estimate"]
            == 1.0
        )

        # The E7 severity trace fits an intercept, severity, category indicators,
        # and length among unsafe positions. Keep this fixture small but full rank
        # so the smoke test exercises the actual aggregation path.
        e7_records: list[dict] = []
        unsafe_profiles = [
            (1, ["Cybercrime"], 20),
            (2, ["Privacy Violation"], 30),
            (3, ["Cybercrime", "Privacy Violation"], 40),
            (2, ["Cybercrime"], 50),
            (1, ["Privacy Violation"], 60),
            (3, ["Cybercrime", "Privacy Violation"], 70),
        ]
        for pair, (severity, categories, length) in enumerate(unsafe_profiles, start=1):
            safer = 0 if pair % 2 else 1
            unsafe_position = 1 - safer
            e7_records.extend(
                [
                    e7_record(pair, safer, True, -float(pair), safer),
                    e7_record(pair, unsafe_position, False, float(pair), safer),
                ]
            )
            unsafe_record = e7_records[-1]
            unsafe_record["severity_level"] = severity
            unsafe_record["harm_categories"] = categories
            unsafe_record["response_character_length"] = length
        write_records(private / "full_scores.jsonl", e7_records)
        run(
            "e7_aggregate.py",
            "--private-run",
            str(private),
            "--aggregate-dir",
            str(output / "e7"),
            "--score-direction",
            "higher_is_unsafe",
            "--bootstrap-replicates",
            "5",
        )
        e7 = json.loads((output / "e7" / "e7_summary.json").read_text())
        assert e7["e7a_binary_boundary"]["auroc_unsafe_vs_safe"] == 1.0
        assert e7["e7a_binary_boundary"]["design_weighted_population_auroc"] == 1.0
    print("aggregate smoke tests passed")


if __name__ == "__main__":
    main()
