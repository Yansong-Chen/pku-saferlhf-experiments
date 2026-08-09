#!/usr/bin/env python3
"""Validate frozen GPU/API protocols without loading a model or calling an API."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path


SCRIPT = Path(__file__).resolve()
EXPERIMENTS = SCRIPT.parents[2]
WORKSPACE = EXPERIMENTS.parent
CONFIG_DIR = EXPERIMENTS / "gpu_api" / "config"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def gpu_status() -> dict:
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return {"nvidia_smi_available": False, "devices": []}
    completed = subprocess.run(
        [nvidia_smi, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "nvidia_smi_available": True,
        "returncode": completed.returncode,
        "devices": [
            line.strip() for line in completed.stdout.splitlines() if line.strip()
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audit", choices=("e5", "e6", "e7"))
    parser.add_argument(
        "--check-runtime",
        action="store_true",
        help="Report local GPU visibility and whether the relevant credential is set; never print a secret.",
    )
    arguments = parser.parse_args()
    config_names = {
        "e5": "e5_external_boundary.json",
        "e6": "e6_ccai.json",
        "e7": "e7_cost_probe.json",
    }
    config = read_json(CONFIG_DIR / config_names[arguments.audit])
    p0_path = EXPERIMENTS / "cpu" / "results" / "p0_snapshot.json"
    native_path = EXPERIMENTS / "cpu" / "results" / "native_audit.json"
    missing = [str(path) for path in (p0_path, native_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Required CPU provenance output is missing: " + ", ".join(missing)
        )
    p0 = read_json(p0_path)
    native = read_json(native_path)
    result = {
        "audit": arguments.audit.upper(),
        "protocol_schema": config["protocol_schema"],
        "p0_dual_rows": p0["releases"]["dual"]["rows"],
        "native_audit_rows": native["analysis_population"]["rows"],
        "external_calls_made": 0,
        "model_weights_loaded": False,
    }
    if arguments.audit == "e5":
        result["planned_primary_pairs"] = config["primary_sample"]["pairs"]
        result["planned_primary_positions"] = config["primary_sample"]["response_positions"]
        result["planned_pilot_positions"] = config["pilot"]["pairs"] * 2
        result["sites"] = [site["id"] for site in config["sites"]]
        if arguments.check_runtime:
            result["runtime"] = {
                "gpu": gpu_status(),
                "hf_token_present": bool(os.environ.get("HF_TOKEN")),
                "openai_api_key_present": bool(os.environ.get("OPENAI_API_KEY")),
            }
    elif arguments.audit == "e6":
        manifest = CONFIG_DIR / "e6_sample_manifest.csv"
        summary = CONFIG_DIR / "e6_sample_summary.json"
        result["sample_manifest_present"] = manifest.exists()
        result["sample_summary_present"] = summary.exists()
        result["planned_primary_judgements"] = (
            config["sample"]["pairs"] * len(config["principles"]) * 2
        )
        result["planned_repeat_judgements"] = int(
            config["sample"]["pairs"]
            * config["controls"]["repeat_batch"]["share_of_base_pairs"]
            * len(config["principles"])
            * 2
        )
        if arguments.check_runtime:
            result["runtime"] = {
                "openai_api_key_present": bool(os.environ.get("OPENAI_API_KEY"))
            }
    else:
        result["planned_primary_pairs"] = config["primary_sample"]["pairs"]
        result["planned_score_positions"] = config["primary_sample"]["response_positions"]
        result["model_id"] = config["artefact"]["model_id"]
        if arguments.check_runtime:
            result["runtime"] = {"gpu": gpu_status()}
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
