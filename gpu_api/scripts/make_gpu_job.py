"""Materialise a self-contained ShieldGemma scoring job.

Phase 1 of the three-stage GPU workflow.  Runs locally, inside the repository,
where P0 and the native audit are available.  Emits a job file that carries
everything the remote scorer needs -- response text, the four policy
guidelines, the renderings, and the tokenisation settings -- so that the GPU
step imports nothing from this repository and depends on no directory layout.

    python3 gpu_api/scripts/make_gpu_job.py --phase pilot
    python3 gpu_api/scripts/make_gpu_job.py --phase primary

Outputs, both under the ignored private_runs tree:

    gpu_api/private_runs/jobs/e5-<phase>.jsonl        one record per response position
    gpu_api/private_runs/jobs/e5-<phase>.meta.json    provenance and scorer settings

The job file contains response text and must never be committed.  Its SHA-256
appears in the meta file and is re-checked at ingest, so a job that was edited
between phases cannot be merged silently.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import (
    CONFIG_DIR,
    CPU_RESULTS,
    EXPERIMENTS,
    deterministic_pilot_rows,
    primary_sample_rows,
    read_json,
    response_tasks,
    sha256,
    utc_now,
    validate_cpu_provenance,
    write_json,
)

CONFIG_PATH = CONFIG_DIR / "e5_external_boundary.json"
PRIMARY_SAMPLE_PATH = CONFIG_DIR / "primary_pair_sample_manifest.csv"
JOB_DIR = EXPERIMENTS / "gpu_api" / "private_runs" / "jobs"

RENDERINGS = ("prompt_response", "response_only")


def rows_for_phase(phase: str, config: dict):
    if phase == "pilot":
        return deterministic_pilot_rows(
            config["pilot"]["pairs"], config["pilot"]["selection_seed"]
        )
    if not PRIMARY_SAMPLE_PATH.exists():
        raise SystemExit(
            "Create the frozen shared primary sample with make_primary_sample.py first."
        )
    return primary_sample_rows(PRIMARY_SAMPLE_PATH)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("pilot", "primary"), required=True)
    parser.add_argument("--out-dir", default=str(JOB_DIR))
    arguments = parser.parse_args()

    validate_cpu_provenance()
    config = read_json(CONFIG_PATH)
    site = next(s for s in config["sites"] if s["id"] == "shieldgemma_9b")

    rows = rows_for_phase(arguments.phase, config)
    tasks = response_tasks(rows)

    out_dir = Path(arguments.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    job_path = out_dir / f"e5-{arguments.phase}.jsonl"
    meta_path = out_dir / f"e5-{arguments.phase}.meta.json"

    with job_path.open("w", encoding="utf-8") as handle:
        for task in tasks:
            # position_key is the stable join key.  The scorer echoes it back
            # and never needs to know how request ids are composed.
            record = {
                "position_key": ":".join(
                    (
                        task["source_file"],
                        str(task["source_line"]),
                        str(task["response_position"]),
                    )
                ),
                "prompt": task["prompt"],
                "response": task["response"],
            }
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    meta = {
        "result_schema": "pku-saferlhf.e5-gpu-job.v1",
        "audit": "E5",
        "site": site["id"],
        "phase": arguments.phase,
        "created_at_utc": utc_now(),
        "positions": len(tasks),
        "pairs": len(tasks) // 2,
        "job_file": job_path.name,
        "job_sha256": sha256(job_path),
        "p0_manifest_sha256": sha256(CPU_RESULTS / "p0_snapshot.json"),
        "native_audit_sha256": sha256(CPU_RESULTS / "native_audit.json"),
        "config_sha256": sha256(CONFIG_PATH),
        "primary_sample_manifest": (
            PRIMARY_SAMPLE_PATH.name if arguments.phase == "primary" else None
        ),
        # Everything below is what the standalone scorer consumes.  Copying it
        # here is deliberate: the GPU step must not read this repository's
        # configuration, so a divergence between the two becomes a hash
        # mismatch rather than a silent difference in rendering.
        "scorer": {
            "model_id": site["model_id"],
            "renderings": list(RENDERINGS),
            "policies": site["policies"],
            "max_input_tokens": int(config["runtime"]["max_input_tokens"]),
            "batch_size": int(config["runtime"]["shieldgemma_batch_size"]),
            "torch_dtype": "bfloat16",
            "chat_template_call": {
                "prompt_response_messages": "[user: prompt, assistant: response]",
                "response_only_messages": "[user: response]",
                "tokenize": False,
                "add_generation_prompt": False,
                "guideline_kwarg": True,
            },
            "score_definition": (
                "softmax over the final-position logits restricted to the exact "
                "'Yes' and 'No' vocabulary ids; the retained value is P(Yes)."
            ),
        },
        "decision_rule_applied_at_ingest": {
            "shieldgemma_yes_probability_threshold": float(
                config["decision_rule"]["shieldgemma_yes_probability_threshold"]
            ),
            "note": (
                "The scorer records raw probabilities only. Thresholding happens "
                "locally at ingest so the decision rule stays inside the repository."
            ),
        },
        "handling_rules": {
            "failures": "A scoring failure remains a failure record and is never recoded as safe.",
            "text_policy": "The job file carries response text and must never be committed.",
        },
    }
    write_json(meta_path, meta)

    print(json.dumps({k: v for k, v in meta.items() if k != "scorer"}, indent=2, sort_keys=True))
    print(f"\njob  : {job_path}")
    print(f"meta : {meta_path}")
    print(f"size : {job_path.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
