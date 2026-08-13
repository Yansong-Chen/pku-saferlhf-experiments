"""Merge standalone ShieldGemma scores back into the run tree -- phase 3.

Runs locally, inside the repository.  Takes the raw probabilities produced by
score_shieldgemma_standalone.py on the GPU machine and writes records in the
exact schema e5_run.py produces, so e5_aggregate.py consumes them unchanged.

    python3 gpu_api/scripts/ingest_gpu_scores.py \
        --phase pilot \
        --run-id e5-sg-pilot-500 \
        --scores /path/to/e5-pilot.scores.jsonl \
        --score-manifest /path/to/e5-pilot.scores.manifest.json

The decision threshold is applied here, not on the GPU machine, so the rule
stays inside the repository with the rest of the protocol.

Three refusals, all deliberate:
  * a job hash that does not match the recorded meta;
  * a scored position that is not in the regenerated task set;
  * silent gaps -- missing positions are reported and, unless --allow-partial
    is given, block the merge.
A failed score remains a failure record and is never recoded as safe.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from common import (
    CONFIG_DIR,
    CPU_RESULTS,
    EXPERIMENTS,
    base_run_manifest,
    deterministic_pilot_rows,
    primary_sample_rows,
    public_task_fields,
    read_json,
    read_jsonl,
    response_tasks,
    run_directory,
    sha256,
    utc_now,
    validate_cpu_provenance,
    write_json,
)

CONFIG_PATH = CONFIG_DIR / "e5_external_boundary.json"
PRIMARY_SAMPLE_PATH = CONFIG_DIR / "primary_pair_sample_manifest.csv"
JOB_DIR = EXPERIMENTS / "gpu_api" / "private_runs" / "jobs"
SITE_ID = "shieldgemma_9b"


def position_key(task: dict) -> str:
    return ":".join(
        (task["source_file"], str(task["source_line"]), str(task["response_position"]))
    )


def request_id(rendering: str, task: dict) -> str:
    return ":".join(
        (
            SITE_ID,
            rendering,
            task["source_file"],
            str(task["source_line"]),
            str(task["response_position"]),
        )
    )


def rows_for_phase(phase: str, config: dict):
    if phase == "pilot":
        return deterministic_pilot_rows(
            config["pilot"]["pairs"], config["pilot"]["selection_seed"]
        )
    return primary_sample_rows(PRIMARY_SAMPLE_PATH)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("pilot", "primary"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--score-manifest", required=True)
    parser.add_argument("--job-meta", default=None)
    parser.add_argument("--allow-partial", action="store_true")
    arguments = parser.parse_args()

    validate_cpu_provenance()
    config = read_json(CONFIG_PATH)
    site = next(s for s in config["sites"] if s["id"] == SITE_ID)
    threshold = float(config["decision_rule"]["shieldgemma_yes_probability_threshold"])

    job_meta_path = Path(
        arguments.job_meta or (JOB_DIR / f"e5-{arguments.phase}.meta.json")
    )
    job_meta = read_json(job_meta_path)
    score_manifest = read_json(Path(arguments.score_manifest))

    if score_manifest["job_sha256"] != job_meta["job_sha256"]:
        raise SystemExit(
            "Score manifest was produced from a different job file.\n"
            f"  job meta : {job_meta['job_sha256']}\n"
            f"  scores   : {score_manifest['job_sha256']}"
        )

    tasks = response_tasks(rows_for_phase(arguments.phase, config))
    by_key = {position_key(task): task for task in tasks}

    scores = list(read_jsonl(Path(arguments.scores)))
    unknown = sorted({r["position_key"] for r in scores} - set(by_key))
    if unknown:
        raise SystemExit(
            f"{len(unknown)} scored positions are absent from the regenerated task "
            f"set; first: {unknown[0]}. The job and the phase disagree."
        )

    # The standalone scorer is resumable.  If a batch first fails and a later
    # invocation completes it, its append-only score file contains both
    # attempts.  The experiment result must use the terminal record for each
    # (position, rendering), while keeping execution-attempt counts in the
    # manifest for transparent diagnostics.
    attempts_by_key: defaultdict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in scores:
        attempts_by_key[(record["position_key"], record["rendering"])].append(record)
    terminal_scores = {
        key: attempts[-1] for key, attempts in attempts_by_key.items()
    }
    recovered_terminal_positions = sum(
        terminal["status"] == "ok"
        and any(attempt.get("status") != "ok" for attempt in attempts[:-1])
        for attempts in attempts_by_key.values()
        for terminal in [attempts[-1]]
    )
    failed_attempt_records = sum(
        attempt.get("status") != "ok"
        for attempts in attempts_by_key.values()
        for attempt in attempts
    )

    run_dir = run_directory("e5", arguments.run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    renderings = job_meta["scorer"]["renderings"]
    written = {rendering: {"ok": 0, "failed": 0} for rendering in renderings}
    missing: dict[str, list[str]] = {}

    for rendering in renderings:
        output_path = run_dir / f"{SITE_ID}_{rendering}.jsonl"
        seen: set[str] = set()
        with output_path.open("w", encoding="utf-8") as handle:
            for task in tasks:
                key = position_key(task)
                record = terminal_scores.get((key, rendering))
                if record is None:
                    continue
                seen.add(key)
                common_fields = {
                    "record_schema": "pku-saferlhf.e5-position.v1",
                    "request_id": request_id(rendering, task),
                    "site": SITE_ID,
                    "rendering": rendering,
                    "completed_at_utc": record.get("completed_at_utc", utc_now()),
                    **public_task_fields(task),
                }
                if record.get("status") != "ok":
                    payload = {
                        **common_fields,
                        "status": "failed",
                        "error_type": record.get("error_type", "UnknownScorerFailure"),
                    }
                    written[rendering]["failed"] += 1
                else:
                    probabilities = record["policy_yes_probability"]
                    flags = {pid: value >= threshold for pid, value in probabilities.items()}
                    payload = {
                        **common_fields,
                        "status": "ok",
                        "model_id": record.get("model_id", site["model_id"]),
                        "model_revision": record["model_revision"],
                        "external_unsafe": any(flags.values()),
                        "policy_flags": flags,
                        "policy_yes_probabilities": probabilities,
                        "input_may_be_truncated": bool(record.get("input_truncated", False)),
                    }
                    written[rendering]["ok"] += 1
                handle.write(json.dumps(payload, sort_keys=True) + "\n")
        gap = sorted(set(by_key) - seen)
        if gap:
            missing[rendering] = gap

    if missing and not arguments.allow_partial:
        summary = {k: len(v) for k, v in missing.items()}
        raise SystemExit(
            f"Missing scored positions: {summary}. Resume the scorer, or pass "
            "--allow-partial to record the run as incomplete."
        )

    manifest = base_run_manifest("E5", CONFIG_PATH, arguments.run_id, Path(__file__))
    manifest.update(
        {
            "site": SITE_ID,
            "phase": arguments.phase,
            "ingest_mode": "standalone_gpu_scores",
            "job_meta_path": str(job_meta_path),
            "job_sha256": job_meta["job_sha256"],
            "scores_sha256": sha256(Path(arguments.scores)),
            "score_manifest_sha256": sha256(Path(arguments.score_manifest)),
            "scorer_environment": {
                key: score_manifest.get(key)
                for key in (
                    "model_id",
                    "resolved_revision",
                    "torch_dtype",
                    "batch_size",
                    "max_input_tokens",
                    "torch_version",
                    "transformers_version",
                    "device_name",
                    "device_capability",
                    "elapsed_seconds",
                )
            },
            "decision_rule": {"shieldgemma_yes_probability_threshold": threshold},
            "records_written": written,
            "missing_positions": {k: len(v) for k, v in missing.items()},
            "complete": not missing,
            "score_attempt_diagnostics": {
                "raw_score_records": len(scores),
                "terminal_position_renderings": len(terminal_scores),
                "failed_attempt_records": failed_attempt_records,
                "recovered_terminal_positions": recovered_terminal_positions,
            },
            "finished_at_utc": utc_now(),
            "private_output_policy": (
                "Ingested records carry labels and hashes only; no prompt or response text."
            ),
        }
    )
    write_json(run_dir / "run_manifest.json", manifest)

    print(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"\nrun directory: {run_dir}")
    print("next: python3 gpu_api/scripts/e5_aggregate.py "
          f"--private-run {run_dir.relative_to(EXPERIMENTS)} "
          f"--aggregate-dir gpu_api/results/{arguments.run_id}")


if __name__ == "__main__":
    main()
