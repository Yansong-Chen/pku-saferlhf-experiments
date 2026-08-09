#!/usr/bin/env python3
"""Run the E6 CCAI comparison through the OpenAI Chat Completions API.

The runner rehydrates selected prompts and responses only in memory from the
pinned raw release.  Its private JSONL log contains source coordinates, hashes,
labels, token usage, and API outcomes, but no prompt or response text.

Run a dry-run first, then a small pilot:

  python3 gpu_api/scripts/e6_run.py --phase pilot --run-id e6-pilot --dry-run
  python3 gpu_api/scripts/e6_run.py --phase pilot --run-id e6-pilot
  python3 gpu_api/scripts/e6_run.py --phase primary --run-id e6-primary
  python3 gpu_api/scripts/e6_run.py --phase repeat --run-id e6-primary
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
import threading
import time
from pathlib import Path

from common import (
    CONFIG_DIR,
    ProtocolError,
    append_jsonl,
    base_run_manifest,
    environment_present,
    existing_ids,
    load_rows_for_manifest,
    public_task_fields,
    read_json,
    run_directory,
    sha256,
    utc_now,
    validate_cpu_provenance,
    write_json,
)


SCRIPT = Path(__file__).resolve()
CONFIG_PATH = CONFIG_DIR / "e6_ccai.json"
SAMPLE_PATH = CONFIG_DIR / "e6_sample_manifest.csv"
THREAD_LOCAL = threading.local()


def parse_label(content: str | None) -> str | None:
    if content is None:
        return None
    normalized = content.strip().strip('"').strip("'").strip().lower()
    if normalized == "a":
        return "A"
    if normalized == "b":
        return "B"
    if normalized in {"no preference", "no-preference", "nopreference"}:
        return "No preference"
    return None


def client():
    if getattr(THREAD_LOCAL, "client", None) is None:
        try:
            from openai import OpenAI
        except ImportError as error:
            raise ProtocolError("Install the openai package before running E6.") from error
        THREAD_LOCAL.client = OpenAI()
    return THREAD_LOCAL.client


def selected_pairs(phase: str, config: dict) -> list[tuple[dict, dict]]:
    pairs = load_rows_for_manifest(SAMPLE_PATH)
    if phase == "primary":
        return pairs
    if phase == "pilot":
        size = int(config["runtime"]["pilot_pairs"])
        return random.Random(config["sample"]["seed"] + 1).sample(pairs, size)
    repeat_size = int(
        len(pairs) * float(config["controls"]["repeat_batch"]["share_of_base_pairs"])
    )
    return random.Random(config["controls"]["repeat_batch"]["selection_seed"]).sample(
        pairs, repeat_size
    )


def request_id(phase: str, manifest: dict, principle_index: int, order: int) -> str:
    return ":".join(
        (
            phase,
            manifest["source_file"],
            str(manifest["source_line"]),
            str(principle_index),
            str(order),
        )
    )


def task_records(phase: str, selected: list[tuple[dict, dict]], config: dict) -> list[dict]:
    records: list[dict] = []
    for manifest, row in selected:
        for principle_index, principle in enumerate(config["principles"]):
            for order in (0, 1):
                response_a_id, response_b_id = ((0, 1) if order == 0 else (1, 0))
                records.append(
                    {
                        "request_id": request_id(phase, manifest, principle_index, order),
                        "phase": phase,
                        "manifest": manifest,
                        "row": row,
                        "principle_index": principle_index,
                        "principle": principle,
                        "order": order,
                        "response_a_id": response_a_id,
                        "response_b_id": response_b_id,
                    }
                )
    return records


def prompt_for(task: dict, config: dict) -> str:
    row = task["row"]
    return config["oracle"]["template"].format(
        prompt=row["prompt"],
        principle=task["principle"],
        response_a=row[f"response_{task['response_a_id']}"],
        response_b=row[f"response_{task['response_b_id']}"],
    )


def call(task: dict, config: dict) -> dict:
    oracle = config["oracle"]
    attempts = int(config["runtime"]["max_retries"])
    delay = float(config["runtime"]["retry_base_seconds"])
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            result = client().chat.completions.create(
                model=oracle["model"],
                temperature=oracle["temperature"],
                max_tokens=oracle["max_tokens"],
                messages=[
                    {
                        "role": "system",
                        "content": "Return exactly one of: A, B, or No preference.",
                    },
                    {"role": "user", "content": prompt_for(task, config)},
                ],
            )
            content = result.choices[0].message.content
            label = parse_label(content)
            if label is None:
                raise ValueError("UnparseableModelOutput")
            usage = getattr(result, "usage", None)
            return {
                "status": "ok",
                "returned_model": getattr(result, "model", oracle["model"]),
                "api_request_id": getattr(result, "id", None),
                "label": label,
                "usage": {
                    "prompt_tokens": getattr(usage, "prompt_tokens", None),
                    "completion_tokens": getattr(usage, "completion_tokens", None),
                    "total_tokens": getattr(usage, "total_tokens", None),
                },
            }
        except Exception as error:
            last_error = error
            if attempt + 1 < attempts:
                time.sleep(delay * (2**attempt))
    assert last_error is not None
    return {"status": "failed", "error_type": type(last_error).__name__}


def result_record(task: dict, outcome: dict) -> dict:
    manifest = task["manifest"]
    row = task["row"]
    # public_task_fields expects a response-position task; pair-level E6 keeps
    # its own source metadata and never emits either response text.
    record = {
        "record_schema": "pku-saferlhf.e6-principle-order.v1",
        "request_id": task["request_id"],
        "status": outcome["status"],
        "phase": task["phase"],
        "source_file": manifest["source_file"],
        "source_line": int(manifest["source_line"]),
        "response_0_sha256": manifest["response_0_sha256"],
        "response_1_sha256": manifest["response_1_sha256"],
        "native_stratum": manifest["stratum"],
        "stratum_population_N_h": int(manifest["stratum_population_N_h"]),
        "stratum_sample_n_h": int(manifest["stratum_sample_n_h"]),
        "design_weight_N_h_over_n_h": float(manifest["design_weight_N_h_over_n_h"]),
        "safer_response_id": int(row["safer_response_id"]),
        "principle_index": task["principle_index"],
        "order": task["order"],
        "response_a_id": task["response_a_id"],
        "response_b_id": task["response_b_id"],
        "completed_at_utc": utc_now(),
    }
    record.update(outcome)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("pilot", "primary", "repeat"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()

    validate_cpu_provenance()
    config = read_json(CONFIG_PATH)
    selected = selected_pairs(arguments.phase, config)
    tasks = task_records(arguments.phase, selected, config)
    if arguments.limit is not None:
        tasks = tasks[: arguments.limit]
    run_dir = run_directory("e6", arguments.run_id)
    output_path = run_dir / f"{arguments.phase}_judgements.jsonl"
    completed = existing_ids(output_path)
    todo = [task for task in tasks if task["request_id"] not in completed]
    metadata = base_run_manifest("E6", CONFIG_PATH, arguments.run_id, SCRIPT)
    metadata.update(
        {
            "phase": arguments.phase,
            "sample_manifest": str(SAMPLE_PATH.relative_to(CONFIG_DIR.parent.parent)),
            "sample_manifest_sha256": sha256(SAMPLE_PATH),
            "selected_pairs": len(selected),
            "planned_judgements": len(tasks),
            "remaining_judgements": len(todo),
            "workers": arguments.workers or int(config["runtime"]["workers"]),
            "dry_run": arguments.dry_run,
        }
    )
    write_json(run_dir / "run_manifest.json", metadata)
    if arguments.dry_run:
        print(json.dumps(metadata, indent=2, sort_keys=True))
        return
    if not environment_present("OPENAI_API_KEY"):
        raise ProtocolError("OPENAI_API_KEY is required for E6.")
    workers = arguments.workers or int(config["runtime"]["workers"])
    completed_count = failed_count = 0
    returned_models: set[str] = set()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(call, task, config): task for task in todo}
        for future in concurrent.futures.as_completed(futures):
            task = futures[future]
            try:
                outcome = future.result()
            except Exception as error:
                outcome = {"status": "failed", "error_type": type(error).__name__}
            append_jsonl(output_path, result_record(task, outcome))
            if outcome["status"] == "ok":
                completed_count += 1
                returned_models.add(str(outcome["returned_model"]))
            else:
                failed_count += 1
    metadata["finished_at_utc"] = utc_now()
    metadata["new_completed_judgements"] = completed_count
    metadata["new_failed_judgements"] = failed_count
    metadata["returned_models"] = sorted(returned_models)
    write_json(run_dir / "run_manifest.json", metadata)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
