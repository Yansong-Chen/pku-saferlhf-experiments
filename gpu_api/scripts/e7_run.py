#!/usr/bin/env python3
"""Score the released PKU cost-model artefact for E7.

The runner uses the scorer and conversation template published on the model
card.  A pilot may use the candidate revision.  A full run requires an explicit
resolved revision and a manually frozen score direction, because a cost-model
sign convention cannot be inferred safely from its model identifier alone.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

from common import (
    CONFIG_DIR,
    ProtocolError,
    all_release_rows,
    append_jsonl,
    base_run_manifest,
    deterministic_pilot_rows,
    environment_present,
    existing_ids,
    public_task_fields,
    read_json,
    response_tasks,
    run_directory,
    utc_now,
    validate_cpu_provenance,
    write_json,
)


SCRIPT = Path(__file__).resolve()
CONFIG_PATH = CONFIG_DIR / "e7_cost_probe.json"


def record_id(task: dict) -> str:
    return ":".join(
        (
            task["source_file"],
            str(task["source_line"]),
            str(task["response_position"]),
        )
    )


def selected_rows(phase: str, config: dict) -> list[tuple[str, int, dict]]:
    if phase == "pilot":
        return deterministic_pilot_rows(
            config["pilot"]["pairs"], config["pilot"]["selection_seed"]
        )
    return list(all_release_rows())


def load_scorer(config: dict, revision: str):
    try:
        import torch
        from safe_rlhf.models import AutoModelForScore
        from transformers import AutoTokenizer
    except ImportError as error:
        raise ProtocolError(
            "E7 requires torch, transformers, and the PKU safe_rlhf package. "
            "See gpu_api/RUNBOOK.md for an isolated environment."
        ) from error
    artefact = config["artefact"]
    tokenizer = AutoTokenizer.from_pretrained(artefact["model_id"], revision=revision)
    model = AutoModelForScore.from_pretrained(
        artefact["model_id"],
        revision=revision,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    resolved = getattr(model.config, "_commit_hash", None) or revision
    device = next(model.parameters()).device
    return tokenizer, model, torch, device, resolved


def conversation(task: dict, template: str) -> str:
    return template.format(prompt=task["prompt"], response=task["response"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("pilot", "full"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model-revision")
    parser.add_argument(
        "--score-direction",
        choices=("higher_is_unsafe", "lower_is_unsafe", "unknown"),
        default="unknown",
    )
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()

    validate_cpu_provenance()
    config = read_json(CONFIG_PATH)
    if arguments.phase == "full" and not arguments.model_revision:
        raise ProtocolError("A full E7 run requires --model-revision from the completed pilot.")
    if arguments.phase == "full" and arguments.score_direction == "unknown":
        raise ProtocolError(
            "A full E7 run requires a frozen --score-direction after pilot review."
        )
    revision = arguments.model_revision or config["artefact"]["candidate_revision"]
    rows = selected_rows(arguments.phase, config)
    tasks = response_tasks(rows)
    run_dir = run_directory("e7", arguments.run_id)
    output_path = run_dir / f"{arguments.phase}_scores.jsonl"
    completed = existing_ids(output_path)
    todo = [task for task in tasks if record_id(task) not in completed]
    metadata = base_run_manifest("E7", CONFIG_PATH, arguments.run_id, SCRIPT)
    metadata.update(
        {
            "phase": arguments.phase,
            "candidate_or_requested_revision": revision,
            "score_direction": arguments.score_direction,
            "pair_rows": len(rows),
            "response_positions": len(tasks),
            "remaining_response_positions": len(todo),
            "dry_run": arguments.dry_run,
        }
    )
    write_json(run_dir / "run_manifest.json", metadata)
    if arguments.dry_run:
        print(json.dumps(metadata, indent=2, sort_keys=True))
        return
    if not environment_present("HF_TOKEN"):
        # The model is currently public, but an explicit token makes retrieval
        # reproducible and avoids silently using a different local cache.
        raise ProtocolError("Set HF_TOKEN before an E7 model retrieval.")
    tokenizer, model, torch, device, resolved = load_scorer(config, revision)
    metadata["resolved_model_revision"] = resolved
    metadata["torch_dtype"] = str(getattr(model, "dtype", "unknown"))
    batch_size = arguments.batch_size or int(config["runtime"]["batch_size"])
    max_length = int(config["runtime"]["max_input_tokens"])
    template = config["artefact"]["documented_conversation_template"]
    completed_count = failed_count = 0
    for start in range(0, len(todo), batch_size):
        batch = todo[start : start + batch_size]
        try:
            texts = [conversation(task, template) for task in batch]
            encoded = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(device)
            with torch.no_grad():
                output = model(**encoded)
            scores = output.end_scores.detach().float().cpu().reshape(-1).tolist()
            lengths = encoded["attention_mask"].sum(dim=1).tolist()
            if len(scores) != len(batch):
                raise ProtocolError("Cost scorer returned a score count that does not match input count.")
            for task, score, length in zip(batch, scores, lengths):
                append_jsonl(
                    output_path,
                    {
                        "record_schema": "pku-saferlhf.e7-score.v1",
                        "request_id": record_id(task),
                        "status": "ok",
                        "model_id": config["artefact"]["model_id"],
                        "model_revision": resolved,
                        "score_direction": arguments.score_direction,
                        "completed_at_utc": utc_now(),
                        **public_task_fields(task),
                        "raw_cost_score": float(score),
                        "input_token_count": int(length),
                        "input_may_be_truncated": int(length) >= max_length,
                    },
                )
                completed_count += 1
        except Exception as error:
            for task in batch:
                append_jsonl(
                    output_path,
                    {
                        "record_schema": "pku-saferlhf.e7-score.v1",
                        "request_id": record_id(task),
                        "status": "failed",
                        "completed_at_utc": utc_now(),
                        **public_task_fields(task),
                        "error_type": type(error).__name__,
                    },
                )
                failed_count += 1
    metadata["finished_at_utc"] = utc_now()
    metadata["new_completed_scores"] = completed_count
    metadata["new_failed_scores"] = failed_count
    write_json(run_dir / "run_manifest.json", metadata)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
