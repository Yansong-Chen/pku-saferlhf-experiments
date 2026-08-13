#!/usr/bin/env python3
"""Standalone, resumable Colab scorer for the E7 Beaver cost-model probe.

The script reads a frozen private job file containing the already selected
PKU-SafeRLHF positions.  It never needs the full release in Colab and it
never writes prompt or response text to a score file.  Re-running ``score``
after an interruption resumes from successful positions; a previous failed
attempt can be superseded by a later successful attempt.

The score convention is fixed by the author implementation: a safer response
has lower cost and a safe response has negative cost.  Thus higher scores
denote greater released unsafe risk.  The pilot still reports both AUROC
orientations as a runtime check.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCORER_VARIANT = "pku-safe-rlhf-automodelforscore-v1"
MODEL_ID = "PKU-Alignment/beaver-7b-v2.0-cost"
MODEL_REVISION = "26bf7161f09fee958ae64c8b4bb70fb420f7ba39"
SAFE_RLHF_REVISION = "e8cca16665ef2340ac92c6514f05519310251581"
CONVERSATION_TEMPLATE = "BEGINNING OF CONVERSATION: USER: {prompt} ASSISTANT: {response}"
SCORE_DIRECTION = "higher_is_unsafe"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def terminal_records(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Use the final append-order attempt for each frozen position."""

    terminal: dict[str, dict[str, Any]] = {}
    for record in records:
        position_key = record.get("position_key")
        if position_key:
            terminal[str(position_key)] = record
    return terminal


def successful_keys(records: list[dict[str, Any]]) -> set[str]:
    return {
        key
        for key, record in terminal_records(records).items()
        if record.get("status") == "ok"
    }


def load_job(job_path: Path, meta_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    meta = read_json(meta_path)
    if meta.get("job_schema") != "pku-saferlhf.e7-colab-job.v1":
        raise SystemExit("Unexpected E7 job schema. Use the supplied job and metadata together.")
    observed = sha256_file(job_path)
    if observed != meta.get("job_sha256"):
        raise SystemExit(
            "Job SHA-256 mismatch. The uploaded job is not the frozen job described by its metadata."
        )
    tasks = read_jsonl(job_path)
    expected = int(meta.get("expected_response_positions", 0))
    if len(tasks) != expected:
        raise SystemExit(f"Expected {expected} job positions, found {len(tasks)}.")
    keys = [str(task.get("position_key", "")) for task in tasks]
    if not all(keys) or len(set(keys)) != len(keys):
        raise SystemExit("The job must contain one unique non-empty position_key per task.")
    required = {
        "position_key",
        "pair_id",
        "source_file",
        "source_line",
        "response_position",
        "response_sha256",
        "is_safe",
        "severity_level",
        "harm_categories",
        "response_character_length",
        "safer_response_id",
        "native_stratum",
        "prompt",
        "response",
    }
    missing = sorted(required - set(tasks[0])) if tasks else sorted(required)
    if missing:
        raise SystemExit("Frozen job misses required fields: " + ", ".join(missing))
    return tasks, meta


def validate_frozen_protocol(meta: dict[str, Any], revision: str, direction: str, max_length: int) -> None:
    model = meta.get("model", {})
    if model.get("model_id") != MODEL_ID:
        raise SystemExit("Unexpected model identifier in job metadata.")
    if revision != model.get("revision") or revision != MODEL_REVISION:
        raise SystemExit("The scorer accepts only the frozen model revision recorded in the job metadata.")
    if direction != model.get("score_direction") or direction != SCORE_DIRECTION:
        raise SystemExit("The scorer accepts only the author-documented higher_is_unsafe direction.")
    if max_length != int(meta.get("runtime", {}).get("max_input_tokens", -1)):
        raise SystemExit("max_input_tokens is frozen by the job metadata and cannot be changed in Colab.")
    if model.get("conversation_template") != CONVERSATION_TEMPLATE:
        raise SystemExit("Unexpected conversation template in job metadata.")


def public_task_fields(task: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "position_key",
        "pair_id",
        "source_file",
        "source_line",
        "response_position",
        "response_sha256",
        "is_safe",
        "severity_level",
        "harm_categories",
        "response_character_length",
        "safer_response_id",
        "native_stratum",
        "pair_inclusion_probability",
        "pair_design_weight",
        "sampling_manifest",
    )
    return {key: task[key] for key in keys if key in task}


def conversation(task: dict[str, Any]) -> str:
    return CONVERSATION_TEMPLATE.format(prompt=task["prompt"], response=task["response"])


def model_kwargs(revision: str) -> dict[str, str]:
    token = os.environ.get("HF_TOKEN")
    return {"revision": revision, **({"token": token} if token else {})}


def load_runtime(revision: str) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
    try:
        import torch
        import transformers
        from safe_rlhf.models import AutoModelForScore
        from transformers import AutoTokenizer
    except ImportError as error:
        raise SystemExit(
            "Install the exact notebook dependencies before loading the model."
        ) from error
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required. In Colab, select an A100/L4 GPU runtime first.")
    if not torch.cuda.is_bf16_supported():
        raise SystemExit("The frozen protocol requires a GPU with bfloat16 support.")
    kwargs = model_kwargs(revision)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, **kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # ScoreModelMixin selects the final unmasked token, so right-padding is valid
    # and keeps the tokenizer's standard behaviour explicit in the manifest.
    tokenizer.padding_side = "right"
    model = AutoModelForScore.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        **kwargs,
    )
    model.eval()
    device = next(model.parameters()).device
    resolved = getattr(model.config, "_commit_hash", None) or revision
    environment = {
        "scorer_variant": SCORER_VARIANT,
        "safe_rlhf_revision": SAFE_RLHF_REVISION,
        "model_id": MODEL_ID,
        "requested_model_revision": revision,
        "resolved_model_revision": resolved,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "torch_dtype": str(getattr(model, "dtype", "unknown")),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0),
        "device_capability": list(torch.cuda.get_device_capability(0)),
        "tokenizer_padding_side": tokenizer.padding_side,
    }
    if resolved != revision:
        raise SystemExit(
            f"Model revision mismatch: requested {revision}, resolved {resolved}. Stop rather than mix weights."
        )
    return tokenizer, model, torch, device, environment


def encoded_batch(tokenizer: Any, tasks: list[dict[str, Any]], max_length: int, device: Any) -> Any:
    return tokenizer(
        [conversation(task) for task in tasks],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(device)


def score_tasks(
    tokenizer: Any,
    model: Any,
    torch: Any,
    device: Any,
    tasks: list[dict[str, Any]],
    max_length: int,
) -> tuple[list[float], list[int]]:
    encoded = encoded_batch(tokenizer, tasks, max_length, device)
    with torch.inference_mode():
        output = model(**encoded)
    scores = output.end_scores.detach().float().cpu().reshape(-1).tolist()
    lengths = [int(value) for value in encoded["attention_mask"].sum(dim=1).tolist()]
    if len(scores) != len(tasks):
        raise RuntimeError("The cost model returned a score count that does not match the input batch.")
    return [float(value) for value in scores], lengths


def manifest_path_for(scores_path: Path) -> Path:
    if scores_path.name.endswith(".jsonl"):
        return scores_path.with_name(scores_path.name[:-6] + ".manifest.json")
    return scores_path.with_name(scores_path.name + ".manifest.json")


def update_manifest(
    *,
    scores_path: Path,
    job_path: Path,
    meta_path: Path,
    meta: dict[str, Any],
    environment: dict[str, Any],
    batch_size: int,
    max_length: int,
    started: str,
    completed: int,
    failed: int,
) -> Path:
    path = manifest_path_for(scores_path)
    previous = read_json(path) if path.exists() else {}
    invocations = list(previous.get("invocations", []))
    records = read_jsonl(scores_path)
    terminal = terminal_records(records)
    status_counts = Counter(record.get("status", "unknown") for record in terminal.values())
    invocation = {
        "started_at_utc": started,
        "finished_at_utc": utc_now(),
        "batch_size": batch_size,
        "max_input_tokens": max_length,
        "new_successful_scores": completed,
        "new_failed_attempts": failed,
        "terminal_status_counts": dict(sorted(status_counts.items())),
    }
    invocations.append(invocation)
    payload = {
        "manifest_schema": "pku-saferlhf.e7-colab-score-manifest.v1",
        "created_at_utc": previous.get("created_at_utc", started),
        "updated_at_utc": utc_now(),
        "job_path": job_path.name,
        "job_sha256": sha256_file(job_path),
        "job_meta_path": meta_path.name,
        "job_meta_sha256": sha256_file(meta_path),
        "phase": meta["phase"],
        "expected_response_positions": meta["expected_response_positions"],
        "model": meta["model"],
        "runtime": {"max_input_tokens": max_length},
        "environment": environment,
        "python": sys.version,
        "platform": platform.platform(),
        "private_output_policy": "Score records contain identifiers and scores; prompt and response text remain in the private job file only.",
        "resume_policy": "Only successful terminal positions are skipped. A later successful attempt supersedes a recorded failed attempt.",
        "invocations": invocations,
    }
    write_json(path, payload)
    return path


def command_score(arguments: argparse.Namespace) -> None:
    job_path, meta_path, out_path = Path(arguments.job), Path(arguments.meta), Path(arguments.out)
    tasks, meta = load_job(job_path, meta_path)
    validate_frozen_protocol(meta, arguments.revision, arguments.score_direction, arguments.max_input_tokens)
    if arguments.batch_size < 1:
        raise SystemExit("batch-size must be positive.")
    existing = read_jsonl(out_path)
    completed_keys = successful_keys(existing)
    todo = [task for task in tasks if task["position_key"] not in completed_keys]
    if arguments.limit is not None:
        todo = todo[: arguments.limit]
    started = utc_now()
    tokenizer, model, torch, device, environment = load_runtime(arguments.revision)
    new_completed = new_failed = 0
    for start in range(0, len(todo), arguments.batch_size):
        batch = todo[start : start + arguments.batch_size]
        try:
            scores, lengths = score_tasks(tokenizer, model, torch, device, batch, arguments.max_input_tokens)
            for task, score, length in zip(batch, scores, lengths):
                append_jsonl(
                    out_path,
                    {
                        # This is deliberately the local E7 schema too, so the
                        # reviewed score file can be aggregated by the repository
                        # script without a lossy conversion step.
                        "record_schema": "pku-saferlhf.e7-score.v1",
                        "request_id": task["position_key"],
                        "status": "ok",
                        "completed_at_utc": utc_now(),
                        "model_id": MODEL_ID,
                        "model_revision": environment["resolved_model_revision"],
                        "score_direction": SCORE_DIRECTION,
                        "raw_cost_score": score,
                        "input_token_count": length,
                        "input_may_be_truncated": length >= arguments.max_input_tokens,
                        **public_task_fields(task),
                    },
                )
                new_completed += 1
        except Exception as error:  # Keep the no-text terminal failure auditable.
            for task in batch:
                append_jsonl(
                    out_path,
                    {
                        "record_schema": "pku-saferlhf.e7-score.v1",
                        "request_id": task["position_key"],
                        "status": "failed",
                        "completed_at_utc": utc_now(),
                        "model_id": MODEL_ID,
                        "model_revision": environment["resolved_model_revision"],
                        "score_direction": SCORE_DIRECTION,
                        "error_type": type(error).__name__,
                        **public_task_fields(task),
                    },
                )
                new_failed += 1
            torch.cuda.empty_cache()
            print(
                f"Batch beginning at {start} failed with {type(error).__name__}. "
                "It can be retried by re-running this cell with a smaller batch size.",
                file=sys.stderr,
            )
            if arguments.stop_on_error:
                raise
        if (start // arguments.batch_size + 1) % arguments.progress_every == 0:
            print(
                json.dumps(
                    {
                        "processed_this_invocation": min(start + len(batch), len(todo)),
                        "todo_this_invocation": len(todo),
                        "new_ok": new_completed,
                        "new_failed_attempts": new_failed,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    manifest = update_manifest(
        scores_path=out_path,
        job_path=job_path,
        meta_path=meta_path,
        meta=meta,
        environment=environment,
        batch_size=arguments.batch_size,
        max_length=arguments.max_input_tokens,
        started=started,
        completed=new_completed,
        failed=new_failed,
    )
    print(
        json.dumps(
            {
                "scores": str(out_path),
                "manifest": str(manifest),
                "new_ok": new_completed,
                "new_failed_attempts": new_failed,
                "skipped_existing_ok": len(completed_keys),
            },
            indent=2,
            sort_keys=True,
        )
    )


def command_probe(arguments: argparse.Namespace) -> None:
    job_path, meta_path, report_path = Path(arguments.job), Path(arguments.meta), Path(arguments.report)
    tasks, meta = load_job(job_path, meta_path)
    validate_frozen_protocol(meta, arguments.revision, arguments.score_direction, arguments.max_input_tokens)
    if arguments.batch_size < 1:
        raise SystemExit("batch-size must be positive.")
    # Long character strings are a conservative deterministic proxy for long token inputs.
    batch = sorted(tasks, key=lambda task: int(task["response_character_length"]), reverse=True)[: arguments.batch_size]
    started = utc_now()
    try:
        tokenizer, model, torch, device, environment = load_runtime(arguments.revision)
        torch.cuda.reset_peak_memory_stats(device)
        scores, lengths = score_tasks(tokenizer, model, torch, device, batch, arguments.max_input_tokens)
        payload: dict[str, Any] = {
            "report_schema": "pku-saferlhf.e7-colab-probe.v1",
            "status": "ok",
            "started_at_utc": started,
            "finished_at_utc": utc_now(),
            "batch_size": arguments.batch_size,
            "positions_scored": len(scores),
            "max_observed_input_tokens": max(lengths),
            "input_may_be_truncated_positions": sum(length >= arguments.max_input_tokens for length in lengths),
            "peak_allocated_cuda_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_cuda_bytes": int(torch.cuda.max_memory_reserved(device)),
            "environment": environment,
        }
    except Exception as error:
        payload = {
            "report_schema": "pku-saferlhf.e7-colab-probe.v1",
            "status": "failed",
            "started_at_utc": started,
            "finished_at_utc": utc_now(),
            "batch_size": arguments.batch_size,
            "error_type": type(error).__name__,
            "traceback_tail": traceback.format_exc().splitlines()[-8:],
        }
    write_json(report_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "ok":
        raise SystemExit(1)


def auc(labels: list[bool], scores: list[float]) -> float | None:
    positive = sum(labels)
    negative = len(labels) - positive
    if not positive or not negative:
        return None
    indexed = sorted(enumerate(scores), key=lambda item: item[1])
    ranks = [0.0] * len(scores)
    start = 0
    while start < len(indexed):
        end = start + 1
        while end < len(indexed) and indexed[end][1] == indexed[start][1]:
            end += 1
        rank = (start + 1 + end) / 2
        for index, _ in indexed[start:end]:
            ranks[index] = rank
        start = end
    positive_rank_sum = sum(rank for rank, label in zip(ranks, labels) if label)
    return (positive_rank_sum - positive * (positive + 1) / 2) / (positive * negative)


def describe(values: list[float]) -> dict[str, float | int | None]:
    ordered = sorted(values)
    if not ordered:
        return {"n": 0, "mean": None, "minimum": None, "median": None, "maximum": None}
    return {
        "n": len(ordered),
        "mean": sum(ordered) / len(ordered),
        "minimum": ordered[0],
        "median": ordered[len(ordered) // 2],
        "maximum": ordered[-1],
    }


def command_orientation(arguments: argparse.Namespace) -> None:
    records = [
        record
        for record in terminal_records(read_jsonl(Path(arguments.scores))).values()
        if record.get("record_schema") == "pku-saferlhf.e7-score.v1" and record.get("status") == "ok"
    ]
    if not records:
        raise SystemExit("No successful score records were found.")
    labels = [not bool(record["is_safe"]) for record in records]
    raw = [float(record["raw_cost_score"]) for record in records]
    result = {
        "result_schema": "pku-saferlhf.e7-colab-pilot-orientation.v1",
        "created_at_utc": utc_now(),
        "scored_positions": len(records),
        "raw_score_by_released_state": {
            "released_unsafe": describe([score for score, label in zip(raw, labels) if label]),
            "released_safe": describe([score for score, label in zip(raw, labels) if not label]),
        },
        "auroc_if_higher_score_means_unsafe": auc(labels, raw),
        "auroc_if_lower_score_means_unsafe": auc(labels, [-score for score in raw]),
        "frozen_primary_direction": SCORE_DIRECTION,
        "direction_rationale": (
            "PKU's released CostTrainer says safer samples have lower costs and safe samples "
            "have negative costs; this pilot reports both orientations only as an execution check."
        ),
    }
    write_json(Path(arguments.output), result)
    print(json.dumps(result, indent=2, sort_keys=True))


def command_verify(arguments: argparse.Namespace) -> None:
    tasks, meta = load_job(Path(arguments.job), Path(arguments.meta))
    records = read_jsonl(Path(arguments.scores))
    terminal = terminal_records(records)
    expected = {task["position_key"] for task in tasks}
    unexpected = sorted(set(terminal) - expected)
    missing = sorted(expected - set(terminal))
    failed = sorted(
        key for key, record in terminal.items() if record.get("status") != "ok"
    )
    wrong_model = [
        key
        for key, record in terminal.items()
        if record.get("status") == "ok"
        and (record.get("model_id") != MODEL_ID or record.get("model_revision") != MODEL_REVISION)
    ]
    result = {
        "result_schema": "pku-saferlhf.e7-colab-completeness.v1",
        "created_at_utc": utc_now(),
        "phase": meta["phase"],
        "expected_response_positions": len(expected),
        "terminal_records": len(terminal),
        "successful_terminal_records": sum(record.get("status") == "ok" for record in terminal.values()),
        "failed_terminal_records": len(failed),
        "missing_positions": len(missing),
        "unexpected_positions": len(unexpected),
        "wrong_model_or_revision_positions": len(wrong_model),
        "complete": not (missing or unexpected or failed or wrong_model),
    }
    write_json(Path(arguments.output), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["complete"]:
        raise SystemExit(1)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def frozen_options(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--revision", default=MODEL_REVISION)
        subparser.add_argument("--score-direction", default=SCORE_DIRECTION, choices=(SCORE_DIRECTION,))
        subparser.add_argument("--max-input-tokens", type=int, default=2048)

    score = subparsers.add_parser("score", help="Score a frozen job and resume from successful positions.")
    score.add_argument("--job", required=True)
    score.add_argument("--meta", required=True)
    score.add_argument("--out", required=True)
    score.add_argument("--batch-size", type=int, required=True)
    score.add_argument("--limit", type=int)
    score.add_argument("--progress-every", type=int, default=10)
    score.add_argument("--stop-on-error", action="store_true")
    frozen_options(score)
    score.set_defaults(handler=command_score)

    probe = subparsers.add_parser("probe", help="Score one deterministic long-input batch without saving scores.")
    probe.add_argument("--job", required=True)
    probe.add_argument("--meta", required=True)
    probe.add_argument("--report", required=True)
    probe.add_argument("--batch-size", type=int, required=True)
    frozen_options(probe)
    probe.set_defaults(handler=command_probe)

    orientation = subparsers.add_parser("orientation", help="Report the pilot score distribution and both AUROC orientations.")
    orientation.add_argument("--scores", required=True)
    orientation.add_argument("--output", required=True)
    orientation.set_defaults(handler=command_orientation)

    verify = subparsers.add_parser("verify", help="Verify that all frozen positions have a successful terminal score.")
    verify.add_argument("--job", required=True)
    verify.add_argument("--meta", required=True)
    verify.add_argument("--scores", required=True)
    verify.add_argument("--output", required=True)
    verify.set_defaults(handler=command_verify)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    arguments.handler(arguments)


if __name__ == "__main__":
    main()
