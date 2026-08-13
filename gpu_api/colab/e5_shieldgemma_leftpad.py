#!/usr/bin/env python3
"""Validated left-padding scorer for the E5 ShieldGemma primary run.

This is deliberately separate from the original right-padding scorer.  It uses
Gemma's last-logit option to avoid materialising logits for every input token.
The primary job must only be run after this scorer has passed `compare` against
the completed right-padding pilot scores, with no threshold-label changes.

Examples
--------
python e5_shieldgemma_leftpad.py score \
  --job e5-pilot.jsonl --meta e5-pilot.meta.json \
  --out e5-pilot-leftpad.scores.jsonl --revision <pilot-commit> --batch-size 4

python e5_shieldgemma_leftpad.py compare \
  --job e5-pilot.jsonl --meta e5-pilot.meta.json \
  --baseline e5-pilot-rightpad-baseline.scores.jsonl \
  --candidate e5-pilot-leftpad.scores.jsonl \
  --report e5-leftpad-equivalence.json

python e5_shieldgemma_leftpad.py probe \
  --job e5-primary.jsonl --meta e5-primary.meta.json \
  --revision <pilot-commit> --batch-size 16 --report e5-primary-probe-16.json
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import platform
import sys
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCORER_VARIANT = "left-padding-last-logit-v1"
POLICY_THRESHOLD = 0.5


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def task_key(position_key: str, rendering: str) -> tuple[str, str]:
    return position_key, rendering


def expected_keys(items: list[dict[str, Any]], renderings: list[str]) -> set[tuple[str, str]]:
    return {(item["position_key"], rendering) for item in items for rendering in renderings}


def terminal_records(records: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    """Return the append-order terminal attempt for every position/rendering."""
    terminal: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        terminal[task_key(record["position_key"], record["rendering"])] = record
    return terminal


def successful_keys(records: list[dict[str, Any]]) -> set[tuple[str, str]]:
    """A later success supersedes a previous recorded failure on resume."""
    return {
        key
        for key, record in terminal_records(records).items()
        if record.get("status") == "ok"
    }


def model_kwargs(revision: str) -> dict[str, str]:
    kwargs = {"revision": revision}
    token = os.environ.get("HF_TOKEN")
    if token:
        kwargs["token"] = token
    return kwargs


def load_runtime(meta: dict[str, Any], revision: str):
    try:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise SystemExit("Install torch, transformers, and accelerate before scoring.") from error

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this scorer.")
    capability = torch.cuda.get_device_capability()
    if capability[0] < 8:
        raise SystemExit(
            f"GPU compute capability {capability} does not support the frozen bfloat16 protocol."
        )

    scorer = meta["scorer"]
    kwargs = model_kwargs(revision)
    tokenizer = AutoTokenizer.from_pretrained(scorer["model_id"], **kwargs)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            scorer["model_id"], device_map="auto", dtype=torch.bfloat16, **kwargs
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            scorer["model_id"], device_map="auto", torch_dtype=torch.bfloat16, **kwargs
        )
    model.eval()
    if model.dtype != torch.bfloat16:
        raise SystemExit(f"Model loaded as {model.dtype}, expected torch.bfloat16.")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    vocab = tokenizer.get_vocab()
    if "Yes" not in vocab or "No" not in vocab:
        raise SystemExit("ShieldGemma tokenizer does not expose exact Yes and No tokens.")
    signature = inspect.signature(model.forward).parameters
    if "logits_to_keep" in signature:
        logit_argument = "logits_to_keep"
    elif "num_logits_to_keep" in signature:
        logit_argument = "num_logits_to_keep"
    else:
        raise SystemExit(
            "This Transformers/Gemma implementation does not expose a supported last-logit argument. "
            "Do not silently fall back to the original scorer."
        )
    resolved = getattr(model.config, "_commit_hash", None) or revision
    if resolved != revision:
        raise SystemExit(
            f"Model revision mismatch: requested {revision}, resolved {resolved}. Stop rather than mix weights."
        )
    environment = {
        "model_id": scorer["model_id"],
        "resolved_revision": resolved,
        "torch_dtype": str(model.dtype),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "device_name": torch.cuda.get_device_name(0),
        "device_capability": list(capability),
        "padding_side": tokenizer.padding_side,
        "last_logit_argument": logit_argument,
        "last_logit_count": 1,
    }
    return tokenizer, model, torch, vocab["Yes"], vocab["No"], logit_argument, environment


def build_prompt(tokenizer, item: dict[str, Any], rendering: str, policy: dict[str, Any]) -> str:
    if rendering == "prompt_response":
        messages = [
            {"role": "user", "content": item["prompt"]},
            {"role": "assistant", "content": item["response"]},
        ]
        guideline = policy["prompt_response_guideline"]
    elif rendering == "response_only":
        messages = [{"role": "user", "content": item["response"]}]
        guideline = policy["response_only_guideline"]
    else:
        raise ValueError(f"Unknown rendering: {rendering}")
    return tokenizer.apply_chat_template(
        messages,
        guideline=guideline,
        tokenize=False,
        add_generation_prompt=False,
    )


def score_batch(
    *,
    tokenizer,
    model,
    torch,
    yes_id: int,
    no_id: int,
    logit_argument: str,
    tasks: list[dict[str, Any]],
    rendering: str,
    policies: list[dict[str, Any]],
    max_length: int,
) -> tuple[list[dict[str, float]], list[bool], int]:
    """Score a batch and return policy probabilities, truncation, and max length."""
    policy_scores: dict[str, list[float]] = {}
    truncated = [False] * len(tasks)
    observed_max = 0
    for policy in policies:
        prompts = [build_prompt(tokenizer, task, rendering, policy) for task in tasks]
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(model.device)
        inputs = {key: value for key, value in encoded.items() if key in ("input_ids", "attention_mask")}
        with torch.no_grad():
            logits = model(**inputs, **{logit_argument: 1}).logits
        if logits.ndim != 3 or logits.shape[0] != len(tasks) or logits.shape[1] != 1:
            raise RuntimeError(f"Expected (batch, 1, vocab) logits; received {tuple(logits.shape)}.")
        selected = logits[:, -1, [yes_id, no_id]]
        probabilities = torch.softmax(selected.float(), dim=1)[:, 0].tolist()
        policy_scores[policy["id"]] = [float(value) for value in probabilities]
        lengths = encoded["attention_mask"].sum(dim=1).tolist()
        observed_max = max(observed_max, max(int(length) for length in lengths))
        truncated = [
            already or int(length) >= max_length
            for already, length in zip(truncated, lengths)
        ]
    return [
        {policy_id: values[index] for policy_id, values in policy_scores.items()}
        for index in range(len(tasks))
    ], truncated, observed_max


def check_job(meta: dict[str, Any], job_path: Path) -> list[dict[str, Any]]:
    observed = sha256_file(job_path)
    if observed != meta["job_sha256"]:
        raise SystemExit(
            "Job SHA-256 mismatch.\n"
            f"  meta says: {meta['job_sha256']}\n"
            f"  observed : {observed}\n"
            "Regenerate or re-upload the job; never edit it in Colab."
        )
    return read_jsonl(job_path)


def command_score(arguments: argparse.Namespace) -> None:
    job_path = Path(arguments.job)
    meta_path = Path(arguments.meta)
    out_path = Path(arguments.out)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    items = check_job(meta, job_path)
    if arguments.limit:
        items = items[: arguments.limit]
    revision = arguments.revision
    if not revision:
        raise SystemExit("--revision is required: pin the commit resolved by the completed pilot.")

    existing = read_jsonl(out_path)
    done = successful_keys(existing)
    scorer = meta["scorer"]
    renderings = list(scorer["renderings"])
    pending = {
        rendering: [item for item in items if task_key(item["position_key"], rendering) not in done]
        for rendering in renderings
    }
    batch_size = int(arguments.batch_size)
    if batch_size < 1:
        raise SystemExit("--batch-size must be positive.")

    tokenizer, model, torch, yes_id, no_id, logit_argument, environment = load_runtime(meta, revision)
    total = sum(len(tasks) for tasks in pending.values())
    print(
        f"variant={SCORER_VARIANT} positions={len(items)} renderings={len(renderings)} "
        f"pending={total} resumed={len(done)} batch_size={batch_size}",
        flush=True,
    )
    for rendering, tasks in pending.items():
        print(f"  {rendering}: {len(tasks)} pending", flush=True)

    started = time.time()
    new_ok = new_failed = 0
    maximum_observed_input_tokens = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as handle:
        for rendering, tasks_for_rendering in pending.items():
            for start in range(0, len(tasks_for_rendering), batch_size):
                batch = tasks_for_rendering[start : start + batch_size]
                try:
                    scores, truncated, max_tokens = score_batch(
                        tokenizer=tokenizer,
                        model=model,
                        torch=torch,
                        yes_id=yes_id,
                        no_id=no_id,
                        logit_argument=logit_argument,
                        tasks=batch,
                        rendering=rendering,
                        policies=scorer["policies"],
                        max_length=int(scorer["max_input_tokens"]),
                    )
                    maximum_observed_input_tokens = max(maximum_observed_input_tokens, max_tokens)
                    for item, probabilities, was_truncated in zip(batch, scores, truncated):
                        handle.write(json.dumps({
                            "position_key": item["position_key"],
                            "rendering": rendering,
                            "status": "ok",
                            "model_id": scorer["model_id"],
                            "model_revision": revision,
                            "input_truncated": bool(was_truncated),
                            "policy_yes_probability": probabilities,
                            "completed_at_utc": utc_now(),
                        }, sort_keys=True) + "\n")
                    new_ok += len(batch)
                except Exception as error:  # failures remain observable and resumable
                    if new_failed == 0:
                        print("\n=== first scoring failure ===", flush=True)
                        traceback.print_exc()
                        print("=============================\n", flush=True)
                    for item in batch:
                        handle.write(json.dumps({
                            "position_key": item["position_key"],
                            "rendering": rendering,
                            "status": "failed",
                            "error_type": type(error).__name__,
                            "error_message": str(error),
                            "completed_at_utc": utc_now(),
                        }, sort_keys=True) + "\n")
                    new_failed += len(batch)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                handle.flush()
                completed = new_ok + new_failed
                if completed % max(20, batch_size) == 0 or completed == total:
                    elapsed = time.time() - started
                    rate = completed / elapsed if elapsed else 0.0
                    remaining = (total - completed) / rate / 60 if rate else math.inf
                    print(
                        f"  [{rendering}] {completed}/{total} ok={new_ok} failed={new_failed} "
                        f"{rate:.2f}/s eta={remaining:.1f} min",
                        flush=True,
                    )

    manifest = {
        "result_schema": "pku-saferlhf.e5-optimized-scores.v1",
        "scorer_variant": SCORER_VARIANT,
        "job_file": job_path.name,
        "job_sha256": meta["job_sha256"],
        "meta_sha256": sha256_file(meta_path),
        "phase": meta["phase"],
        "requested_revision": revision,
        "renderings": renderings,
        "policy_ids": [policy["id"] for policy in scorer["policies"]],
        "batch_size": batch_size,
        "max_input_tokens": int(scorer["max_input_tokens"]),
        "maximum_observed_input_tokens": maximum_observed_input_tokens,
        "new_ok": new_ok,
        "new_failed": new_failed,
        "resumed_from": len(done),
        "elapsed_seconds": round(time.time() - started, 1),
        "finished_at_utc": utc_now(),
        "platform": platform.platform(),
        "python": sys.version,
        **environment,
    }
    manifest_path = out_path.with_suffix(".manifest.json")
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"\nscores   : {out_path}\nmanifest : {manifest_path}")


def command_compare(arguments: argparse.Namespace) -> None:
    job_path = Path(arguments.job)
    meta_path = Path(arguments.meta)
    baseline_path = Path(arguments.baseline)
    candidate_path = Path(arguments.candidate)
    report_path = Path(arguments.report)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    items = check_job(meta, job_path)
    expected = expected_keys(items, list(meta["scorer"]["renderings"]))
    baseline = terminal_records(read_jsonl(baseline_path))
    candidate = terminal_records(read_jsonl(candidate_path))

    def terminal_ok(records: dict[tuple[str, str], dict[str, Any]]) -> set[tuple[str, str]]:
        return {key for key, record in records.items() if record.get("status") == "ok"}

    baseline_ok = terminal_ok(baseline)
    candidate_ok = terminal_ok(candidate)
    missing_baseline = sorted(expected - baseline_ok)
    missing_candidate = sorted(expected - candidate_ok)
    unexpected_candidate = sorted(candidate_ok - expected)
    policies = [policy["id"] for policy in meta["scorer"]["policies"]]
    deltas: defaultdict[str, list[float]] = defaultdict(list)
    threshold_flips: Counter[str] = Counter()
    truncation_changes = 0
    any_policy_flips = 0
    revisions: set[str] = set()

    for key in sorted(expected & baseline_ok & candidate_ok):
        old = baseline[key]
        new = candidate[key]
        revisions.update({str(old.get("model_revision")), str(new.get("model_revision"))})
        if bool(old.get("input_truncated")) != bool(new.get("input_truncated")):
            truncation_changes += 1
        old_any = new_any = False
        for policy in policies:
            old_score = float(old["policy_yes_probability"][policy])
            new_score = float(new["policy_yes_probability"][policy])
            deltas[policy].append(abs(old_score - new_score))
            old_label = old_score >= POLICY_THRESHOLD
            new_label = new_score >= POLICY_THRESHOLD
            threshold_flips[policy] += int(old_label != new_label)
            old_any = old_any or old_label
            new_any = new_any or new_label
        any_policy_flips += int(old_any != new_any)

    per_policy = {}
    for policy in policies:
        values = sorted(deltas[policy])
        percentile_index = max(0, math.ceil(0.99 * len(values)) - 1) if values else 0
        per_policy[policy] = {
            "n": len(values),
            "mean_absolute_probability_difference": sum(values) / len(values) if values else None,
            "p99_absolute_probability_difference": values[percentile_index] if values else None,
            "maximum_absolute_probability_difference": max(values) if values else None,
            "threshold_label_flips_at_0_5": threshold_flips[policy],
        }
    passed = (
        not missing_baseline
        and not missing_candidate
        and not unexpected_candidate
        and truncation_changes == 0
        and any_policy_flips == 0
        and all(per_policy[policy]["threshold_label_flips_at_0_5"] == 0 for policy in policies)
    )
    report = {
        "result_schema": "pku-saferlhf.e5-leftpad-equivalence.v1",
        "scorer_variant": SCORER_VARIANT,
        "created_at_utc": utc_now(),
        "job_sha256": meta["job_sha256"],
        "baseline_scores_sha256": sha256_file(baseline_path),
        "candidate_scores_sha256": sha256_file(candidate_path),
        "expected_position_renderings": len(expected),
        "baseline_terminal_ok": len(baseline_ok),
        "candidate_terminal_ok": len(candidate_ok),
        "missing_baseline": len(missing_baseline),
        "missing_candidate": len(missing_candidate),
        "unexpected_candidate": len(unexpected_candidate),
        "input_truncation_changes": truncation_changes,
        "any_policy_threshold_flips_at_0_5": any_policy_flips,
        "model_revisions_observed": sorted(revisions),
        "policy_results": per_policy,
        "passed": passed,
        "passing_rule": (
            "Every expected position/rendering must have a terminal OK record; "
            "input-truncation states and all 0.5 policy labels must match. "
            "Probability differences are reported but not treated as exact-equality requirements."
        ),
    }
    write_json(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit("Equivalence check failed. Do not run the optimized primary scorer.")


def command_probe(arguments: argparse.Namespace) -> None:
    job_path = Path(arguments.job)
    meta_path = Path(arguments.meta)
    report_path = Path(arguments.report)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    items = check_job(meta, job_path)
    batch_size = int(arguments.batch_size)
    if batch_size < 1:
        raise SystemExit("--batch-size must be positive.")
    # A deterministic character-length ranking puts likely long prompts into the
    # same batch. Tokenisation below reports the actual token maximum.
    longest = sorted(items, key=lambda item: len(item["prompt"]) + len(item["response"]), reverse=True)[:batch_size]
    tokenizer, model, torch, yes_id, no_id, logit_argument, environment = load_runtime(meta, arguments.revision)
    scorer = meta["scorer"]
    report: dict[str, Any] = {
        "result_schema": "pku-saferlhf.e5-leftpad-batch-probe.v1",
        "scorer_variant": SCORER_VARIANT,
        "job_sha256": meta["job_sha256"],
        "batch_size": batch_size,
        "selected_positions": len(longest),
        "created_at_utc": utc_now(),
        **environment,
    }
    try:
        torch.cuda.reset_peak_memory_stats()
        max_tokens = 0
        for rendering in scorer["renderings"]:
            _, _, observed = score_batch(
                tokenizer=tokenizer,
                model=model,
                torch=torch,
                yes_id=yes_id,
                no_id=no_id,
                logit_argument=logit_argument,
                tasks=longest,
                rendering=rendering,
                policies=scorer["policies"],
                max_length=int(scorer["max_input_tokens"]),
            )
            max_tokens = max(max_tokens, observed)
        report.update({
            "status": "ok",
            "maximum_observed_input_tokens": max_tokens,
            "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        })
    except Exception as error:  # OOM is a probe result, not a silent fallback
        report.update({"status": "failed", "error_type": type(error).__name__, "error_message": str(error)})
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    write_json(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "ok":
        raise SystemExit("Batch probe failed; use a smaller batch size.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    score = subcommands.add_parser("score", help="Score a job with left padding and last-token logits.")
    score.add_argument("--job", required=True)
    score.add_argument("--meta", required=True)
    score.add_argument("--out", required=True)
    score.add_argument("--revision", required=True)
    score.add_argument("--batch-size", type=int, required=True)
    score.add_argument("--limit", type=int)
    score.set_defaults(handler=command_score)

    compare = subcommands.add_parser("compare", help="Compare optimized scores with the completed right-padding pilot.")
    compare.add_argument("--job", required=True)
    compare.add_argument("--meta", required=True)
    compare.add_argument("--baseline", required=True)
    compare.add_argument("--candidate", required=True)
    compare.add_argument("--report", required=True)
    compare.set_defaults(handler=command_compare)

    probe = subcommands.add_parser("probe", help="Test one primary batch on deterministically selected long inputs.")
    probe.add_argument("--job", required=True)
    probe.add_argument("--meta", required=True)
    probe.add_argument("--revision", required=True)
    probe.add_argument("--batch-size", type=int, required=True)
    probe.add_argument("--report", required=True)
    probe.set_defaults(handler=command_probe)

    arguments = parser.parse_args()
    arguments.handler(arguments)


if __name__ == "__main__":
    main()
