#!/usr/bin/env python3
"""Run E5 external safety operationalisations without persisting text.

Examples, from the experiments directory:

  python3 gpu_api/scripts/e5_run.py --site openai --phase pilot --run-id e5-pilot
  python3 gpu_api/scripts/e5_run.py --site shieldgemma --phase pilot --run-id e5-pilot
  python3 gpu_api/scripts/e5_run.py --site all --phase primary --run-id e5-primary \
      --shieldgemma-revision RESOLVED_HF_SHA

The primary ShieldGemma run deliberately requires a resolved revision.  The
pilot records the revision visible to the loader; copy that SHA into the
primary command and the run manifest before starting the frozen sample.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

from common import (
    CONFIG_DIR,
    ProtocolError,
    append_jsonl,
    base_run_manifest,
    deterministic_pilot_rows,
    environment_present,
    existing_ids,
    public_task_fields,
    read_json,
    primary_sample_rows,
    response_tasks,
    run_directory,
    utc_now,
    validate_cpu_provenance,
    write_json,
)


SCRIPT = Path(__file__).resolve()
CONFIG_PATH = CONFIG_DIR / "e5_external_boundary.json"
PRIMARY_SAMPLE_PATH = CONFIG_DIR / "primary_pair_sample_manifest.csv"


def mapping(value: object) -> dict:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, dict):
        return value
    return dict(value.__dict__)


def task_id(site: str, rendering: str, task: dict) -> str:
    return ":".join(
        (
            site,
            rendering,
            task["source_file"],
            str(task["source_line"]),
            str(task["response_position"]),
        )
    )


def render_input(task: dict, rendering: str, config: dict) -> str:
    template_key = "primary" if rendering == "prompt_response" else "sensitivity"
    return config["input_renderings"][template_key].format(
        prompt=task["prompt"], response=task["response"]
    )


def selected_rows(phase: str, config: dict) -> list[tuple]:
    if phase == "pilot":
        return deterministic_pilot_rows(
            config["pilot"]["pairs"], config["pilot"]["selection_seed"]
        )
    if not PRIMARY_SAMPLE_PATH.exists():
        raise ProtocolError(
            "Create the frozen shared primary sample with make_primary_sample.py before E5."
        )
    return primary_sample_rows(PRIMARY_SAMPLE_PATH)


def retry(operation, attempts: int, base_seconds: float):
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return operation()
        except Exception as error:  # API and CUDA exceptions are recorded without text.
            last_error = error
            if attempt + 1 < attempts:
                time.sleep(base_seconds * (2**attempt) + random.random())
    assert last_error is not None
    raise last_error


def run_openai(
    tasks: list[dict],
    rendering: str,
    output_path: Path,
    metadata: dict,
    config: dict,
    dry_run: bool,
) -> tuple[int, int]:
    completed = existing_ids(output_path)
    todo = [
        task
        for task in tasks
        if task_id("openai_omni_moderation_latest", rendering, task) not in completed
    ]
    if dry_run:
        return len(todo), 0
    if not environment_present("OPENAI_API_KEY"):
        raise ProtocolError("OPENAI_API_KEY is required for the E5 OpenAI site.")
    try:
        from openai import OpenAI
    except ImportError as error:
        raise ProtocolError("Install the openai package before running E5 OpenAI.") from error

    site = next(site for site in config["sites"] if site["id"] == "openai_omni_moderation_latest")
    client = OpenAI()
    batch_size = int(config["runtime"]["openai_batch_size"])
    attempts = int(config["runtime"]["max_retries"])
    backoff = float(config["runtime"]["retry_base_seconds"])
    ok = failed = 0
    for start in range(0, len(todo), batch_size):
        batch = todo[start : start + batch_size]
        texts = [render_input(task, rendering, config) for task in batch]
        try:
            response = retry(
                lambda: client.moderations.create(model=site["model_id"], input=texts),
                attempts,
                backoff,
            )
            results = list(response.results)
            if len(results) != len(batch):
                raise ProtocolError("OpenAI moderation result count does not match input count.")
            model_id = getattr(response, "model", site["model_id"])
            for task, result in zip(batch, results):
                result_data = mapping(result)
                record = {
                    "record_schema": "pku-saferlhf.e5-position.v1",
                    "request_id": task_id(site["id"], rendering, task),
                    "status": "ok",
                    "site": site["id"],
                    "model_id": model_id,
                    "rendering": rendering,
                    "completed_at_utc": utc_now(),
                    **public_task_fields(task),
                    "external_unsafe": bool(result_data["flagged"]),
                    "category_flags": mapping(result_data["categories"]),
                    "category_scores": mapping(result_data["category_scores"]),
                    "input_may_be_truncated": False,
                }
                append_jsonl(output_path, record)
                ok += 1
        except Exception as error:
            for task in batch:
                append_jsonl(
                    output_path,
                    {
                        "record_schema": "pku-saferlhf.e5-position.v1",
                        "request_id": task_id(site["id"], rendering, task),
                        "status": "failed",
                        "site": site["id"],
                        "rendering": rendering,
                        "completed_at_utc": utc_now(),
                        **public_task_fields(task),
                        "error_type": type(error).__name__,
                    },
                )
                failed += 1
    metadata["openai_model_identifier"] = site["model_id"]
    return ok, failed


def load_shieldgemma(config: dict, revision: str | None):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise ProtocolError(
            "Install torch, transformers, and accelerate before running ShieldGemma."
        ) from error
    if not environment_present("HF_TOKEN"):
        raise ProtocolError("HF_TOKEN is required after accepting ShieldGemma's licence.")
    site = next(site for site in config["sites"] if site["id"] == "shieldgemma_9b")
    kwargs = {"token": os.environ["HF_TOKEN"]}
    if revision:
        kwargs["revision"] = revision
    tokenizer = AutoTokenizer.from_pretrained(site["model_id"], **kwargs)
    model = AutoModelForCausalLM.from_pretrained(
        site["model_id"],
        device_map="auto",
        torch_dtype=torch.bfloat16,
        **kwargs,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    vocab = tokenizer.get_vocab()
    if "Yes" not in vocab or "No" not in vocab:
        raise ProtocolError("ShieldGemma tokenizer does not expose exact Yes and No tokens.")
    resolved = getattr(model.config, "_commit_hash", None) or revision or "unresolved"
    return site, tokenizer, model, torch, vocab["Yes"], vocab["No"], resolved


def shield_prompt(tokenizer, task: dict, rendering: str, policy: dict) -> str:
    if rendering == "prompt_response":
        messages = [
            {"role": "user", "content": task["prompt"]},
            {"role": "assistant", "content": task["response"]},
        ]
        guideline = policy["prompt_response_guideline"]
    else:
        messages = [{"role": "user", "content": task["response"]}]
        guideline = policy["response_only_guideline"]
    return tokenizer.apply_chat_template(
        messages,
        guideline=guideline,
        tokenize=False,
        add_generation_prompt=False,
    )


def run_shieldgemma(
    tasks: list[dict],
    rendering: str,
    output_path: Path,
    metadata: dict,
    config: dict,
    phase: str,
    revision: str | None,
    dry_run: bool,
) -> tuple[int, int]:
    completed = existing_ids(output_path)
    todo = [
        task
        for task in tasks
        if task_id("shieldgemma_9b", rendering, task) not in completed
    ]
    if dry_run:
        return len(todo), 0
    if phase == "primary" and not revision:
        raise ProtocolError(
            "A primary ShieldGemma run requires --shieldgemma-revision from the completed pilot."
        )
    site, tokenizer, model, torch, yes_id, no_id, resolved = load_shieldgemma(config, revision)
    metadata["shieldgemma_resolved_revision"] = resolved
    metadata["shieldgemma_torch_dtype"] = str(getattr(model, "dtype", "unknown"))
    batch_size = int(config["runtime"]["shieldgemma_batch_size"])
    max_length = int(config["runtime"]["max_input_tokens"])
    threshold = float(config["decision_rule"]["shieldgemma_yes_probability_threshold"])
    policies = site["policies"]
    ok = failed = 0
    for start in range(0, len(todo), batch_size):
        batch = todo[start : start + batch_size]
        policy_scores: dict[str, list[float]] = {}
        truncated = [False] * len(batch)
        try:
            for policy in policies:
                prompts = [shield_prompt(tokenizer, task, rendering, policy) for task in batch]
                encoded = tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                ).to(model.device)
                with torch.no_grad():
                    logits = model(**encoded).logits
                last_index = encoded["attention_mask"].sum(dim=1) - 1
                selected = logits[
                    torch.arange(logits.shape[0], device=logits.device),
                    last_index,
                    [yes_id, no_id],
                ]
                probabilities = torch.softmax(selected.float(), dim=1)[:, 0].tolist()
                policy_scores[policy["id"]] = [float(value) for value in probabilities]
                lengths = encoded["attention_mask"].sum(dim=1).tolist()
                truncated = [
                    current or int(length) >= max_length
                    for current, length in zip(truncated, lengths)
                ]
            for index, task in enumerate(batch):
                scores = {policy_id: values[index] for policy_id, values in policy_scores.items()}
                flags = {policy_id: value >= threshold for policy_id, value in scores.items()}
                append_jsonl(
                    output_path,
                    {
                        "record_schema": "pku-saferlhf.e5-position.v1",
                        "request_id": task_id(site["id"], rendering, task),
                        "status": "ok",
                        "site": site["id"],
                        "model_id": site["model_id"],
                        "model_revision": resolved,
                        "rendering": rendering,
                        "completed_at_utc": utc_now(),
                        **public_task_fields(task),
                        "external_unsafe": any(flags.values()),
                        "policy_flags": flags,
                        "policy_yes_probabilities": scores,
                        "input_may_be_truncated": truncated[index],
                    },
                )
                ok += 1
        except Exception as error:
            for task in batch:
                append_jsonl(
                    output_path,
                    {
                        "record_schema": "pku-saferlhf.e5-position.v1",
                        "request_id": task_id(site["id"], rendering, task),
                        "status": "failed",
                        "site": site["id"],
                        "rendering": rendering,
                        "completed_at_utc": utc_now(),
                        **public_task_fields(task),
                        "error_type": type(error).__name__,
                    },
                )
                failed += 1
    return ok, failed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", choices=("openai", "shieldgemma", "all"), default="all")
    parser.add_argument("--phase", choices=("pilot", "primary"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--rendering",
        choices=("prompt_response", "response_only", "both"),
        default="both",
    )
    parser.add_argument("--shieldgemma-revision")
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()

    validate_cpu_provenance()
    if (
        arguments.phase == "primary"
        and arguments.site in {"shieldgemma", "all"}
        and not arguments.shieldgemma_revision
    ):
        raise ProtocolError(
            "A primary ShieldGemma request requires --shieldgemma-revision from the completed pilot."
        )
    config = read_json(CONFIG_PATH)
    rows = selected_rows(arguments.phase, config)
    tasks = response_tasks(rows)
    renderings = (
        ["prompt_response", "response_only"]
        if arguments.rendering == "both"
        else [arguments.rendering]
    )
    run_dir = run_directory("e5", arguments.run_id)
    manifest_path = run_dir / "run_manifest.json"
    metadata = base_run_manifest("E5", CONFIG_PATH, arguments.run_id, SCRIPT)
    metadata.update(
        {
            "phase": arguments.phase,
            "sites_requested": arguments.site,
            "renderings": renderings,
            "pair_rows": len(rows),
            "response_positions": len(tasks),
            "sampling_manifest": (
                PRIMARY_SAMPLE_PATH.name if arguments.phase == "primary" else None
            ),
            "dry_run": arguments.dry_run,
        }
    )
    write_json(manifest_path, metadata)
    summary: dict[str, dict] = {}
    requested_sites = (
        ["openai", "shieldgemma"] if arguments.site == "all" else [arguments.site]
    )
    for site_name in requested_sites:
        site_id = (
            "openai_omni_moderation_latest"
            if site_name == "openai"
            else "shieldgemma_9b"
        )
        for rendering in renderings:
            output_path = run_dir / f"{site_id}_{rendering}.jsonl"
            if site_name == "openai":
                planned, failed = run_openai(
                    tasks, rendering, output_path, metadata, config, arguments.dry_run
                )
            else:
                planned, failed = run_shieldgemma(
                    tasks,
                    rendering,
                    output_path,
                    metadata,
                    config,
                    arguments.phase,
                    arguments.shieldgemma_revision,
                    arguments.dry_run,
                )
            summary[f"{site_id}:{rendering}"] = {
                "new_ok_or_planned": planned,
                "new_failed": failed,
                "output_file": output_path.name,
            }
            write_json(manifest_path, metadata)
    summary["run_directory"] = str(run_dir)
    summary["dry_run"] = arguments.dry_run
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
