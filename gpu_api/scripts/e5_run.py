#!/usr/bin/env python3
"""Run E5 external safety operationalisations without persisting text.

Examples, from the experiments directory:

  python3 gpu_api/scripts/e5_run.py --site deepseek --phase pilot --run-id e5-pilot
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
import concurrent.futures
import json
import math
import os
import random
import sys
import threading
import time
from pathlib import Path

from common import (
    CONFIG_DIR,
    ProtocolError,
    acquire_run_lock,
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
THREAD_LOCAL = threading.local()
DEEPSEEK_PARSER_POLICY = "deepseek_json_object_safety_v1"


class UnparseableModelOutput(ValueError):
    """The model returned content outside the frozen E5 label interface."""


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


def parse_deepseek_safety_label(content: str | None) -> str | None:
    """Accept only the frozen two-label JSON response contract."""

    if content is None:
        return None
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or set(payload) != {"label"}:
        return None
    label = payload["label"]
    return label if label in {"safe", "unsafe"} else None


def deepseek_client():
    if getattr(THREAD_LOCAL, "deepseek_client", None) is None:
        try:
            from openai import OpenAI
        except ImportError as error:
            raise ProtocolError("Install the openai package before running E5 DeepSeek.") from error
        THREAD_LOCAL.deepseek_client = OpenAI()
    return THREAD_LOCAL.deepseek_client


def deepseek_prompt(task: dict, rendering: str, config: dict) -> str:
    return config["deepseek_judge"]["template"].format(
        rendered_input=render_input(task, rendering, config)
    )


def deepseek_call(task: dict, rendering: str, config: dict) -> dict:
    judge = config["deepseek_judge"]
    attempts = int(config["runtime"]["max_retries"])
    delay = float(config["runtime"]["retry_base_seconds"])
    last_error: Exception | None = None
    last_output_diagnostic: dict[str, int | str | None] | None = None
    for attempt in range(attempts):
        try:
            result = deepseek_client().chat.completions.create(
                model=judge["model"],
                temperature=judge["temperature"],
                extra_body={"thinking": judge["thinking"]},
                max_tokens=int(config["runtime"]["deepseek_max_tokens"]),
                response_format=judge["response_format"],
                messages=[
                    {"role": "system", "content": judge["system_instruction"]},
                    {"role": "user", "content": deepseek_prompt(task, rendering, config)},
                ],
            )
            choice = result.choices[0]
            content = choice.message.content
            output_diagnostic = {
                "finish_reason": getattr(choice, "finish_reason", None),
                "completion_characters": len(content or ""),
                "parser_policy": DEEPSEEK_PARSER_POLICY,
            }
            label = parse_deepseek_safety_label(content)
            if label is None:
                last_output_diagnostic = output_diagnostic
                raise UnparseableModelOutput()
            usage = getattr(result, "usage", None)
            return {
                "status": "ok",
                "returned_model": getattr(result, "model", judge["model"]),
                "api_request_id": getattr(result, "id", None),
                "external_unsafe": label == config["decision_rule"]["deepseek_label_unsafe"],
                "deepseek_safety_label": label,
                "attempts_made": attempt + 1,
                **output_diagnostic,
                "usage": {
                    "prompt_tokens": getattr(usage, "prompt_tokens", None),
                    "completion_tokens": getattr(usage, "completion_tokens", None),
                    "total_tokens": getattr(usage, "total_tokens", None),
                },
            }
        except Exception as error:
            last_error = error
            if attempt + 1 < attempts:
                time.sleep(delay * (2**attempt) + random.random())
    assert last_error is not None
    outcome = {
        "status": "failed",
        "error_type": type(last_error).__name__,
        "attempts_made": attempts,
        "parser_policy": DEEPSEEK_PARSER_POLICY,
    }
    if isinstance(last_error, UnparseableModelOutput) and last_output_diagnostic is not None:
        outcome.update({f"last_{key}": value for key, value in last_output_diagnostic.items()})
    return outcome


def run_deepseek(
    tasks: list[dict],
    rendering: str,
    output_path: Path,
    metadata: dict,
    config: dict,
    dry_run: bool,
    workers: int,
) -> tuple[int, int]:
    site = next(site for site in config["sites"] if site["id"] == "deepseek_v4_flash_safety_judge")
    completed = existing_ids(output_path)
    todo = [
        task
        for task in tasks
        if task_id(site["id"], rendering, task) not in completed
    ]
    if dry_run:
        return len(todo), 0
    if not environment_present(site["credential_environment"]):
        raise ProtocolError("OPENAI_API_KEY is required for the E5 DeepSeek site.")
    metadata.update(
        {
            "deepseek_configured_model": site["model_id"],
            "deepseek_parser_policy": DEEPSEEK_PARSER_POLICY,
            "deepseek_temperature": config["deepseek_judge"]["temperature"],
            "deepseek_thinking": config["deepseek_judge"]["thinking"],
            "deepseek_max_tokens": config["runtime"]["deepseek_max_tokens"],
            "deepseek_workers": workers,
        }
    )
    ok = failed = 0
    returned_models: set[str] = set()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(deepseek_call, task, rendering, config): task for task in todo
        }
        for future in concurrent.futures.as_completed(futures):
            task = futures[future]
            try:
                outcome = future.result()
            except Exception as error:
                outcome = {"status": "failed", "error_type": type(error).__name__}
            record = {
                "record_schema": "pku-saferlhf.e5-position.v1",
                "request_id": task_id(site["id"], rendering, task),
                "status": outcome["status"],
                "site": site["id"],
                "model_id": outcome.get("returned_model", site["model_id"]),
                "rendering": rendering,
                "completed_at_utc": utc_now(),
                "input_truncation_status": "not_locally_measured",
                **public_task_fields(task),
                **outcome,
            }
            append_jsonl(output_path, record)
            if outcome["status"] == "ok":
                ok += 1
                returned_models.add(str(outcome["returned_model"]))
            else:
                failed += 1
    metadata["deepseek_returned_models"] = sorted(returned_models)
    return ok, failed


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
    # transformers 5.x renamed `torch_dtype` to `dtype` and only warns about
    # the old name; a warned-and-ignored dtype silently loads float32, which
    # does not fit a 24 GB device once batched logits are allocated.
    try:
        model = AutoModelForCausalLM.from_pretrained(
            site["model_id"], device_map="auto", dtype=torch.bfloat16, **kwargs
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            site["model_id"], device_map="auto", torch_dtype=torch.bfloat16, **kwargs
        )
    if model.dtype != torch.bfloat16:
        raise ProtocolError(f"Model loaded as {model.dtype}, not bfloat16.")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # The final-token index used below is attention_mask.sum(dim=1) - 1, which
    # is the last real token only under right padding.  Pin the side rather
    # than inherit whatever the checkpoint defaults to.
    tokenizer.padding_side = "right"
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
                # Two steps deliberately.  Indexing all three axes at once
                # broadcasts (B,), (B,) and (2,) together, which raises for
                # B > 2 and -- worse -- silently returns one logit per row
                # when B == 2.
                final_logits = logits[
                    torch.arange(logits.shape[0], device=logits.device), last_index
                ]  # (B, vocab)
                selected = final_logits[:, [yes_id, no_id]]  # (B, 2)
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
            completed_batches = start // batch_size + 1
            total_batches = math.ceil(len(todo) / batch_size)
            if completed_batches % 25 == 0 or completed_batches == total_batches:
                print(
                    f"ShieldGemma {rendering}: "
                    f"{min(start + len(batch), len(todo))}/{len(todo)} response positions"
                )
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
    parser.add_argument(
        "--site",
        choices=("deepseek", "openai", "shieldgemma", "all"),
        default="deepseek",
    )
    parser.add_argument("--phase", choices=("pilot", "primary"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--rendering",
        choices=("prompt_response", "response_only", "both"),
        default="both",
    )
    parser.add_argument("--shieldgemma-revision")
    parser.add_argument("--workers", type=int)
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
    workers = arguments.workers or int(config["runtime"]["deepseek_workers"])
    rows = selected_rows(arguments.phase, config)
    tasks = response_tasks(rows)
    renderings = (
        ["prompt_response", "response_only"]
        if arguments.rendering == "both"
        else [arguments.rendering]
    )
    run_dir = run_directory("e5", arguments.run_id)
    # The live file lock prevents concurrent jobs from interleaving records for
    # one run ID. It is released automatically on normal exit or interruption.
    run_lock = acquire_run_lock(run_dir)
    manifest_path = run_dir / "run_manifest.json"
    metadata = base_run_manifest("E5", CONFIG_PATH, arguments.run_id, SCRIPT)
    metadata.update(
        {
            "phase": arguments.phase,
            "sites_requested": arguments.site,
            "renderings": renderings,
            "pair_rows": len(rows),
            "response_positions": len(tasks),
            "workers": workers,
            "sampling_manifest": (
                PRIMARY_SAMPLE_PATH.name if arguments.phase == "primary" else None
            ),
            "dry_run": arguments.dry_run,
        }
    )
    write_json(manifest_path, metadata)
    summary: dict[str, dict] = {}
    requested_sites = (
        ["deepseek", "openai", "shieldgemma"]
        if arguments.site == "all"
        else [arguments.site]
    )
    for site_name in requested_sites:
        site_id = {
            "deepseek": "deepseek_v4_flash_safety_judge",
            "openai": "openai_omni_moderation_latest",
            "shieldgemma": "shieldgemma_9b",
        }[site_name]
        for rendering in renderings:
            output_path = run_dir / f"{site_id}_{rendering}.jsonl"
            if site_name == "deepseek":
                planned, failed = run_deepseek(
                    tasks,
                    rendering,
                    output_path,
                    metadata,
                    config,
                    arguments.dry_run,
                    workers,
                )
            elif site_name == "openai":
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
