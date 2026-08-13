"""Standalone ShieldGemma scorer -- phase 2 of the three-stage GPU workflow.

This file imports nothing from the experiment repository and makes no
assumption about directory layout.  Copy it, the job file, and the job meta
file to the GPU machine; nothing else is required.

    python score_shieldgemma_standalone.py \
        --job e5-pilot.jsonl \
        --meta e5-pilot.meta.json \
        --out e5-pilot.scores.jsonl

It is resumable: rerunning skips every (position_key, rendering) already
present in the output file, so an interrupted Colab session costs only the
work in flight.

The scorer records raw P(Yes) probabilities per policy.  It never applies the
decision threshold -- that stays in the repository, at ingest.  A failure is
written as a failure record and is never recoded as safe.

Renderings and policy guidelines are read from the meta file, so this script
cannot silently diverge from the frozen protocol: if the meta changes, the job
hash changes, and ingest refuses the merge.
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
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def completed_keys(path: Path) -> set[tuple[str, str]]:
    return {
        (record["position_key"], record["rendering"])
        for record in read_jsonl(path)
        if record.get("status") == "ok"
    }


def build_prompt(tokenizer, item: dict, rendering: str, policy: dict) -> str:
    """Reproduces gpu_api/scripts/e5_run.py::shield_prompt exactly."""
    if rendering == "prompt_response":
        messages = [
            {"role": "user", "content": item["prompt"]},
            {"role": "assistant", "content": item["response"]},
        ]
        guideline = policy["prompt_response_guideline"]
    else:
        messages = [{"role": "user", "content": item["response"]}]
        guideline = policy["response_only_guideline"]
    return tokenizer.apply_chat_template(
        messages,
        guideline=guideline,
        tokenize=False,
        add_generation_prompt=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True)
    parser.add_argument("--meta", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--revision", default=None, help="Pin a resolved HF commit.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    arguments = parser.parse_args()

    job_path = Path(arguments.job)
    meta_path = Path(arguments.meta)
    out_path = Path(arguments.out)
    manifest_path = out_path.with_suffix(".manifest.json")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    scorer = meta["scorer"]

    observed = sha256_file(job_path)
    if observed != meta["job_sha256"]:
        raise SystemExit(
            f"Job file hash mismatch.\n  meta says : {meta['job_sha256']}\n"
            f"  observed  : {observed}\nRegenerate the job rather than editing it."
        )

    # Credential resolution.  HF_TOKEN takes precedence inside huggingface_hub,
    # so an environment token belonging to an account without access silently
    # displaces a working `login()` session.  Resolve explicitly, report which
    # account is in use, and fail before loading if it cannot reach the repo.
    from huggingface_hub import HfApi, get_token, whoami

    token = os.environ.get("HF_TOKEN") or get_token()
    source = "HF_TOKEN" if os.environ.get("HF_TOKEN") else "stored login"
    if not token:
        raise SystemExit(
            "No Hugging Face credential found. Either run huggingface_hub.login() "
            "or set HF_TOKEN, using an account granted access to ShieldGemma."
        )

    try:
        account = whoami(token=token)["name"]
    except Exception as error:  # noqa: BLE001
        raise SystemExit(f"Credential from {source} is not valid: {error}") from error

    model_id_probe = scorer["model_id"]
    try:
        HfApi().model_info(model_id_probe, token=token, files_metadata=False)
        HfApi().hf_hub_download  # noqa: B018 - attribute presence check only
    except Exception as error:  # noqa: BLE001
        raise SystemExit(
            f"Account '{account}' (from {source}) cannot reach {model_id_probe}: {error}"
        ) from error

    print(f"credential: {source} -> account '{account}'", flush=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    kwargs = {"token": token}
    if arguments.revision:
        kwargs["revision"] = arguments.revision

    model_id = scorer["model_id"]
    print(f"loading {model_id} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, **kwargs)
    # transformers 5.x renamed `torch_dtype` to `dtype` and only warns about the
    # old name.  A warned-and-ignored dtype loads 9B in float32 (~36 GB), which
    # does not fit a 24 GB device: accelerate then offloads layers and the
    # forward pass fails on device mismatch.  Prefer the new name, fall back for
    # older releases, and assert the dtype that actually took effect.
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="auto", dtype=torch.bfloat16, **kwargs
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="auto", torch_dtype=torch.bfloat16, **kwargs
        )
    model.eval()
    if model.dtype != torch.bfloat16:
        raise SystemExit(
            f"Model loaded as {model.dtype}, not bfloat16. A float32 9B model "
            "does not fit a 24 GB device and will fail during the forward pass."
        )
    print(f"dtype: {model.dtype}  device: {model.device}", flush=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # The final-token index below is attention_mask.sum(dim=1) - 1, which is the
    # last real token only under right padding.  Gemma tokenizers frequently
    # default to left padding, where that index lands mid-sequence and yields a
    # silently wrong score.  Pin the side rather than infer it.
    tokenizer.padding_side = "right"

    vocab = tokenizer.get_vocab()
    if "Yes" not in vocab or "No" not in vocab:
        raise SystemExit("ShieldGemma tokenizer does not expose exact Yes and No tokens.")
    yes_id, no_id = vocab["Yes"], vocab["No"]
    resolved = getattr(model.config, "_commit_hash", None) or arguments.revision or "unresolved"

    items = list(read_jsonl(job_path))
    if arguments.limit:
        items = items[: arguments.limit]
    done = completed_keys(out_path)
    policies = scorer["policies"]
    renderings = scorer["renderings"]
    max_length = int(scorer["max_input_tokens"])
    batch_size = int(arguments.batch_size or scorer["batch_size"])

    # Group by rendering rather than flattening.  A flat list batched by slice
    # can straddle a rendering boundary whenever the pending counts per
    # rendering are not multiples of the batch size -- which is exactly what a
    # resumed run produces -- and the straddling entries would be dropped
    # without ever being scored.
    pending: dict[str, list[dict]] = {
        rendering: [
            item for item in items if (item["position_key"], rendering) not in done
        ]
        for rendering in renderings
    }
    total = sum(len(v) for v in pending.values())
    print(
        f"positions={len(items)} renderings={len(renderings)} "
        f"policies={len(policies)} pending={total} resumed={len(done)}",
        flush=True,
    )
    for rendering, queue in pending.items():
        print(f"  {rendering}: {len(queue)} pending", flush=True)

    started = time.time()
    ok = failed = 0
    batches = 0
    handle = out_path.open("a", encoding="utf-8")
    try:
        for rendering, queue in pending.items():
          for start in range(0, len(queue), batch_size):
            tasks = queue[start : start + batch_size]
            batches += 1
            policy_scores: dict[str, list[float]] = {}
            truncated = [False] * len(tasks)
            try:
                for policy in policies:
                    prompts = [build_prompt(tokenizer, t, rendering, policy) for t in tasks]
                    encoded = tokenizer(
                        prompts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=max_length,
                    ).to(model.device)
                    # Some tokenizers emit token_type_ids that Gemma's forward
                    # does not accept; passing them raises a TypeError that
                    # would otherwise be recorded as a scoring failure.
                    inputs = {
                        k: v
                        for k, v in encoded.items()
                        if k in ("input_ids", "attention_mask")
                    }
                    with torch.no_grad():
                        logits = model(**inputs).logits
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
                    policy_scores[policy["id"]] = [float(v) for v in probabilities]
                    lengths = encoded["attention_mask"].sum(dim=1).tolist()
                    truncated = [
                        current or int(length) >= max_length
                        for current, length in zip(truncated, lengths)
                    ]
                for index, task in enumerate(tasks):
                    handle.write(
                        json.dumps(
                            {
                                "position_key": task["position_key"],
                                "rendering": rendering,
                                "status": "ok",
                                "model_id": model_id,
                                "model_revision": resolved,
                                "input_truncated": bool(truncated[index]),
                                "policy_yes_probability": {
                                    pid: values[index] for pid, values in policy_scores.items()
                                },
                                "completed_at_utc": utc_now(),
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
                ok += len(tasks)
            except Exception as error:  # noqa: BLE001 - failures must survive as records
                detail = f"{type(error).__name__}: {error}"
                if failed == 0:
                    # Print the first failure in full.  A silent failure record
                    # is reproducible but useless while the run is in progress.
                    print("\n=== first batch failure ===", flush=True)
                    traceback.print_exc()
                    print("===========================\n", flush=True)
                for task in tasks:
                    handle.write(
                        json.dumps(
                            {
                                "position_key": task["position_key"],
                                "rendering": rendering,
                                "status": "failed",
                                "error_type": type(error).__name__,
                                "error_message": str(error)[:400],
                                "completed_at_utc": utc_now(),
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
                failed += len(tasks)
            handle.flush()
            if batches % 20 == 1:
                elapsed = time.time() - started
                seen = ok + failed
                rate = seen / elapsed if elapsed else 0.0
                remaining = (total - seen) / rate if rate else float("nan")
                print(
                    f"  [{rendering}] {seen}/{total}  ok={ok} failed={failed}  "
                    f"{rate:.1f}/s  eta={remaining/60:.1f} min",
                    flush=True,
                )
    finally:
        handle.close()

    manifest = {
        "result_schema": "pku-saferlhf.e5-standalone-scores.v1",
        "job_file": job_path.name,
        "job_sha256": observed,
        "meta_sha256": sha256_file(meta_path),
        "phase": meta["phase"],
        "model_id": model_id,
        "credential_source": source,
        "huggingface_account": account,
        "requested_revision": arguments.revision,
        "resolved_revision": resolved,
        "torch_dtype": str(getattr(model, "dtype", "unknown")),
        "batch_size": batch_size,
        "max_input_tokens": max_length,
        "renderings": renderings,
        "policy_ids": [p["id"] for p in policies],
        "new_ok": ok,
        "new_failed": failed,
        "resumed_from": len(done),
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "device_capability": (
            list(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None
        ),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "python": sys.version,
        "platform": platform.platform(),
        "finished_at_utc": utc_now(),
        "elapsed_seconds": round(time.time() - started, 1),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"\nscores   : {out_path}")
    print(f"manifest : {manifest_path}")


if __name__ == "__main__":
    main()
