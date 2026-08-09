#!/usr/bin/env python3
"""Shared, text-safe infrastructure for E5, E6, and E7.

The raw PKU release is read only from the workspace data directory.  Every
persisted record carries hashes and source coordinates, never prompt or
response text.  Private run directories are ignored by Git.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


SCRIPT = Path(__file__).resolve()
EXPERIMENTS = SCRIPT.parents[2]
WORKSPACE = EXPERIMENTS.parent
RAW_DUAL = WORKSPACE / "data" / "raw" / "dual"
CONFIG_DIR = EXPERIMENTS / "gpu_api" / "config"
CPU_RESULTS = EXPERIMENTS / "cpu" / "results"
PRIVATE_ROOT = EXPERIMENTS / "gpu_api" / "private_runs"


class ProtocolError(RuntimeError):
    """Raised when a run would violate the frozen protocol."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def script_hash(path: Path) -> str:
    return sha256(path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> Iterable[dict]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def existing_ids(path: Path, key: str = "request_id") -> set[str]:
    return {
        str(record[key])
        for record in read_jsonl(path)
        if key in record and record.get("status") == "ok"
    }


def validate_cpu_provenance() -> tuple[dict, dict]:
    p0_path = CPU_RESULTS / "p0_snapshot.json"
    native_path = CPU_RESULTS / "native_audit.json"
    if not p0_path.exists() or not native_path.exists():
        raise ProtocolError("Run P0 and the native audit before a GPU/API audit.")
    p0 = read_json(p0_path)
    native = read_json(native_path)
    if p0["releases"]["dual"]["rows"] != 73_907:
        raise ProtocolError("P0 does not describe the expected 73,907-row dual release.")
    if native["analysis_population"]["rows"] != 73_907:
        raise ProtocolError("Native audit does not describe the expected 73,907 rows.")
    return p0, native


def category_set(row: dict, position: int) -> list[str]:
    return sorted(
        category
        for category, present in row[f"response_{position}_harm_category"].items()
        if present
    )


def native_stratum(row: dict) -> str:
    safe0 = bool(row["is_response_0_safe"])
    safe1 = bool(row["is_response_1_safe"])
    if safe0 != safe1:
        return "L4_safety_boundary_difference"
    if int(row["response_0_severity_level"]) != int(row["response_1_severity_level"]):
        return "L3_severity_difference_same_safety_state"
    if category_set(row, 0) != category_set(row, 1):
        return "L2_category_difference_tied_safety_and_severity"
    return "L1_no_recorded_safety_field_difference"


def all_release_rows() -> Iterable[tuple[str, int, dict]]:
    for path in sorted(RAW_DUAL.glob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                yield path.name, line_number, json.loads(line)


def deterministic_pilot_rows(pairs: int, seed: int) -> list[tuple[str, int, dict]]:
    rows = list(all_release_rows())
    if len(rows) != 73_907:
        raise ProtocolError("Raw dual release does not contain 73,907 rows.")
    return random.Random(seed).sample(rows, pairs)


def response_task(source_file: str, source_line: int, row: dict, position: int) -> dict:
    response = row[f"response_{position}"]
    return {
        "pair_id": f"{source_file}:{source_line}",
        "source_file": source_file,
        "source_line": source_line,
        "response_position": position,
        "response_sha256": row[f"response_{position}_sha256"],
        "is_safe": bool(row[f"is_response_{position}_safe"]),
        "severity_level": int(row[f"response_{position}_severity_level"]),
        "harm_categories": category_set(row, position),
        "response_character_length": len(response),
        "safer_response_id": int(row["safer_response_id"]),
        "native_stratum": native_stratum(row),
        "prompt": row["prompt"],
        "response": response,
    }


def response_tasks(rows: Iterable[tuple[str, int, dict]]) -> list[dict]:
    tasks: list[dict] = []
    for source_file, source_line, row in rows:
        tasks.append(response_task(source_file, source_line, row, 0))
        tasks.append(response_task(source_file, source_line, row, 1))
    return tasks


def public_task_fields(task: dict) -> dict:
    return {
        key: task[key]
        for key in (
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
        )
    }


def load_rows_for_manifest(manifest: Path) -> list[tuple[dict, dict]]:
    """Return manifest records paired with their raw release rows.

    The function scans each raw file once, verifies that the selected response
    hashes still match the manifest, and never writes the recovered text.
    """

    with manifest.open(newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    requested: dict[str, dict[int, dict]] = defaultdict(dict)
    for record in records:
        requested[record["source_file"]][int(record["source_line"])] = record
    resolved: list[tuple[dict, dict]] = []
    for filename, line_map in requested.items():
        source = RAW_DUAL / filename
        if not source.exists():
            raise ProtocolError(f"Manifest source file is absent: {filename}")
        with source.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                record = line_map.get(line_number)
                if record is None:
                    continue
                row = json.loads(line)
                if (
                    row["response_0_sha256"] != record["response_0_sha256"]
                    or row["response_1_sha256"] != record["response_1_sha256"]
                ):
                    raise ProtocolError(
                        f"Response hash mismatch for {filename}:{line_number}; rerun E6 sampling."
                    )
                resolved.append((record, row))
    if len(resolved) != len(records):
        raise ProtocolError("At least one E6 manifest row could not be resolved.")
    return resolved


def run_directory(audit: str, run_id: str) -> Path:
    if not run_id or "/" in run_id or "\\" in run_id:
        raise ProtocolError("run_id must be a non-empty directory-safe identifier.")
    path = PRIVATE_ROOT / audit / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def base_run_manifest(audit: str, config_path: Path, run_id: str, script_path: Path) -> dict:
    p0_path = CPU_RESULTS / "p0_snapshot.json"
    native_path = CPU_RESULTS / "native_audit.json"
    return {
        "audit": audit,
        "run_id": run_id,
        "started_at_utc": utc_now(),
        "config_path": str(config_path.relative_to(EXPERIMENTS)),
        "config_sha256": sha256(config_path),
        "script_sha256": script_hash(script_path),
        "p0_manifest_sha256": sha256(p0_path),
        "native_audit_sha256": sha256(native_path),
        "python": sys.version,
        "platform": platform.platform(),
        "private_output_policy": "No prompt or response text is written by this runner.",
    }


def environment_present(name: str) -> bool:
    return bool(os.environ.get(name))
