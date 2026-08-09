#!/usr/bin/env python3
"""Freeze the versioned PKU-SafeRLHF input manifest used by this workbench.

Only release metadata, hashes, schemas, and duplicate-aware match counts are
written. No prompt or response text leaves the raw JSONL files.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


SCRIPT = Path(__file__).resolve()
EXPERIMENTS = SCRIPT.parents[1]
WORKSPACE = SCRIPT.parents[2]
RESULTS = EXPERIMENTS / "cpu" / "results"
RELEASES = {
    "dual": {
        "dataset": "PKU-Alignment/PKU-SafeRLHF",
        "revision": "9421ffafec3fa40a1f1a7d567b4d525079477ecb",
        "directory": WORKSPACE / "data" / "raw" / "dual",
    },
    "single": {
        "dataset": "PKU-Alignment/PKU-SafeRLHF-single-dimension",
        "revision": "877416778ed53479d63315795558fe83495276d8",
        "directory": WORKSPACE / "data" / "raw" / "single",
    },
}
EXPECTED_FILE_SHA256 = {
    "Alpaca-7B-train.jsonl": {
        "dual": "e9aeebe2bbbed37d7a28561809f47c6c0622e48b170c80bc2eb72d908d2c4666",
        "single": "643fe074861fa7a0883a590c51e3c5e356d7f00e988cee125c0328a6573d8e5e",
    },
    "Alpaca2-7B-train.jsonl": {
        "dual": "f7845941c570087a320121d222cbdc500189c59310c60aadaf23b14877c92e0a",
        "single": "b8dce8f06a2d608d93f7b49e7ea83df55fab823e06825567476421d880a43f7e",
    },
    "Alpaca3-8B-train.jsonl": {
        "dual": "21c3b4f2572444e42d161f09166471d14208c9e153fcc0ec900ae05013e47d4c",
        "single": "2d39f5e0af751db8b7b61fbb6e5ea16a1cfb01ec3569ba9f654a9227f406ae7f",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pair_key(row: dict) -> tuple[str, str]:
    return tuple(sorted((row["response_0_sha256"], row["response_1_sha256"])))


def scan_release(name: str) -> tuple[dict, dict[tuple[str, str], list[str]]]:
    release = RELEASES[name]
    files = sorted(release["directory"].glob("*.jsonl"))
    if {path.name for path in files} != set(EXPECTED_FILE_SHA256):
        raise FileNotFoundError(f"Unexpected {name} release file set")
    schemas: Counter[tuple[str, ...]] = Counter()
    prompt_counts: Counter[str] = Counter()
    pair_groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    file_reports = []
    row_count = 0
    positions = 0
    safe_severity_agreement = 0

    for path in files:
        file_rows = 0
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                schemas[tuple(row.keys())] += 1
                row_count += 1
                file_rows += 1
                prompt_counts[row["prompt"]] += 1
                pair_groups[pair_key(row)].append(
                    hashlib.sha256(row["prompt"].encode("utf-8")).hexdigest()
                )
                if name == "dual":
                    for position in (0, 1):
                        safe = bool(row[f"is_response_{position}_safe"])
                        severity = int(row[f"response_{position}_severity_level"])
                        safe_severity_agreement += int(safe == (severity == 0))
                        positions += 1
        file_hash = sha256(path)
        expected = EXPECTED_FILE_SHA256[path.name][name]
        if file_hash != expected:
            raise ValueError(f"Checksum mismatch for {path.name}")
        file_reports.append(
            {
                "path": str(path.relative_to(WORKSPACE)),
                "bytes": path.stat().st_size,
                "sha256": file_hash,
                "matches_pinned_lfs_oid": True,
                "rows": file_rows,
            }
        )
    report = {
        "dataset": release["dataset"],
        "revision": release["revision"],
        "files": file_reports,
        "rows": row_count,
        "distinct_prompts": len(prompt_counts),
        "duplicate_pair_groups": sum(len(group) > 1 for group in pair_groups.values()),
        "duplicate_pair_rows": sum(len(group) for group in pair_groups.values() if len(group) > 1),
        "schemas": [
            {"fields": list(schema), "rows": count}
            for schema, count in schemas.items()
        ],
    }
    if name == "dual":
        report["safe_equals_severity_zero"] = {
            "numerator": safe_severity_agreement,
            "denominator": positions,
        }
    return report, pair_groups


def matching_report(
    dual: dict[tuple[str, str], list[str]],
    single: dict[tuple[str, str], list[str]],
) -> dict:
    common = set(dual) & set(single)
    return {
        "common_unordered_response_hash_groups": len(common),
        "raw_cross_product_row_matches": sum(
            len(dual[key]) * len(single[key]) for key in common
        ),
        "one_to_one_row_matches_before_duplicate_exclusion": sum(
            min(len(dual[key]), len(single[key])) for key in common
        ),
        "common_groups_with_prompt_digest_mismatch": sum(
            set(dual[key]) != set(single[key]) for key in common
        ),
        "one_to_one_matches_after_excluding_any_duplicate_group": sum(
            len(dual[key]) == 1 and len(single[key]) == 1 for key in common
        ),
    }


def main() -> None:
    dual_report, dual_groups = scan_release("dual")
    single_report, single_groups = scan_release("single")
    output = {
        "result_schema": "pku-saferlhf.p0-snapshot.v1",
        "script_sha256": sha256(SCRIPT),
        "releases": {"dual": dual_report, "single": single_report},
        "exact_matching": matching_report(dual_groups, single_groups),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    destination = RESULTS / "p0_snapshot.json"
    destination.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
