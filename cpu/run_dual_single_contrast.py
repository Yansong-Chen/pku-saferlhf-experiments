#!/usr/bin/env python3
"""Compare the two released PKU-SafeRLHF record designs on exact text pairs.

This analysis is deliberately a comparison of *released records*.  A matching
pair has identical stored response digests in both releases, but the public
artefacts do not identify annotators, task order, instructions, or a rule that
generated the single-dimension label from the dual-dimension labels.  The
script therefore reports correspondence and representational differences; it
does not treat PKU-SafeRLHF-Single as a known compression of PKU-SafeRLHF-Dual.

Only aggregate counts are written.  Neither prompts nor response text are
copied to the experiment results.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT = Path(__file__).resolve()
WORKSPACE = SCRIPT.parents[2]
EXPERIMENTS = SCRIPT.parents[1]
RAW_DUAL = WORKSPACE / "data" / "raw" / "dual"
RAW_SINGLE = WORKSPACE / "data" / "raw" / "single"
P0_MANIFEST = EXPERIMENTS / "cpu" / "results" / "p0_snapshot.json"
RESULTS = EXPERIMENTS / "cpu" / "results"

EXPECTED_DUAL_ROWS = 73_907
EXPECTED_SINGLE_ROWS = 72_996
EXPECTED_ONE_TO_ONE_MATCHES = 64_640


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pair_key(row: dict[str, Any]) -> tuple[str, str]:
    """Return an orientation-invariant identity for one stored response pair."""

    hashes = (str(row["response_0_sha256"]), str(row["response_1_sha256"]))
    if hashes[0] == hashes[1]:
        raise ValueError("A row contains identical response digests")
    return tuple(sorted(hashes))


def category_set(row: dict[str, Any], position: int) -> tuple[str, ...]:
    flags = row[f"response_{position}_harm_category"]
    if not isinstance(flags, dict):
        raise TypeError("Harm-category field is not an object")
    return tuple(sorted(name for name, present in flags.items() if bool(present)))


def native_stratum(row: dict[str, Any]) -> str:
    """Use the locked L1--L4 ordering from the native-record audit."""

    safe0 = bool(row["is_response_0_safe"])
    safe1 = bool(row["is_response_1_safe"])
    severity0 = int(row["response_0_severity_level"])
    severity1 = int(row["response_1_severity_level"])
    categories0 = category_set(row, 0)
    categories1 = category_set(row, 1)
    if safe0 != safe1:
        return "L4_safety_boundary_difference"
    if severity0 != severity1:
        return "L3_severity_difference_same_safety_state"
    if categories0 != categories1:
        return "L2_category_difference_tied_safety_and_severity"
    return "L1_no_recorded_safety_field_difference"


def read_release(directory: Path, required_fields: set[str]) -> tuple[dict[tuple[str, str], list[dict[str, Any]]], int]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    rows = 0
    files = sorted(directory.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No JSONL files found in {directory}")
    for path in files:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                row = json.loads(line)
                missing = required_fields.difference(row)
                if missing:
                    raise ValueError(f"{path.name}:{line_number} is missing {sorted(missing)}")
                groups[pair_key(row)].append(row)
                rows += 1
    return groups, rows


def rate_rows(counter: Counter[str], ordered_keys: list[str], denominator: int) -> list[dict[str, Any]]:
    return [
        {
            "measure": key,
            "rows": counter[key],
            "share_of_matched_pairs": counter[key] / denominator if denominator else None,
        }
        for key in ordered_keys
    ]


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if not P0_MANIFEST.exists():
        raise FileNotFoundError("Missing P0 manifest; freeze the releases before comparing them")
    p0 = json.loads(P0_MANIFEST.read_text(encoding="utf-8"))
    if p0["releases"]["dual"]["rows"] != EXPECTED_DUAL_ROWS:
        raise ValueError("Pinned dual release does not have the expected row count")
    if p0["releases"]["single"]["rows"] != EXPECTED_SINGLE_ROWS:
        raise ValueError("Pinned single release does not have the expected row count")

    dual_groups, dual_rows = read_release(
        RAW_DUAL,
        {
            "prompt",
            "response_0_sha256",
            "response_1_sha256",
            "better_response_id",
            "safer_response_id",
            "is_response_0_safe",
            "is_response_1_safe",
            "response_0_harm_category",
            "response_1_harm_category",
            "response_0_severity_level",
            "response_1_severity_level",
        },
    )
    single_groups, single_rows = read_release(
        RAW_SINGLE,
        {
            "prompt",
            "response_0_sha256",
            "response_1_sha256",
            "better_response_id",
        },
    )
    if dual_rows != EXPECTED_DUAL_ROWS or single_rows != EXPECTED_SINGLE_ROWS:
        raise ValueError(f"Unexpected input rows: dual={dual_rows}, single={single_rows}")

    common_keys = set(dual_groups).intersection(single_groups)
    one_to_one_keys = sorted(
        key
        for key in common_keys
        if len(dual_groups[key]) == 1 and len(single_groups[key]) == 1
    )
    if len(one_to_one_keys) != EXPECTED_ONE_TO_ONE_MATCHES:
        raise ValueError(
            f"Expected {EXPECTED_ONE_TO_ONE_MATCHES} one-to-one matches, found {len(one_to_one_keys)}"
        )

    overall_agreement = Counter[str]()
    criteria_relation = Counter[str]()
    agreement_decomposition = Counter[str]()
    conflict_by_stratum: Counter[str] = Counter()
    conflict_follow_safety: Counter[str] = Counter()
    conflict_follow_helpfulness: Counter[str] = Counter()
    all_l4_overall_selects_safe = Counter[str]()
    prompt_digest_mismatches = 0

    for key in one_to_one_keys:
        dual = dual_groups[key][0]
        single = single_groups[key][0]
        if dual["prompt"] != single["prompt"]:
            prompt_digest_mismatches += 1

        dual_hashes = (
            str(dual["response_0_sha256"]),
            str(dual["response_1_sha256"]),
        )
        single_hashes = (
            str(single["response_0_sha256"]),
            str(single["response_1_sha256"]),
        )
        dual_better = dual_hashes[int(dual["better_response_id"])]
        dual_safer = dual_hashes[int(dual["safer_response_id"])]
        single_overall = single_hashes[int(single["better_response_id"])]

        if single_overall not in key:
            raise AssertionError("Single winner is not a member of its response pair")
        overall_agreement["single_equals_dual_better"] += int(single_overall == dual_better)
        overall_agreement["single_equals_dual_safer"] += int(single_overall == dual_safer)

        if dual_better == dual_safer:
            criteria_relation["dual_better_and_safer_agree"] += 1
            if single_overall == dual_better:
                agreement_decomposition["single_preserves_shared_direction"] += 1
            else:
                agreement_decomposition["single_departs_from_shared_direction"] += 1
            continue

        criteria_relation["dual_better_and_safer_conflict"] += 1
        stratum = native_stratum(dual)
        conflict_by_stratum[stratum] += 1
        if single_overall == dual_safer:
            conflict_follow_safety[stratum] += 1
        elif single_overall == dual_better:
            conflict_follow_helpfulness[stratum] += 1
        else:
            raise AssertionError("A binary Single choice followed neither Dual direction")

        if stratum == "L4_safety_boundary_difference":
            safe_by_hash = {
                dual_hashes[position]: bool(dual[f"is_response_{position}_safe"])
                for position in (0, 1)
            }
            if safe_by_hash[dual_safer] is not True:
                raise AssertionError("L4 safer selection did not select the released-safe response")
            all_l4_overall_selects_safe["conflict_rows"] += 1
            all_l4_overall_selects_safe["single_selects_released_safe"] += int(
                safe_by_hash[single_overall]
            )

    if prompt_digest_mismatches:
        raise AssertionError(f"Found {prompt_digest_mismatches} prompt mismatches in exact matches")
    matched_rows = len(one_to_one_keys)
    conflict_total = criteria_relation["dual_better_and_safer_conflict"]
    agreement_total = criteria_relation["dual_better_and_safer_agree"]
    if agreement_total + conflict_total != matched_rows:
        raise AssertionError("Criteria relation does not partition the matched cohort")
    if sum(conflict_by_stratum.values()) != conflict_total:
        raise AssertionError("Conflict strata do not sum to the conflict cohort")

    stratum_order = [
        "L1_no_recorded_safety_field_difference",
        "L2_category_difference_tied_safety_and_severity",
        "L3_severity_difference_same_safety_state",
        "L4_safety_boundary_difference",
    ]
    stratum_rows = []
    for stratum in stratum_order:
        n = conflict_by_stratum[stratum]
        follows_safety = conflict_follow_safety[stratum]
        follows_helpfulness = conflict_follow_helpfulness[stratum]
        if follows_safety + follows_helpfulness != n:
            raise AssertionError(f"Conflict direction counts do not sum in {stratum}")
        stratum_rows.append(
            {
                "stratum": stratum,
                "dual_better_safer_conflict_pairs": n,
                "single_follows_dual_safer": follows_safety,
                "share_single_follows_dual_safer": follows_safety / n if n else None,
                "single_follows_dual_better": follows_helpfulness,
                "share_single_follows_dual_better": follows_helpfulness / n if n else None,
            }
        )

    attrition_rows = []
    for release, groups, total in (
        ("dual", dual_groups, dual_rows),
        ("single", single_groups, single_rows),
    ):
        matched_records = sum(
            len(groups[key])
            for key in one_to_one_keys
        )
        attrition_rows.append(
            {
                "release": release,
                "released_rows": total,
                "one_to_one_matched_rows": matched_records,
                "share_one_to_one_matched": matched_records / total,
                "rows_outside_one_to_one_cohort": total - matched_records,
            }
        )

    results = {
        "result_schema": "pku-saferlhf.dual-single-contrast.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "research_question": (
            "How do the released composite overall preferences in PKU-SafeRLHF-Single "
            "correspond to the separately released helpfulness and safety preferences "
            "and safety fields in PKU-SafeRLHF-Dual on identical response pairs?"
        ),
        "interpretive_boundary": (
            "This is a correspondence analysis of separately released records. Exact "
            "response-pair matching does not establish common annotators, a shared task "
            "procedure, a temporal order, or a documented transformation from Dual to Single."
        ),
        "provenance": {
            "p0_manifest_path": str(P0_MANIFEST.relative_to(EXPERIMENTS)),
            "p0_manifest_sha256": sha256_file(P0_MANIFEST),
            "dual_revision": p0["releases"]["dual"]["revision"],
            "single_revision": p0["releases"]["single"]["revision"],
            "script_sha256": sha256_file(SCRIPT),
            "match_key": "sorted pair of stored response_0_sha256 and response_1_sha256",
            "matched_prompt_mismatches": prompt_digest_mismatches,
        },
        "matching": {
            "dual_released_rows": dual_rows,
            "single_released_rows": single_rows,
            "common_unordered_response_hash_groups": len(common_keys),
            "one_to_one_matched_pairs": matched_rows,
            "attrition_rows_file": "dual_single_match_attrition.csv",
        },
        "overall_direction_correspondence": {
            "single_equals_dual_better": {
                "rows": overall_agreement["single_equals_dual_better"],
                "share": overall_agreement["single_equals_dual_better"] / matched_rows,
            },
            "single_equals_dual_safer": {
                "rows": overall_agreement["single_equals_dual_safer"],
                "share": overall_agreement["single_equals_dual_safer"] / matched_rows,
            },
        },
        "dual_preference_relation": {
            "agree": {
                "rows": agreement_total,
                "share": agreement_total / matched_rows,
                "single_preserves_shared_direction": {
                    "rows": agreement_decomposition["single_preserves_shared_direction"],
                    "share_given_agreement": agreement_decomposition[
                        "single_preserves_shared_direction"
                    ]
                    / agreement_total,
                },
                "single_departs_from_shared_direction": {
                    "rows": agreement_decomposition["single_departs_from_shared_direction"],
                    "share_given_agreement": agreement_decomposition[
                        "single_departs_from_shared_direction"
                    ]
                    / agreement_total,
                },
            },
            "conflict": {
                "rows": conflict_total,
                "share": conflict_total / matched_rows,
                "single_follows_dual_safer": {
                    "rows": sum(conflict_follow_safety.values()),
                    "share_given_conflict": sum(conflict_follow_safety.values()) / conflict_total,
                },
                "single_follows_dual_better": {
                    "rows": sum(conflict_follow_helpfulness.values()),
                    "share_given_conflict": sum(conflict_follow_helpfulness.values()) / conflict_total,
                },
                "by_native_dual_stratum_file": "dual_single_conflict_strata.csv",
            },
        },
        "l4_binary_boundary_contrast": {
            "definition": (
                "Subset of matched pairs where Dual better and safer select different responses "
                "and Dual records different is_safe states. In every such matched row, the Dual "
                "safer selection is the released-safe response."
            ),
            "rows": all_l4_overall_selects_safe["conflict_rows"],
            "single_selects_dual_released_safe": all_l4_overall_selects_safe[
                "single_selects_released_safe"
            ],
            "share": all_l4_overall_selects_safe["single_selects_released_safe"]
            / all_l4_overall_selects_safe["conflict_rows"],
        },
        "output_files": {
            "conflict_strata": "dual_single_conflict_strata.csv",
            "match_attrition": "dual_single_match_attrition.csv",
        },
    }

    RESULTS.mkdir(parents=True, exist_ok=True)
    write_csv(
        RESULTS / "dual_single_conflict_strata.csv",
        stratum_rows,
        [
            "stratum",
            "dual_better_safer_conflict_pairs",
            "single_follows_dual_safer",
            "share_single_follows_dual_safer",
            "single_follows_dual_better",
            "share_single_follows_dual_better",
        ],
    )
    write_csv(
        RESULTS / "dual_single_match_attrition.csv",
        attrition_rows,
        [
            "release",
            "released_rows",
            "one_to_one_matched_rows",
            "share_one_to_one_matched",
            "rows_outside_one_to_one_cohort",
        ],
    )
    (RESULTS / "dual_single_contrast.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "one_to_one_matched_pairs": matched_rows,
                "dual_preference_conflicts": conflict_total,
                "single_follows_safer_given_conflict": results["dual_preference_relation"][
                    "conflict"
                ]["single_follows_dual_safer"],
                "l4_binary_boundary_contrast": results["l4_binary_boundary_contrast"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
