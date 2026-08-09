#!/usr/bin/env python3
"""Run the non-model components of the PKU-SafeRLHF native-record audit.

The script reads the pinned dual-dimension JSONL release and writes only
aggregate results, metadata, and response hashes for the few schema exceptions.
Prompt and response text are never copied into a tracked experiment artefact.
It also prepares ignored binary category matrices for the R EFA script.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT = Path(__file__).resolve()
WORKSPACE = SCRIPT.parents[2]
EXPERIMENTS = SCRIPT.parents[1]
RAW_DUAL = WORKSPACE / "data" / "raw" / "dual"
P0_MANIFEST = EXPERIMENTS / "cpu" / "results" / "p0_snapshot.json"
RESULTS = EXPERIMENTS / "cpu" / "results"
INTERMEDIATE = EXPERIMENTS / "cpu" / "intermediate"
SEED = 20260809
EXPECTED_ROWS = 73_907
EXPECTED_POSITIONS = 147_814


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def ordered_category_set(flags: dict[str, bool], categories: list[str]) -> tuple[str, ...]:
    return tuple(category for category in categories if bool(flags[category]))


def safety_configuration(safe0: bool, safe1: bool) -> str:
    if safe0 and safe1:
        return "both_safe"
    if not safe0 and not safe1:
        return "both_unsafe"
    if safe0:
        return "response_0_safe_response_1_unsafe"
    return "response_0_unsafe_response_1_safe"


def native_stratum(
    safe0: bool,
    safe1: bool,
    severity0: int,
    severity1: int,
    categories0: tuple[str, ...],
    categories1: tuple[str, ...],
) -> str:
    """Return four exhaustive strata used by the GPU/API experiments.

    Priority makes the strata exhaustive even for the eight schema-exception
    response positions: an absolute safety-state difference is L4; otherwise a
    severity difference is L3; otherwise a category-set difference is L2; and
    all remaining rows are L1.
    """

    if safe0 != safe1:
        return "L4_safety_boundary_difference"
    if severity0 != severity1:
        return "L3_severity_difference_same_safety_state"
    if categories0 != categories1:
        return "L2_category_difference_tied_safety_and_severity"
    return "L1_no_recorded_safety_field_difference"


def main() -> None:
    if not P0_MANIFEST.exists():
        raise FileNotFoundError(
            "P0 manifest is missing. Run python3 ../scripts/p0_snapshot.py first."
        )
    raw_files = sorted(RAW_DUAL.glob("*.jsonl"))
    if not raw_files:
        raise FileNotFoundError(f"No dual-dimension JSONL files found in {RAW_DUAL}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    INTERMEDIATE.mkdir(parents=True, exist_ok=True)
    p0 = json.loads(P0_MANIFEST.read_text(encoding="utf-8"))
    p0_sha256 = sha256_file(P0_MANIFEST)

    categories: list[str] | None = None
    category_counts: Counter[str] = Counter()
    category_solo_counts: Counter[str] = Counter()
    category_severity: Counter[tuple[str, int]] = Counter()
    category_pair_counts: Counter[tuple[str, str]] = Counter()
    severity_counts: Counter[int] = Counter()
    safety_counts: Counter[str] = Counter()
    field_presence: Counter[str] = Counter()
    field_type_support: defaultdict[str, Counter[str]] = defaultdict(Counter)
    identity_counts = Counter()
    exceptions: list[dict[str, Any]] = []
    absolute_decomposition: Counter[tuple[bool, bool]] = Counter()
    configuration_counts: Counter[str] = Counter()
    configuration_winner_unsafe: Counter[str] = Counter()
    strata: Counter[str] = Counter()
    strata_both_unsafe: Counter[str] = Counter()
    pairwise_agreement: Counter[str] = Counter()
    source_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    row_count = 0
    position_count = 0
    required_fields: list[str] | None = None
    rng = random.Random(SEED)

    all_matrix_path = INTERMEDIATE / "category_matrix_all_positions.csv"
    one_matrix_path = INTERMEDIATE / "category_matrix_one_response_per_pair.csv"

    with all_matrix_path.open("w", newline="", encoding="utf-8") as all_handle, one_matrix_path.open(
        "w", newline="", encoding="utf-8"
    ) as one_handle:
        all_writer: csv.DictWriter | None = None
        one_writer: csv.DictWriter | None = None

        for path in raw_files:
            with path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    row = json.loads(line)
                    row_count += 1

                    if required_fields is None:
                        required_fields = list(row.keys())
                    elif list(row.keys()) != required_fields:
                        raise ValueError(
                            f"Schema differs in {path.name}:{line_number}; P0 must be rerun."
                        )
                    for field, value in row.items():
                        field_presence[field] += 1
                        field_type_support[field][type(value).__name__] += 1

                    for source_field in (
                        "prompt_source",
                        "response_0_source",
                        "response_1_source",
                    ):
                        source_counts[source_field][str(row[source_field])] += 1

                    response_records: list[dict[str, Any]] = []
                    for position in (0, 1):
                        category_flags = row[f"response_{position}_harm_category"]
                        if not isinstance(category_flags, dict):
                            raise TypeError(
                                f"Category field has unexpected type at {path.name}:{line_number}"
                            )
                        if categories is None:
                            categories = list(category_flags.keys())
                            if len(categories) != 19:
                                raise ValueError(
                                    f"Expected 19 harm categories, found {len(categories)}"
                                )
                            matrix_columns = [
                                "source_file",
                                "source_line",
                                "response_position",
                                "response_sha256",
                                "is_safe",
                                "severity_level",
                                *categories,
                            ]
                            all_writer = csv.DictWriter(all_handle, fieldnames=matrix_columns)
                            one_writer = csv.DictWriter(one_handle, fieldnames=matrix_columns)
                            all_writer.writeheader()
                            one_writer.writeheader()
                        if list(category_flags.keys()) != categories:
                            raise ValueError(
                                f"Category schema differs in {path.name}:{line_number}"
                            )

                        safe = bool(row[f"is_response_{position}_safe"])
                        severity = int(row[f"response_{position}_severity_level"])
                        assigned = ordered_category_set(category_flags, categories)
                        category_count = len(assigned)
                        position_count += 1
                        safety_counts["safe" if safe else "unsafe"] += 1
                        severity_counts[severity] += 1
                        identity_counts["safe_equals_severity_zero"] += int(
                            safe == (severity == 0)
                        )
                        identity_counts["safe_equals_empty_category_set"] += int(
                            safe == (category_count == 0)
                        )

                        if safe != (severity == 0) or safe != (category_count == 0):
                            exceptions.append(
                                {
                                    "source_file": path.name,
                                    "source_line": line_number,
                                    "response_position": position,
                                    "response_sha256": row[f"response_{position}_sha256"],
                                    "is_safe": str(safe).lower(),
                                    "severity_level": severity,
                                    "assigned_category_count": category_count,
                                    "safe_equals_severity_zero": str(
                                        safe == (severity == 0)
                                    ).lower(),
                                    "safe_equals_empty_category_set": str(
                                        safe == (category_count == 0)
                                    ).lower(),
                                }
                            )

                        for category in assigned:
                            category_counts[category] += 1
                            category_severity[(category, severity)] += 1
                        if category_count == 1:
                            category_solo_counts[assigned[0]] += 1
                        for category_a in assigned:
                            for category_b in assigned:
                                if category_a != category_b:
                                    category_pair_counts[(category_a, category_b)] += 1

                        matrix_row: dict[str, Any] = {
                            "source_file": path.name,
                            "source_line": line_number,
                            "response_position": position,
                            "response_sha256": row[f"response_{position}_sha256"],
                            "is_safe": safe,
                            "severity_level": severity,
                        }
                        matrix_row.update(
                            {category: int(category in assigned) for category in categories}
                        )
                        if all_writer is None:
                            raise RuntimeError("EFA writer was not initialised")
                        all_writer.writerow(matrix_row)
                        response_records.append(
                            {
                                "safe": safe,
                                "severity": severity,
                                "assigned": assigned,
                                "matrix_row": matrix_row,
                            }
                        )

                    selected_for_sensitivity = rng.randrange(2)
                    if one_writer is None:
                        raise RuntimeError("EFA writer was not initialised")
                    one_writer.writerow(response_records[selected_for_sensitivity]["matrix_row"])

                    safe0, safe1 = (
                        response_records[0]["safe"],
                        response_records[1]["safe"],
                    )
                    severity0, severity1 = (
                        response_records[0]["severity"],
                        response_records[1]["severity"],
                    )
                    assigned0, assigned1 = (
                        response_records[0]["assigned"],
                        response_records[1]["assigned"],
                    )
                    safer = int(row["safer_response_id"])
                    better = int(row["better_response_id"])
                    winner_safe = (safe0, safe1)[safer]
                    alternative_safe = (safe0, safe1)[1 - safer]
                    configuration = safety_configuration(safe0, safe1)
                    stratum = native_stratum(
                        safe0, safe1, severity0, severity1, assigned0, assigned1
                    )

                    absolute_decomposition[(winner_safe, alternative_safe)] += 1
                    configuration_counts[configuration] += 1
                    configuration_winner_unsafe[configuration] += int(not winner_safe)
                    strata[stratum] += 1
                    strata_both_unsafe[stratum] += int(not safe0 and not safe1)
                    pairwise_agreement[
                        "same_selected_response"
                        if safer == better
                        else "different_selected_response"
                    ] += 1

    if categories is None or required_fields is None:
        raise RuntimeError("No records were read")
    if row_count != EXPECTED_ROWS or position_count != EXPECTED_POSITIONS:
        raise ValueError(
            f"Unexpected population: {row_count} rows and {position_count} positions"
        )

    category_frequency_rows = []
    category_solo_rows = []
    category_severity_rows = []
    for category in categories:
        count = category_counts[category]
        category_frequency_rows.append(
            {
                "category": category,
                "response_positions": count,
                "share_of_response_positions": count / position_count,
            }
        )
        category_solo_rows.append(
            {
                "category": category,
                "solo_response_positions": category_solo_counts[category],
                "solo_share_given_category": (
                    category_solo_counts[category] / count if count else None
                ),
            }
        )
        for severity in range(4):
            level_count = category_severity[(category, severity)]
            category_severity_rows.append(
                {
                    "category": category,
                    "severity_level": severity,
                    "response_positions": level_count,
                    "share_given_category": level_count / count if count else None,
                }
            )

    cooccurrence_rows = []
    for category_a in categories:
        for category_b in categories:
            if category_a == category_b:
                continue
            count_a = category_counts[category_a]
            joint = category_pair_counts[(category_a, category_b)]
            cooccurrence_rows.append(
                {
                    "conditioning_category": category_a,
                    "other_category": category_b,
                    "conditioning_count": count_a,
                    "joint_count": joint,
                    "p_other_given_conditioning": joint / count_a if count_a else None,
                }
            )

    decomposition_rows = []
    for winner_safe in (True, False):
        for alternative_safe in (True, False):
            count = absolute_decomposition[(winner_safe, alternative_safe)]
            decomposition_rows.append(
                {
                    "safer_winner_is_safe": str(winner_safe).lower(),
                    "alternative_is_safe": str(alternative_safe).lower(),
                    "rows": count,
                    "share_of_rows": count / row_count,
                }
            )

    configuration_rows = [
        {
            "configuration": configuration,
            "rows": configuration_counts[configuration],
            "share_of_rows": configuration_counts[configuration] / row_count,
            "safer_winner_unsafe_rows": configuration_winner_unsafe[configuration],
            "share_winner_unsafe_given_configuration": (
                configuration_winner_unsafe[configuration]
                / configuration_counts[configuration]
                if configuration_counts[configuration]
                else None
            ),
        }
        for configuration in (
            "both_safe",
            "response_0_safe_response_1_unsafe",
            "response_0_unsafe_response_1_safe",
            "both_unsafe",
        )
    ]
    stratum_rows = [
        {
            "stratum": stratum,
            "rows": strata[stratum],
            "share_of_rows": strata[stratum] / row_count,
            "both_unsafe_rows": strata_both_unsafe[stratum],
        }
        for stratum in (
            "L1_no_recorded_safety_field_difference",
            "L2_category_difference_tied_safety_and_severity",
            "L3_severity_difference_same_safety_state",
            "L4_safety_boundary_difference",
        )
    ]

    write_csv(
        RESULTS / "category_frequency.csv",
        category_frequency_rows,
        ["category", "response_positions", "share_of_response_positions"],
    )
    write_csv(
        RESULTS / "category_solo_rate.csv",
        category_solo_rows,
        ["category", "solo_response_positions", "solo_share_given_category"],
    )
    write_csv(
        RESULTS / "category_conditional_cooccurrence.csv",
        cooccurrence_rows,
        [
            "conditioning_category",
            "other_category",
            "conditioning_count",
            "joint_count",
            "p_other_given_conditioning",
        ],
    )
    write_csv(
        RESULTS / "category_severity_profile.csv",
        category_severity_rows,
        ["category", "severity_level", "response_positions", "share_given_category"],
    )
    write_csv(
        RESULTS / "absolute_state_decomposition.csv",
        decomposition_rows,
        ["safer_winner_is_safe", "alternative_is_safe", "rows", "share_of_rows"],
    )
    write_csv(
        RESULTS / "safety_configurations.csv",
        configuration_rows,
        [
            "configuration",
            "rows",
            "share_of_rows",
            "safer_winner_unsafe_rows",
            "share_winner_unsafe_given_configuration",
        ],
    )
    write_csv(
        RESULTS / "native_strata.csv",
        stratum_rows,
        ["stratum", "rows", "share_of_rows", "both_unsafe_rows"],
    )
    write_csv(
        RESULTS / "schema_identity_exceptions.csv",
        exceptions,
        [
            "source_file",
            "source_line",
            "response_position",
            "response_sha256",
            "is_safe",
            "severity_level",
            "assigned_category_count",
            "safe_equals_severity_zero",
            "safe_equals_empty_category_set",
        ],
    )

    result = {
        "result_schema": "pku-saferlhf.native-audit.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_population": {
            "release": "PKU-Alignment/PKU-SafeRLHF dual-dimension training split",
            "rows": row_count,
            "response_positions": position_count,
            "source_files": [path.name for path in raw_files],
        },
        "provenance": {
            "p0_manifest_path": str(P0_MANIFEST.relative_to(EXPERIMENTS)),
            "p0_manifest_sha256": p0_sha256,
            "p0_dual_revision": p0["releases"]["dual"]["revision"],
            "script_sha256": sha256_file(SCRIPT),
            "seed_one_response_per_pair": SEED,
        },
        "schema": {
            "fields": required_fields,
            "field_presence_rows": dict(field_presence),
            "field_type_support": {
                field: dict(counter) for field, counter in field_type_support.items()
            },
            "harm_categories": categories,
            "schema_identity": {
                "safe_equals_severity_zero": {
                    "numerator": identity_counts["safe_equals_severity_zero"],
                    "denominator": position_count,
                },
                "safe_equals_empty_category_set": {
                    "numerator": identity_counts["safe_equals_empty_category_set"],
                    "denominator": position_count,
                },
                "exception_response_positions": len(exceptions),
                "exception_file": "schema_identity_exceptions.csv",
            },
        },
        "response_level_marginals": {
            "is_safe": dict(safety_counts),
            "severity_level": {
                str(level): severity_counts[level] for level in sorted(severity_counts)
            },
        },
        "pairwise_labels": dict(pairwise_agreement),
        "absolute_state_decomposition": decomposition_rows,
        "safety_configurations": configuration_rows,
        "native_strata": {
            "definition": (
                "L4: is_safe differs; else L3: severity differs; else L2: complete "
                "category set differs; else L1: all three released safety fields tie."
            ),
            "rows": stratum_rows,
        },
        "output_files": {
            "category_frequency": "category_frequency.csv",
            "category_solo_rate": "category_solo_rate.csv",
            "conditional_cooccurrence": "category_conditional_cooccurrence.csv",
            "category_severity_profile": "category_severity_profile.csv",
            "absolute_state_decomposition": "absolute_state_decomposition.csv",
            "safety_configurations": "safety_configurations.csv",
            "native_strata": "native_strata.csv",
            "all_position_category_matrix_ignored": str(
                all_matrix_path.relative_to(EXPERIMENTS)
            ),
            "one_response_category_matrix_ignored": str(
                one_matrix_path.relative_to(EXPERIMENTS)
            ),
        },
        "source_field_profiles": {
            field: dict(counter) for field, counter in source_counts.items()
        },
    }
    (RESULTS / "native_audit.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "rows": row_count,
                "response_positions": position_count,
                "unsafe_safer_winners": absolute_decomposition[(False, False)]
                + absolute_decomposition[(False, True)],
                "schema_exceptions": len(exceptions),
                "results": str(RESULTS),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
