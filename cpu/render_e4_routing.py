#!/usr/bin/env python3
"""Render the frozen, text-free E4 field-routing documentation table."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


SCRIPT = Path(__file__).resolve()
EXPERIMENTS = SCRIPT.parents[1]
CONFIG = SCRIPT.parent / "e4_field_routing.json"
RESULTS = SCRIPT.parent / "results"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    data = json.loads(CONFIG.read_text(encoding="utf-8"))
    expected = {
        "better_response_id",
        "safer_response_id",
        "is_response_0_safe / is_response_1_safe",
        "response_0_harm_category / response_1_harm_category",
        "response_0_severity_level / response_1_severity_level",
    }
    observed = {entry["field"] for entry in data["fields"]}
    if observed != expected:
        raise ValueError("E4 field list differs from the frozen protocol")

    RESULTS.mkdir(parents=True, exist_ok=True)
    result = {
        "result_schema": "pku-saferlhf.e4-field-routing-result.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha256(CONFIG),
        "script_sha256": sha256(SCRIPT),
        **data,
    }
    (RESULTS / "e4_field_routing.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# E4: field-to-objective and branch-routing trace",
        "",
        f"Protocol SHA-256: {result['protocol_sha256']}",
        "",
        "| Released field | Released role | Moderation branch | Reward loss | Cost loss | Evidence |",
        "|---|---|---|---|---|---|",
    ]
    for field in result["fields"]:
        lines.append(
            "| {field} | {released_role} | {moderation_branch} | {reward_loss} | {cost_loss} | {evidence} |".format(
                **field
            )
        )
    lines.extend(["", "## Source hierarchy", ""])
    for source in result["source_hierarchy"]:
        lines.append(f"{source['rank']}. {source['source']} — {source['status']}.")
    lines.extend(["", "## Registered discrepancy", ""])
    for discrepancy in result["discrepancy_register"]:
        lines.append(
            f"- **{discrepancy['id']}** — {discrepancy['description']} "
            f"Treatment: {discrepancy['treatment']}"
        )
    lines.extend(["", "## Interpretive boundary", "", result["interpretive_boundary"], ""])
    (RESULTS / "e4_field_routing.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"fields": len(observed), "results": str(RESULTS)}, indent=2))


if __name__ == "__main__":
    main()
