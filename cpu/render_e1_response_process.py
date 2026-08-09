#!/usr/bin/env python3
"""Render the E1 response-process trace from its frozen documentary protocol."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


SCRIPT = Path(__file__).resolve()
CPU = SCRIPT.parent
CONFIG = CPU / "e1_response_process.json"
RESULTS = CPU / "results"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    protocol = json.loads(CONFIG.read_text(encoding="utf-8"))
    if len(protocol["sources"]) != 4:
        raise ValueError("E1 response-process trace must contain the four frozen source findings")
    result = {
        "result_schema": "pku-saferlhf.e1-response-process-result.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha256(CONFIG),
        "script_sha256": sha256(SCRIPT),
        **protocol,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "e1_response_process.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = ["# E1: response-process documentary trace", ""]
    for index, entry in enumerate(result["sources"], start=1):
        lines.extend([f"## Source {index}", "", entry["source"], "", entry["finding"], ""])
    lines.extend(
        [
            "## Interpretation supported by this trace",
            "",
            result["interpretation"]["supported"],
            "",
            "## Interpretation outside this trace",
            "",
            result["interpretation"]["not_supported"],
            "",
            "## Link to the native data audit",
            "",
            result["link_to_native_audit"],
            "",
        ]
    )
    (RESULTS / "e1_response_process.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"sources": len(result["sources"]), "results": str(RESULTS)}, indent=2))


if __name__ == "__main__":
    main()
