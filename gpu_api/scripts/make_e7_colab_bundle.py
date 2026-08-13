#!/usr/bin/env python3
"""Build the private, frozen E7 Colab package from this workspace.

The archive contains only the selected pilot and primary response positions,
never the full PKU-SafeRLHF release.  The job files are private because they
contain prompts and responses.  They are deliberately excluded from Git.

Run from ``experiments``:

    python gpu_api/scripts/make_e7_colab_bundle.py \
        --output /Users/yansongchen/Downloads/e7_beaver_cost_colab_bundle.tgz
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from common import (
    CONFIG_DIR,
    CPU_RESULTS,
    ProtocolError,
    deterministic_pilot_rows,
    primary_sample_rows,
    public_task_fields,
    read_json,
    response_tasks,
    sha256,
    validate_cpu_provenance,
)


SCRIPT = Path(__file__).resolve()
EXPERIMENTS = SCRIPT.parents[2]
WORKSPACE = EXPERIMENTS.parent
CONFIG_PATH = CONFIG_DIR / "e7_cost_probe.json"
PRIMARY_SAMPLE = CONFIG_DIR / "primary_pair_sample_manifest.csv"
COLAB_DIR = EXPERIMENTS / "gpu_api" / "colab"
SAFE_RLHF_REVISION = "e8cca16665ef2340ac92c6514f05519310251581"
VENDOR_FILES = {
    "safe_rlhf/__init__.py": "# Source-locked minimal package for E7 Colab scoring.\n",
    "safe_rlhf/models/__init__.py": (
        "# Deliberately excludes safe_rlhf.models.pretrained, which imports deepspeed.\n"
        "from safe_rlhf.models.score_model import AutoModelForScore, ScoreModelOutput\n\n"
        "__all__ = ['AutoModelForScore', 'ScoreModelOutput']\n"
    ),
}
# AutoModelForScore registers eleven architectures through a lazy mapping.  The
# lazy loader resolves a family module by name, so vendoring only ``llama``
# leaves the registry able to name families it cannot import and the probe fails
# with ModuleNotFoundError on ``score_model.bloom``.  Vendoring every registered
# family keeps the upstream ``__init__`` unmodified, which matters more than the
# few extra files: no hand-written substitute enters the scoring path.
SCORE_MODEL_FAMILIES = (
    "bloom",
    "gemma",
    "gpt_neo",
    "gpt_neox",
    "gpt2",
    "gptj",
    "llama",
    "mistral",
    "opt",
    "phi",
    "qwen2",
)

UPSTREAM_VENDOR_PATHS = (
    "safe_rlhf/models/normalizer.py",
    "safe_rlhf/models/score_model/__init__.py",
    *(
        path
        for family in SCORE_MODEL_FAMILIES
        for path in (
            f"safe_rlhf/models/score_model/{family}/__init__.py",
            f"safe_rlhf/models/score_model/{family}/modeling_{family}.py",
        )
    ),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    return sha256(path)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def position_key(task: dict) -> str:
    return ":".join(
        (task["source_file"], str(task["source_line"]), str(task["response_position"]))
    )


def make_job(tasks: list[dict], phase: str, config: dict, output_dir: Path) -> None:
    job_rows = [
        {
            "position_key": position_key(task),
            **task,
        }
        for task in tasks
    ]
    if len({task["position_key"] for task in job_rows}) != len(job_rows):
        raise ProtocolError("E7 Colab job positions must be unique.")
    job_path = output_dir / f"e7-{phase}.jsonl"
    write_jsonl(job_path, job_rows)
    metadata = {
        "job_schema": "pku-saferlhf.e7-colab-job.v1",
        "created_at_utc": utc_now(),
        "phase": phase,
        "expected_response_positions": len(job_rows),
        "job_sha256": sha256_file(job_path),
        "model": {
            "model_id": config["artefact"]["model_id"],
            "revision": config["artefact"]["candidate_revision"],
            "conversation_template": config["artefact"]["documented_conversation_template"],
            "score_direction": "higher_is_unsafe",
            "score_direction_source": (
                "PKU safe_rlhf CostTrainer source at the vendored commit: safer samples "
                "have lower costs and safe samples have negative costs."
            ),
        },
        "runtime": {
            "max_input_tokens": int(config["runtime"]["max_input_tokens"]),
            "precision": "torch.bfloat16",
            "tokenizer_padding_side": "right",
        },
        "sampling": {
            "pilot_selection_seed": config["pilot"]["selection_seed"] if phase == "pilot" else None,
            "primary_sampling_manifest": PRIMARY_SAMPLE.name if phase == "primary" else None,
            "primary_sampling_manifest_sha256": sha256_file(PRIMARY_SAMPLE) if phase == "primary" else None,
        },
        "provenance": {
            "e7_config_sha256": sha256_file(CONFIG_PATH),
            "p0_snapshot_sha256": sha256_file(CPU_RESULTS / "p0_snapshot.json"),
            "native_audit_sha256": sha256_file(CPU_RESULTS / "native_audit.json"),
            "bundle_builder_sha256": sha256_file(SCRIPT),
        },
        "private_output_policy": (
            "This job contains selected prompt and response text. Keep it private; do not commit, "
            "share, or include it in dissertation materials. The scorer writes text-free score records."
        ),
    }
    write_json(output_dir / f"e7-{phase}.meta.json", metadata)


def copy_vendor_source(bundle_dir: Path) -> None:
    """Vendor the exact score-model code without unrelated training dependencies."""

    import urllib.request

    manifest: dict[str, dict[str, str]] = {}
    for relative, contents in VENDOR_FILES.items():
        target = bundle_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(contents, encoding="utf-8")
        manifest[relative] = {"origin": "bundle shim", "sha256": sha256_file(target)}
    for relative in UPSTREAM_VENDOR_PATHS:
        url = (
            "https://raw.githubusercontent.com/PKU-Alignment/safe-rlhf/"
            f"{SAFE_RLHF_REVISION}/{relative}"
        )
        target = bundle_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                contents = response.read()
        except OSError as error:
            raise RuntimeError(f"Could not retrieve source-locked file {url}") from error
        target.write_bytes(contents)
        manifest[relative] = {"origin": url, "sha256": sha256_file(target)}
    write_json(
        bundle_dir / "vendor_source_manifest.json",
        {
            "safe_rlhf_revision": SAFE_RLHF_REVISION,
            "files": manifest,
            "purpose": (
                "Exact score-model implementation used by the model card, vendored without the "
                "unneeded safe_rlhf training/deepspeed stack for Colab inference."
            ),
        },
    )


def write_colab_common(path: Path) -> None:
    path.write_text(
        '"""Minimal helpers for the standalone E7 aggregate script."""\n\n'
        "from __future__ import annotations\n\n"
        "import json\n"
        "from datetime import datetime, timezone\n"
        "from pathlib import Path\n\n"
        "def utc_now() -> str:\n"
        "    return datetime.now(timezone.utc).isoformat()\n\n"
        "def read_jsonl(path: Path):\n"
        "    if not path.exists():\n"
        "        return\n"
        "    with path.open(encoding='utf-8') as handle:\n"
        "        for line in handle:\n"
        "            if line.strip():\n"
        "                yield json.loads(line)\n\n"
        "def write_json(path: Path, payload: dict) -> None:\n"
        "    path.parent.mkdir(parents=True, exist_ok=True)\n"
        "    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\\n', encoding='utf-8')\n"
        ,
        encoding="utf-8",
    )


def write_readme(path: Path) -> None:
    path.write_text(
        """# E7 Beaver cost-model Colab bundle

This is a private execution package for the dissertation's E7 probe. It has
two frozen jobs: `e7-pilot.jsonl` (500 pairs / 1,000 response positions) and
`e7-primary.jsonl` (the shared stratified 4,000-pair / 8,000-position sample).
The two job files contain selected prompts and responses. Do not upload them
to GitHub, attach them to the dissertation, or share this archive publicly.

`e7_cost_model_standalone.py` follows the model-card input format:

    BEGINNING OF CONVERSATION: USER: {prompt} ASSISTANT: {response}

The model is pinned to `26bf7161f09fee958ae64c8b4bb70fb420f7ba39`. The
score direction is frozen as `higher_is_unsafe`: in PKU's `CostTrainer`, safer
samples receive lower costs and safe samples have negative costs. The pilot
still reports both AUROC orientations as a runtime diagnostic.

The bundle vendors only the score-model files from the cited `safe-rlhf`
commit. This avoids installing unrelated DeepSpeed training dependencies; the
vendor manifest records upstream URLs and hashes.

After a complete primary run, retain/download the score JSONL, its manifest,
the completeness report, and `e7-primary-aggregate/`. They have no prompt or
response text. Do not download or archive the job files outside your private
workspace.
""",
        encoding="utf-8",
    )


def copy_required_files(bundle_dir: Path) -> None:
    shutil.copy2(COLAB_DIR / "e7_cost_model_standalone.py", bundle_dir)
    shutil.copy2(EXPERIMENTS / "gpu_api" / "scripts" / "e7_aggregate.py", bundle_dir)
    write_colab_common(bundle_dir / "common.py")
    (bundle_dir / "requirements-colab.txt").write_text(
        "# Torch is supplied by the CUDA Colab runtime.\n"
        # The vendored score model subclasses the Llama classes, which are
        # stable across the 4.x line but were refactored in 5.x.  Pinning an
        # exact 4.37.2 fails on the current Python 3.12 / torch 2.11 runtime,
        # which postdates it; pinning the major line keeps the API the vendored
        # source expects while remaining installable.  The resolved version is
        # recorded in every run manifest, so the executed version is auditable
        # even though it is not fixed in advance.
        # The vendored score model imports two private docstring constants,
        # _CONFIG_FOR_DOC and LLAMA_INPUTS_DOCSTRING, from the Transformers
        # Llama module.  Both were removed in the later docstring refactor, so
        # the pin must stay inside the release window that still exports them.
        # The notebook asserts those two symbols directly rather than trusting
        # the version string, and the resolved version is recorded in every run
        # manifest.
        "transformers==4.45.2\n"
        "accelerate==0.34.2\n"
        "huggingface_hub>=0.20,<1\n"
        # Colab's Python 3.12 image already ships a NumPy 2 ABI. Keeping this
        # major version prevents a forced downgrade from leaving preloaded
        # binary extensions (such as numpy.random) incompatible in-memory.
        "numpy>=2.0,<2.5\n"
        "sentencepiece>=0.1.99\n"
        "safetensors>=0.4\n",
        encoding="utf-8",
    )
    write_readme(bundle_dir / "README.md")
    copy_vendor_source(bundle_dir)


def archive_directory(bundle_dir: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz") as archive:
        for source in sorted(bundle_dir.rglob("*")):
            if source.is_file():
                archive.add(source, arcname=source.relative_to(bundle_dir))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    validate_cpu_provenance()
    config = read_json(CONFIG_PATH)
    with tempfile.TemporaryDirectory(prefix="e7_colab_bundle_") as temporary:
        bundle_dir = Path(temporary)
        pilot = response_tasks(
            deterministic_pilot_rows(config["pilot"]["pairs"], config["pilot"]["selection_seed"])
        )
        primary = response_tasks(primary_sample_rows(PRIMARY_SAMPLE))
        make_job(pilot, "pilot", config, bundle_dir)
        make_job(primary, "primary", config, bundle_dir)
        copy_required_files(bundle_dir)
        archive_directory(bundle_dir, arguments.output)
    print(
        json.dumps(
            {
                "bundle": str(arguments.output),
                "bundle_sha256": sha256_file(arguments.output),
                "pilot_response_positions": len(pilot),
                "primary_response_positions": len(primary),
                "warning": "Private archive: it contains selected prompts and responses.",
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
