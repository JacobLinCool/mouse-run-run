"""Shared conventions for the analysis scripts in this directory.

- Repo-root-anchored default paths, so the scripts behave the same from any
  working directory.
- The canonical architecture -> experiment mapping for the cross-architecture
  study (the single source of truth for the four ``cross_arch_*`` scripts).
- The MANIFEST.json convention used by the older report scripts: input and
  output content hashes plus full provenance, written atomically next to the
  artifact it certifies.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mouse_run_run.provenance import collect_provenance, hash_file, write_json_atomic

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = REPO_ROOT / "runs"
ANALYSES_ROOT = RUNS_ROOT / "analyses"
REPORTS_ROOT = RUNS_ROOT / "reports"
TABLES_ROOT = RUNS_ROOT / "tables"

# Architecture label -> experiment name, in canonical display order (matches
# mouse_run_run.viewer_payloads.ARCHITECTURE_LABELS / ARCHITECTURE_ORDER).
CROSS_ARCH_EXPERIMENTS = {
    "RNN": "mouse-run-run-1",
    "MLP": "mouse-run-run-2-mlp",
    "SSM": "mouse-run-run-2-ssm",
    "Transformer": "mouse-run-run-2-transformer",
}


def manifest_sidecar(artifact: Path) -> Path:
    """Manifest path for a single-file artifact: ``<stem>.MANIFEST.json``
    next to it (several scripts share the runs/analyses root, so a bare
    MANIFEST.json there would collide)."""
    return artifact.with_name(f"{artifact.stem}.MANIFEST.json")


def write_manifest(
    manifest_path: Path,
    *,
    script: Path,
    inputs: Iterable[Path],
    outputs: Mapping[str, Any],
    extra: Mapping[str, Any] | None = None,
) -> None:
    """Write a MANIFEST.json recording the exact input and output files with
    their content hashes plus full provenance."""
    input_paths = sorted({Path(item) for item in inputs})
    output_paths = sorted({Path(item) for item in _flatten_outputs(outputs)})
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "inputs": [str(path) for path in input_paths],
        "input_hashes": {str(path): hash_file(path) for path in input_paths},
        "outputs": {key: value for key, value in outputs.items()},
        "output_hashes": {
            str(path): hash_file(path) for path in output_paths if path.is_file()
        },
        "analysis_script": str(Path(script).resolve()),
        "provenance": collect_provenance(cwd=REPO_ROOT),
    }
    if extra:
        manifest.update(extra)
    write_json_atomic(manifest_path, manifest)


def _flatten_outputs(outputs: Mapping[str, Any]) -> Iterator[Path]:
    for value in outputs.values():
        if isinstance(value, str | Path):
            yield Path(value)
        elif isinstance(value, list | tuple):
            for item in value:
                yield Path(item)
