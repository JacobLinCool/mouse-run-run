"""Shared analysis result and output helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from safetensors.torch import save_file

from mouse_run_run.artifacts.rollout import RolloutArtifact
from mouse_run_run.core.progress import ProgressSink


ANALYSIS_FORMAT = "mrr-analysis-v2"


@dataclass(frozen=True)
class AnalysisResult:
    output: Path
    manifest: dict[str, Any]
    table: pa.Table
    tensors: dict[str, torch.Tensor]


class Analysis(Protocol):
    @property
    def name(self) -> str: ...

    def run(
        self,
        rollout: RolloutArtifact,
        output: Path,
        *,
        progress: ProgressSink | None = None,
    ) -> AnalysisResult: ...


def write_analysis(
    output: Path,
    *,
    name: str,
    rollout: RolloutArtifact,
    config: Mapping[str, Any],
    table: pa.Table,
    tensors: Mapping[str, torch.Tensor] | None = None,
) -> AnalysisResult:
    if output.exists():
        raise FileExistsError(f"analysis output already exists: {output}")
    output.mkdir(parents=True)
    schema_metadata = dict(table.schema.metadata or {})
    schema_metadata[b"mrr_format"] = ANALYSIS_FORMAT.encode()
    table = table.replace_schema_metadata(schema_metadata)
    pq.write_table(table, output / "results.parquet", compression="zstd")
    payload = {
        key: value.detach().cpu().contiguous()
        for key, value in ({} if tensors is None else tensors).items()
    }
    if payload:
        save_file(
            payload,
            str(output / "tensors.safetensors"),
            metadata={"format": ANALYSIS_FORMAT},
        )
    manifest = {
        "format": ANALYSIS_FORMAT,
        "analysis": name,
        "rollout": str(rollout.path),
        "config": dict(config),
        "rows": table.num_rows,
        "tensor_keys": sorted(payload),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return AnalysisResult(
        output=output,
        manifest=manifest,
        table=table,
        tensors=payload,
    )
