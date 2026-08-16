"""PLSC fitting, significance selection, controls, and inverse projections."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from mouse_run_run.artifacts.rollout import RolloutArtifact
from mouse_run_run.core.progress import ProgressSink, ProgressTracker, TerminalProgress
from mouse_run_run.core.types import AgentId


PLSC_FORMAT = "mrr-plsc-v2"
NullModel = Literal["time_shuffle", "episode_shuffle", "circular_shift"]


@dataclass(frozen=True)
class FixedRank:
    rank: int

    def as_dict(self) -> dict[str, Any]:
        return {"kind": "fixed_rank", "rank": self.rank}


@dataclass(frozen=True)
class PermutationThreshold:
    alpha: float = 0.05
    permutations: int = 200
    null_model: NullModel = "episode_shuffle"
    seed: int = 0
    min_shift: int = 10

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": "permutation_threshold",
            "alpha": self.alpha,
            "permutations": self.permutations,
            "null_model": self.null_model,
            "seed": self.seed,
            "min_shift": self.min_shift,
        }


SelectionRule = FixedRank | PermutationThreshold


@dataclass(frozen=True)
class PLSCConfig:
    agent_a: AgentId
    agent_b: AgentId
    site: str = "hidden"
    selection: SelectionRule = PermutationThreshold()
    control_rank: int | None = None
    control_seed: int = 0


@dataclass(frozen=True)
class StandardizedSubspace:
    agent_id: AgentId
    site: str
    mean: torch.Tensor
    scale: torch.Tensor
    basis: torch.Tensor

    def standardize(self, values: torch.Tensor) -> torch.Tensor:
        self._validate_values(values)
        return (values - self.mean.to(values)) / self.scale.to(values)

    def restore(self, standardized: torch.Tensor) -> torch.Tensor:
        self._validate_values(standardized)
        return self.mean.to(standardized) + self.scale.to(standardized) * standardized

    def project_standardized(self, values: torch.Tensor) -> torch.Tensor:
        standardized = self.standardize(values)
        basis = self.basis.to(values)
        return (standardized @ basis) @ basis.T

    def remove(self, values: torch.Tensor) -> torch.Tensor:
        standardized = self.standardize(values)
        basis = self.basis.to(values)
        return self.restore(standardized - (standardized @ basis) @ basis.T)

    def keep(self, values: torch.Tensor) -> torch.Tensor:
        return self.restore(self.project_standardized(values))

    def _validate_values(self, values: torch.Tensor) -> None:
        if values.shape[-1] != self.mean.numel():
            raise ValueError(
                f"activation width {values.shape[-1]} != subspace width {self.mean.numel()}"
            )
        if self.basis.ndim != 2 or self.basis.shape[0] != self.mean.numel():
            raise ValueError("subspace basis shape is invalid")


@dataclass(frozen=True)
class FittedPLSC:
    config: PLSCConfig
    means: dict[AgentId, torch.Tensor]
    scales: dict[AgentId, torch.Tensor]
    bases: dict[AgentId, torch.Tensor]
    singular_values: torch.Tensor
    null_values: torch.Tensor
    thresholds: torch.Tensor
    selected: torch.Tensor
    p_values: torch.Tensor
    control_bases: dict[str, dict[AgentId, torch.Tensor]]
    samples: int
    episodes: int

    @property
    def selected_indices(self) -> torch.Tensor:
        return torch.nonzero(self.selected, as_tuple=False).flatten()

    def subspace(self, agent_id: AgentId, basis_name: str = "shared") -> StandardizedSubspace:
        if agent_id not in self.bases:
            raise ValueError(f"PLSC has no agent {agent_id!r}")
        if basis_name == "shared":
            basis = self.bases[agent_id][:, self.selected]
        else:
            try:
                basis = self.control_bases[basis_name][agent_id]
            except KeyError:
                raise ValueError(f"PLSC has no control basis {basis_name!r}") from None
        if basis.shape[1] < 1:
            raise ValueError(f"subspace {basis_name!r} has no selected dimensions")
        return StandardizedSubspace(
            agent_id=agent_id,
            site=self.config.site,
            mean=self.means[agent_id],
            scale=self.scales[agent_id],
            basis=basis,
        )


def fit_plsc(
    rollout: RolloutArtifact,
    config: PLSCConfig,
    *,
    progress: ProgressSink | None = None,
) -> FittedPLSC:
    if config.agent_a == config.agent_b:
        raise ValueError("PLSC requires two different agents")
    for agent_id in (config.agent_a, config.agent_b):
        if agent_id not in rollout.trajectory.agents:
            raise ValueError(f"rollout has no agent {agent_id!r}")
    left_series = rollout.trajectory.agents[config.agent_a].activations.get(config.site)
    right_series = rollout.trajectory.agents[config.agent_b].activations.get(config.site)
    if left_series is None or right_series is None:
        raise ValueError(f"PLSC activation site {config.site!r} is missing")
    left_series = left_series.detach().cpu()
    right_series = right_series.detach().cpu()
    if left_series.shape[:2] != right_series.shape[:2]:
        raise ValueError("PLSC agent activations are not time-aligned")
    mask = rollout.trajectory.active.detach().cpu()
    left = left_series[mask].double()
    right = right_series[mask].double()
    if left.shape[0] < 2:
        raise ValueError("PLSC requires at least two valid paired samples")
    if not torch.isfinite(left).all() or not torch.isfinite(right).all():
        raise ValueError("PLSC input contains non-finite activations")
    left_z, left_mean, left_scale = _standardize(left)
    right_z, right_mean, right_scale = _standardize(right)
    cross = left_z.T @ right_z / (left_z.shape[0] - 1)
    left_basis, singular, right_vh = torch.linalg.svd(cross, full_matrices=False)
    right_basis = right_vh.T
    rank = singular.numel()

    tracker: ProgressTracker | None = None
    if isinstance(config.selection, FixedRank):
        if not 1 <= config.selection.rank <= rank:
            raise ValueError(f"fixed PLSC rank must be in [1, {rank}]")
        selected = torch.arange(rank) < config.selection.rank
        null = torch.empty(0, rank, dtype=torch.float64)
        thresholds = torch.empty(0, dtype=torch.float64)
        p_values = torch.full((rank,), torch.nan, dtype=torch.float64)
    else:
        _validate_threshold(config.selection)
        tracker = ProgressTracker(
            "analyze:plsc-null",
            config.selection.permutations,
            progress or TerminalProgress(),
        )
        null = _permutation_null(
            left_series.double(),
            right_series.double(),
            mask,
            config.selection,
            tracker,
        )
        thresholds = torch.quantile(null, 1.0 - config.selection.alpha, dim=0)
        selected = singular > thresholds
        p_values = (1 + (null >= singular).sum(dim=0)).double() / (
            config.selection.permutations + 1
        )
        tracker.emit(config.selection.permutations, state="completed")

    control_bases: dict[str, dict[AgentId, torch.Tensor]] = {}
    if config.control_rank is not None:
        shared_rank = int(selected.sum())
        available = min(left.shape[1], right.shape[1]) - shared_rank
        if not 1 <= config.control_rank <= available:
            raise ValueError("control_rank must fit inside both unique complements")
        generator = torch.Generator().manual_seed(config.control_seed)
        control_bases = {
            "top_unique": {
                config.agent_a: _top_unique(left_z, left_basis[:, selected], config.control_rank),
                config.agent_b: _top_unique(right_z, right_basis[:, selected], config.control_rank),
            },
            "random_unique": {
                config.agent_a: _random_unique(
                    left.shape[1], left_basis[:, selected], config.control_rank, generator
                ),
                config.agent_b: _random_unique(
                    right.shape[1], right_basis[:, selected], config.control_rank, generator
                ),
            },
        }
    return FittedPLSC(
        config=config,
        means={config.agent_a: left_mean, config.agent_b: right_mean},
        scales={config.agent_a: left_scale, config.agent_b: right_scale},
        bases={config.agent_a: left_basis, config.agent_b: right_basis},
        singular_values=singular,
        null_values=null,
        thresholds=thresholds,
        selected=selected,
        p_values=p_values,
        control_bases=control_bases,
        samples=left.shape[0],
        episodes=rollout.trajectory.batch_size,
    )


def save_fitted_plsc(path: Path, fitted: FittedPLSC, *, rollout: Path) -> None:
    if path.exists():
        raise FileExistsError(f"PLSC output already exists: {path}")
    path.mkdir(parents=True)
    tensors: dict[str, torch.Tensor] = {
        "singular_values": fitted.singular_values,
        "null_values": fitted.null_values,
        "thresholds": fitted.thresholds,
        "selected": fitted.selected,
        "p_values": fitted.p_values,
    }
    for agent_id in fitted.bases:
        tensors[f"agent.{agent_id}.mean"] = fitted.means[agent_id]
        tensors[f"agent.{agent_id}.scale"] = fitted.scales[agent_id]
        tensors[f"agent.{agent_id}.basis"] = fitted.bases[agent_id]
    for name, agents in fitted.control_bases.items():
        for agent_id, basis in agents.items():
            tensors[f"control.{name}.{agent_id}"] = basis
    save_file(
        {key: value.detach().cpu().contiguous() for key, value in tensors.items()},
        str(path / "subspace.safetensors"),
        metadata={"format": PLSC_FORMAT},
    )
    threshold_values = (
        fitted.thresholds.tolist()
        if fitted.thresholds.numel()
        else [None] * fitted.singular_values.numel()
    )
    rows = [
        {
            "rank": index + 1,
            "singular_value": float(fitted.singular_values[index]),
            "threshold": threshold_values[index],
            "selected": bool(fitted.selected[index]),
            "p_value": (
                None if torch.isnan(fitted.p_values[index]) else float(fitted.p_values[index])
            ),
        }
        for index in range(fitted.singular_values.numel())
    ]
    table = pa.Table.from_pylist(rows).replace_schema_metadata(
        {b"mrr_format": PLSC_FORMAT.encode()}
    )
    pq.write_table(table, path / "spectrum.parquet", compression="zstd")
    manifest = {
        "format": PLSC_FORMAT,
        "rollout": str(rollout),
        "agent_a": str(fitted.config.agent_a),
        "agent_b": str(fitted.config.agent_b),
        "site": fitted.config.site,
        "selection": fitted.config.selection.as_dict(),
        "control_rank": fitted.config.control_rank,
        "control_seed": fitted.config.control_seed,
        "samples": fitted.samples,
        "episodes": fitted.episodes,
        "selected_indices": fitted.selected_indices.tolist(),
        "control_bases": sorted(fitted.control_bases),
    }
    (path / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_fitted_plsc(path: Path) -> FittedPLSC:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != PLSC_FORMAT:
        raise ValueError(f"{path} is not a {PLSC_FORMAT} artifact")
    tensor_path = path / "subspace.safetensors"
    with safe_open(str(tensor_path), framework="pt", device="cpu") as handle:
        if (handle.metadata() or {}).get("format") != PLSC_FORMAT:
            raise ValueError("PLSC tensor payload format is invalid")
    tensors = load_file(str(tensor_path), device="cpu")
    agent_a = AgentId(manifest["agent_a"])
    agent_b = AgentId(manifest["agent_b"])
    selection_payload = manifest["selection"]
    if selection_payload["kind"] == "fixed_rank":
        selection: SelectionRule = FixedRank(rank=int(selection_payload["rank"]))
    elif selection_payload["kind"] == "permutation_threshold":
        selection = PermutationThreshold(
            alpha=float(selection_payload["alpha"]),
            permutations=int(selection_payload["permutations"]),
            null_model=selection_payload["null_model"],
            seed=int(selection_payload["seed"]),
            min_shift=int(selection_payload["min_shift"]),
        )
    else:
        raise ValueError(f"unknown PLSC selection kind {selection_payload['kind']!r}")
    controls: dict[str, dict[AgentId, torch.Tensor]] = {}
    for name in manifest["control_bases"]:
        controls[name] = {
            agent_id: tensors[f"control.{name}.{agent_id}"]
            for agent_id in (agent_a, agent_b)
        }
    return FittedPLSC(
        config=PLSCConfig(
            agent_a=agent_a,
            agent_b=agent_b,
            site=manifest["site"],
            selection=selection,
            control_rank=manifest["control_rank"],
            control_seed=int(manifest["control_seed"]),
        ),
        means={agent_id: tensors[f"agent.{agent_id}.mean"] for agent_id in (agent_a, agent_b)},
        scales={agent_id: tensors[f"agent.{agent_id}.scale"] for agent_id in (agent_a, agent_b)},
        bases={agent_id: tensors[f"agent.{agent_id}.basis"] for agent_id in (agent_a, agent_b)},
        singular_values=tensors["singular_values"],
        null_values=tensors["null_values"],
        thresholds=tensors["thresholds"],
        selected=tensors["selected"].bool(),
        p_values=tensors["p_values"],
        control_bases=controls,
        samples=int(manifest["samples"]),
        episodes=int(manifest["episodes"]),
    )


def _standardize(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = values.mean(dim=0)
    scale = values.std(dim=0, correction=0)
    if bool((scale < 1e-12).any()):
        raise ValueError("PLSC requires non-constant activation dimensions")
    return (values - mean) / scale, mean, scale


def _validate_threshold(rule: PermutationThreshold) -> None:
    if not 0.0 < rule.alpha < 1.0:
        raise ValueError("PLSC alpha must be in (0, 1)")
    if rule.permutations < 1:
        raise ValueError("PLSC permutations must be positive")
    if rule.null_model not in ("time_shuffle", "episode_shuffle", "circular_shift"):
        raise ValueError(f"unknown PLSC null model {rule.null_model!r}")
    if rule.min_shift < 1:
        raise ValueError("PLSC min_shift must be positive")


def _permutation_null(
    left_series: torch.Tensor,
    right_series: torch.Tensor,
    mask: torch.Tensor,
    rule: PermutationThreshold,
    tracker: ProgressTracker,
) -> torch.Tensor:
    left, _, _ = _standardize(left_series[mask])
    right, _, _ = _standardize(right_series[mask])
    generator = torch.Generator().manual_seed(rule.seed)
    rank = min(left.shape[1], right.shape[1])
    result = torch.empty(rule.permutations, rank, dtype=torch.float64)
    time, episodes = mask.shape
    if rule.null_model != "time_shuffle" and not bool(mask.all()):
        raise ValueError(f"{rule.null_model} requires equal full-length episodes")
    right_episode = right_series.double()
    for index in range(rule.permutations):
        if rule.null_model == "time_shuffle":
            permuted = right[torch.randperm(right.shape[0], generator=generator)]
        elif rule.null_model == "episode_shuffle":
            if episodes < 2:
                raise ValueError("episode_shuffle requires at least two episodes")
            order = torch.randperm(episodes, generator=generator)
            candidate = right_episode[:, order].reshape(time * episodes, -1)
            permuted, _, _ = _standardize(candidate)
        else:
            if time <= 2 * rule.min_shift:
                raise ValueError(
                    "circular_shift horizon must be greater than twice min_shift"
                )
            blocks = []
            for episode in range(episodes):
                shift = int(
                    torch.randint(
                        rule.min_shift,
                        time - rule.min_shift + 1,
                        (),
                        generator=generator,
                    )
                )
                blocks.append(torch.roll(right_episode[:, episode], shift, dims=0))
            candidate = torch.stack(blocks, dim=1).reshape(time * episodes, -1)
            permuted, _, _ = _standardize(candidate)
        result[index] = torch.linalg.svdvals(left.T @ permuted / (left.shape[0] - 1))
        completed = index + 1
        interval = max(rule.permutations // 20, 1)
        if completed % interval == 0 or completed == rule.permutations:
            tracker.emit(completed)
    return result


def _top_unique(
    standardized: torch.Tensor,
    shared: torch.Tensor,
    rank: int,
) -> torch.Tensor:
    projected = _unique_projector(standardized.shape[1], shared)
    covariance = standardized.T @ standardized / (standardized.shape[0] - 1)
    values, vectors = torch.linalg.eigh(projected @ covariance @ projected)
    return _orthogonalize(vectors[:, torch.argsort(values, descending=True)[:rank]], shared)


def _random_unique(
    width: int,
    shared: torch.Tensor,
    rank: int,
    generator: torch.Generator,
) -> torch.Tensor:
    candidates = torch.randn(width, rank, dtype=torch.float64, generator=generator)
    return _orthogonalize(candidates, shared)


def _unique_projector(width: int, shared: torch.Tensor) -> torch.Tensor:
    identity = torch.eye(width, dtype=shared.dtype, device=shared.device)
    return identity - shared @ shared.T


def _orthogonalize(candidates: torch.Tensor, shared: torch.Tensor) -> torch.Tensor:
    residual = candidates - shared @ (shared.T @ candidates)
    basis, triangular = torch.linalg.qr(residual, mode="reduced")
    if bool((triangular.diag().abs() < 1e-10).any()):
        raise ValueError("control basis candidates are rank deficient")
    if not torch.allclose(basis.T @ basis, torch.eye(basis.shape[1], dtype=basis.dtype), atol=1e-9, rtol=0.0):
        raise ValueError("control basis is not orthonormal")
    if shared.numel() and not torch.allclose(
        shared.T @ basis,
        torch.zeros(shared.shape[1], basis.shape[1], dtype=basis.dtype),
        atol=1e-9,
        rtol=0.0,
    ):
        raise ValueError("control basis overlaps the shared subspace")
    return basis
