"""Zhang et al. (2025) PLSC recipe with its two-statistic null gate."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from mouse_run_run.analyses.plsc import StandardizedSubspace
from mouse_run_run.artifacts.rollout import RolloutArtifact
from mouse_run_run.core.progress import ProgressSink, ProgressTracker, TerminalProgress
from mouse_run_run.core.types import AgentId


PAPER_PLSC_FORMAT = "mrr-zhang-2025-plsc-v1"


@dataclass(frozen=True)
class PaperPLSCConfig:
    agent_a: AgentId
    agent_b: AgentId
    site: str = "hidden"
    permutations: int = 2_000
    percentile: float = 0.975
    min_shift: int = 60
    seed: int = 0
    null_batch_size: int = 4
    compute_device: str = "cpu"

    def validate(self, horizon: int) -> None:
        if self.agent_a == self.agent_b:
            raise ValueError("paper PLSC requires two different agents")
        if self.permutations < 1 or self.null_batch_size < 1:
            raise ValueError("paper PLSC permutation counts must be positive")
        if not 0.5 < self.percentile < 1.0:
            raise ValueError("paper PLSC percentile must be in (0.5, 1)")
        if self.min_shift < 1 or horizon <= 2 * self.min_shift:
            raise ValueError("paper PLSC horizon must exceed twice min_shift")
        if self.compute_device not in ("cpu", "cuda"):
            raise ValueError("paper PLSC compute_device must be cpu or cuda")
        if self.compute_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("paper PLSC requested CUDA but CUDA is unavailable")

    def as_dict(self) -> dict[str, Any]:
        return {
            "agent_a": str(self.agent_a),
            "agent_b": str(self.agent_b),
            "site": self.site,
            "permutations": self.permutations,
            "percentile": self.percentile,
            "min_shift": self.min_shift,
            "seed": self.seed,
            "null_batch_size": self.null_batch_size,
            "compute_device": self.compute_device,
        }


@dataclass(frozen=True)
class PaperPLSCResult:
    config: PaperPLSCConfig
    means: dict[AgentId, torch.Tensor]
    scales: dict[AgentId, torch.Tensor]
    bases: dict[AgentId, torch.Tensor]
    covariance: torch.Tensor
    projected_score_pcc: torch.Tensor
    null_covariance: torch.Tensor
    null_projected_score_pcc: torch.Tensor
    covariance_threshold: torch.Tensor
    pcc_threshold: torch.Tensor
    passes_both: torch.Tensor
    selected: torch.Tensor
    covariance_p_values: torch.Tensor
    pcc_p_values: torch.Tensor
    samples: int
    episodes: int
    episode_mask: torch.Tensor
    sample_mask: torch.Tensor

    @property
    def significant_count(self) -> int:
        return int(self.selected.sum())

    @property
    def plsc1_pcc(self) -> float:
        return float(self.projected_score_pcc[0])

    @property
    def null_plsc1_pcc(self) -> float:
        return float(self.null_projected_score_pcc[:, 0].mean())

    @property
    def delta_pcc(self) -> float:
        return self.plsc1_pcc - self.null_plsc1_pcc

    def subspace(self, agent_id: AgentId, *, rank: int | None = None) -> StandardizedSubspace:
        if agent_id not in self.bases:
            raise ValueError(f"paper PLSC has no agent {agent_id!r}")
        selected_rank = self.significant_count if rank is None else rank
        if not 1 <= selected_rank <= self.bases[agent_id].shape[1]:
            raise ValueError("paper PLSC subspace rank is outside the fitted basis")
        return StandardizedSubspace(
            agent_id=agent_id,
            site=self.config.site,
            mean=self.means[agent_id],
            scale=self.scales[agent_id],
            basis=self.bases[agent_id][:, :selected_rank],
        )


def fit_paper_plsc(
    rollout: RolloutArtifact,
    config: PaperPLSCConfig,
    *,
    episode_mask: torch.Tensor | None = None,
    sample_mask: torch.Tensor | None = None,
    progress: ProgressSink | None = None,
) -> PaperPLSCResult:
    config.validate(rollout.trajectory.horizon)
    for agent_id in (config.agent_a, config.agent_b):
        if agent_id not in rollout.trajectory.agents:
            raise ValueError(f"rollout has no agent {agent_id!r}")
    left_series = rollout.trajectory.agents[config.agent_a].activations.get(config.site)
    right_series = rollout.trajectory.agents[config.agent_b].activations.get(config.site)
    if left_series is None or right_series is None:
        raise ValueError(f"paper PLSC activation site {config.site!r} is missing")
    compute_device = torch.device(config.compute_device)
    left_series = left_series.detach().to(compute_device, dtype=torch.float64)
    right_series = right_series.detach().to(compute_device, dtype=torch.float64)
    time, episodes = left_series.shape[:2]
    if right_series.shape[:2] != (time, episodes):
        raise ValueError("paper PLSC activations are not time aligned")
    included_episodes = (
        _nondegenerate_episode_mask(rollout)
        if episode_mask is None
        else episode_mask.detach().to(compute_device).bool()
    )
    included_episodes = included_episodes.to(compute_device)
    if included_episodes.shape != (episodes,) or not bool(included_episodes.any()):
        raise ValueError("paper PLSC episode mask is invalid or empty")
    active = rollout.trajectory.active.detach().to(compute_device).bool()
    if not bool(active[:, included_episodes].all()):
        raise ValueError("paper circular-shift PLSC requires full-length included episodes")
    analysis_mask = active & included_episodes.unsqueeze(0)
    if sample_mask is not None:
        if sample_mask.shape != (time, episodes):
            raise ValueError("paper PLSC sample mask shape differs from rollout")
        analysis_mask &= sample_mask.detach().to(compute_device).bool()
    if int(analysis_mask.sum()) < 3:
        raise ValueError("paper PLSC requires at least three included samples")

    left, left_mean, left_scale = _standardize(left_series[analysis_mask])
    right, right_mean, right_scale = _standardize(right_series[analysis_mask])
    cross = left.T @ right / (left.shape[0] - 1)
    left_basis, covariance, right_vh = torch.linalg.svd(cross, full_matrices=False)
    right_basis = right_vh.T
    observed_pcc = _column_pcc(left @ left_basis, right @ right_basis)
    tracker = ProgressTracker(
        "analyze:zhang-plsc-null",
        config.permutations,
        progress or TerminalProgress(),
    )
    null_covariance, null_pcc = _circular_shift_null(
        left_series,
        right_series,
        analysis_mask,
        included_episodes,
        config,
        tracker,
    )
    covariance_threshold = torch.quantile(null_covariance, config.percentile, dim=0)
    pcc_threshold = torch.quantile(null_pcc, config.percentile, dim=0)
    passes_both = (covariance > covariance_threshold) & (observed_pcc > pcc_threshold)
    selected = torch.cumprod(passes_both.to(torch.int64), dim=0).bool()
    covariance_p = (1 + (null_covariance >= covariance).sum(dim=0)).double() / (
        config.permutations + 1
    )
    pcc_p = (1 + (null_pcc >= observed_pcc).sum(dim=0)).double() / (
        config.permutations + 1
    )
    tracker.emit(config.permutations, state="completed")
    return PaperPLSCResult(
        config=config,
        means={config.agent_a: left_mean.cpu(), config.agent_b: right_mean.cpu()},
        scales={config.agent_a: left_scale.cpu(), config.agent_b: right_scale.cpu()},
        bases={config.agent_a: left_basis.cpu(), config.agent_b: right_basis.cpu()},
        covariance=covariance.cpu(),
        projected_score_pcc=observed_pcc.cpu(),
        null_covariance=null_covariance.cpu(),
        null_projected_score_pcc=null_pcc.cpu(),
        covariance_threshold=covariance_threshold.cpu(),
        pcc_threshold=pcc_threshold.cpu(),
        passes_both=passes_both.cpu(),
        selected=selected.cpu(),
        covariance_p_values=covariance_p.cpu(),
        pcc_p_values=pcc_p.cpu(),
        samples=int(analysis_mask.sum()),
        episodes=int(included_episodes.sum()),
        episode_mask=included_episodes.cpu(),
        sample_mask=analysis_mask.cpu(),
    )


def identical_action_mask(rollout: RolloutArtifact, left: AgentId, right: AgentId) -> torch.Tensor:
    return (
        rollout.trajectory.agents[left].actions
        != rollout.trajectory.agents[right].actions
    ) & rollout.trajectory.active


def temporally_shifted_random_pc_basis(
    result: PaperPLSCResult,
    rollout: RolloutArtifact,
    *,
    agent_id: AgentId,
    shared_rank: int,
    control_rank: int,
    seed: int,
) -> torch.Tensor:
    """Top PCs after independent temporal shifts, restricted to shared complement."""
    if agent_id not in (result.config.agent_a, result.config.agent_b):
        raise ValueError("control basis agent is outside fitted PLSC pair")
    own = rollout.trajectory.agents[agent_id].activations[result.config.site].double()
    standardized = (own - result.means[agent_id]) / result.scales[agent_id]
    generator = torch.Generator().manual_seed(seed)
    time, episodes, width = standardized.shape
    shifts = torch.randint(
        result.config.min_shift,
        time - result.config.min_shift + 1,
        (episodes, width),
        generator=generator,
    )
    time_indices = (
        torch.arange(time).view(time, 1, 1) - shifts.unsqueeze(0)
    ).remainder(time)
    shifted = torch.gather(standardized, 0, time_indices)
    shared = result.bases[agent_id][:, :shared_rank]
    if control_rank > width - shared_rank:
        raise ValueError("control rank does not fit shared complement")
    projector = torch.eye(width, dtype=shifted.dtype) - shared @ shared.T
    residual = shifted[result.sample_mask] @ projector
    covariance = residual.T @ residual / (residual.shape[0] - 1)
    _, eigenvectors = torch.linalg.eigh(covariance)
    candidates = eigenvectors.flip(dims=(1,))
    return _orthogonalize(candidates, shared, control_rank)


def removed_variance_fraction(
    values: torch.Tensor,
    subspace: StandardizedSubspace,
) -> float:
    standardized = subspace.standardize(values.double())
    projected = standardized @ subspace.basis
    total = standardized.square().sum()
    return float(projected.square().sum() / total.clamp_min(1e-12))


def save_paper_plsc(path: Path, result: PaperPLSCResult, *, rollout: Path) -> None:
    if path.exists():
        raise FileExistsError(f"paper PLSC output already exists: {path}")
    path.mkdir(parents=True)
    tensors = {
        "covariance": result.covariance,
        "projected_score_pcc": result.projected_score_pcc,
        "null_covariance": result.null_covariance,
        "null_projected_score_pcc": result.null_projected_score_pcc,
        "covariance_threshold": result.covariance_threshold,
        "pcc_threshold": result.pcc_threshold,
        "passes_both": result.passes_both,
        "selected": result.selected,
        "covariance_p_values": result.covariance_p_values,
        "pcc_p_values": result.pcc_p_values,
        "episode_mask": result.episode_mask,
        "sample_mask": result.sample_mask,
    }
    for agent_id in result.bases:
        tensors[f"agent.{agent_id}.mean"] = result.means[agent_id]
        tensors[f"agent.{agent_id}.scale"] = result.scales[agent_id]
        tensors[f"agent.{agent_id}.basis"] = result.bases[agent_id]
    save_file(
        {key: value.detach().cpu().contiguous() for key, value in tensors.items()},
        str(path / "subspace.safetensors"),
        metadata={"format": PAPER_PLSC_FORMAT},
    )
    rows = [
        {
            "dimension": index + 1,
            "covariance": float(result.covariance[index]),
            "projected_score_pcc": float(result.projected_score_pcc[index]),
            "covariance_threshold": float(result.covariance_threshold[index]),
            "pcc_threshold": float(result.pcc_threshold[index]),
            "passes_both": bool(result.passes_both[index]),
            "selected_contiguous_prefix": bool(result.selected[index]),
            "covariance_p_value": float(result.covariance_p_values[index]),
            "pcc_p_value": float(result.pcc_p_values[index]),
        }
        for index in range(result.covariance.numel())
    ]
    pq.write_table(
        pa.Table.from_pylist(rows).replace_schema_metadata(
            {b"mrr_format": PAPER_PLSC_FORMAT.encode()}
        ),
        path / "spectrum.parquet",
        compression="zstd",
    )
    manifest = {
        "format": PAPER_PLSC_FORMAT,
        "rollout": str(rollout),
        "config": result.config.as_dict(),
        "samples": result.samples,
        "episodes": result.episodes,
        "significant_count": result.significant_count,
        "plsc1_pcc": result.plsc1_pcc,
        "null_plsc1_pcc": result.null_plsc1_pcc,
        "delta_pcc": result.delta_pcc,
    }
    (path / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_paper_plsc(path: Path) -> PaperPLSCResult:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != PAPER_PLSC_FORMAT:
        raise ValueError(f"{path} is not a {PAPER_PLSC_FORMAT} artifact")
    tensor_path = path / "subspace.safetensors"
    with safe_open(str(tensor_path), framework="pt", device="cpu") as handle:
        if (handle.metadata() or {}).get("format") != PAPER_PLSC_FORMAT:
            raise ValueError("paper PLSC tensor payload format is invalid")
    tensors = load_file(str(tensor_path), device="cpu")
    payload = manifest["config"]
    config = PaperPLSCConfig(
        agent_a=AgentId(payload["agent_a"]),
        agent_b=AgentId(payload["agent_b"]),
        site=payload["site"],
        permutations=int(payload["permutations"]),
        percentile=float(payload["percentile"]),
        min_shift=int(payload["min_shift"]),
        seed=int(payload["seed"]),
        null_batch_size=int(payload["null_batch_size"]),
        compute_device=payload["compute_device"],
    )
    agents = (config.agent_a, config.agent_b)
    return PaperPLSCResult(
        config=config,
        means={agent: tensors[f"agent.{agent}.mean"] for agent in agents},
        scales={agent: tensors[f"agent.{agent}.scale"] for agent in agents},
        bases={agent: tensors[f"agent.{agent}.basis"] for agent in agents},
        covariance=tensors["covariance"],
        projected_score_pcc=tensors["projected_score_pcc"],
        null_covariance=tensors["null_covariance"],
        null_projected_score_pcc=tensors["null_projected_score_pcc"],
        covariance_threshold=tensors["covariance_threshold"],
        pcc_threshold=tensors["pcc_threshold"],
        passes_both=tensors["passes_both"].bool(),
        selected=tensors["selected"].bool(),
        covariance_p_values=tensors["covariance_p_values"],
        pcc_p_values=tensors["pcc_p_values"],
        samples=int(manifest["samples"]),
        episodes=int(manifest["episodes"]),
        episode_mask=tensors["episode_mask"].bool(),
        sample_mask=tensors["sample_mask"].bool(),
    )


def _circular_shift_null(
    left_series: torch.Tensor,
    right_series: torch.Tensor,
    sample_mask: torch.Tensor,
    episode_mask: torch.Tensor,
    config: PaperPLSCConfig,
    tracker: ProgressTracker,
) -> tuple[torch.Tensor, torch.Tensor]:
    left, _, _ = _standardize(left_series[sample_mask])
    rank = min(left.shape[1], right_series.shape[2])
    covariance = torch.empty(
        config.permutations,
        rank,
        dtype=torch.float64,
        device=left.device,
    )
    pcc = torch.empty_like(covariance)
    generator = torch.Generator().manual_seed(config.seed)
    completed = 0
    while completed < config.permutations:
        batch = min(config.null_batch_size, config.permutations - completed)
        candidates = torch.stack(
            [
                _shift_episodes(
                    right_series,
                    episode_mask,
                    config.min_shift,
                    generator,
                )[sample_mask]
                for _ in range(batch)
            ]
        )
        candidate_mean = candidates.mean(dim=1, keepdim=True)
        candidate_scale = candidates.std(dim=1, correction=0, keepdim=True)
        candidate_scale = torch.where(
            candidate_scale < 1e-12,
            torch.ones_like(candidate_scale),
            candidate_scale,
        )
        right = (candidates - candidate_mean) / candidate_scale
        cross = torch.einsum("nd,bne->bde", left, right) / (left.shape[0] - 1)
        left_basis, singular, right_vh = torch.linalg.svd(cross, full_matrices=False)
        left_scores = torch.einsum("nd,bdr->bnr", left, left_basis)
        right_scores = torch.einsum("bne,bre->bnr", right, right_vh)
        covariance[completed : completed + batch] = singular
        pcc[completed : completed + batch] = _batched_column_pcc(
            left_scores, right_scores
        )
        completed += batch
        tracker.emit(completed)
    return covariance, pcc


def _shift_episodes(
    series: torch.Tensor,
    episode_mask: torch.Tensor,
    min_shift: int,
    generator: torch.Generator,
) -> torch.Tensor:
    result = series.clone()
    time = series.shape[0]
    for episode in torch.nonzero(episode_mask, as_tuple=False).flatten().tolist():
        shift = int(
            torch.randint(
                min_shift,
                time - min_shift + 1,
                (),
                generator=generator,
            )
        )
        result[:, episode] = torch.roll(series[:, episode], shift, dims=0)
    return result


def _standardize(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = values.mean(dim=0)
    scale = values.std(dim=0, correction=0)
    # Dead ReLU units carry zero variance and therefore zero covariance.  They
    # remain explicit zero columns instead of making an otherwise valid fit
    # undefined; a unit scale is the exact neutral transform for such columns.
    scale = torch.where(scale < 1e-12, torch.ones_like(scale), scale)
    return (values - mean) / scale, mean, scale


def _column_pcc(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left = left - left.mean(dim=0)
    right = right - right.mean(dim=0)
    numerator = (left * right).sum(dim=0)
    denominator = left.square().sum(dim=0).sqrt() * right.square().sum(dim=0).sqrt()
    return numerator / denominator.clamp_min(1e-12)


def _batched_column_pcc(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left = left - left.mean(dim=1, keepdim=True)
    right = right - right.mean(dim=1, keepdim=True)
    numerator = (left * right).sum(dim=1)
    denominator = left.square().sum(dim=1).sqrt() * right.square().sum(dim=1).sqrt()
    return numerator / denominator.clamp_min(1e-12)


def _orthogonalize(
    candidates: torch.Tensor,
    shared: torch.Tensor,
    rank: int,
) -> torch.Tensor:
    residual = candidates - shared @ (shared.T @ candidates)
    basis, triangular = torch.linalg.qr(residual, mode="reduced")
    valid = triangular.diag().abs() > 1e-10
    basis = basis[:, valid]
    if basis.shape[1] < rank:
        raise ValueError("temporally shifted random-PC candidates are rank deficient")
    basis = basis[:, :rank]
    if shared.numel() and not torch.allclose(
        shared.T @ basis,
        torch.zeros(shared.shape[1], rank, dtype=basis.dtype),
        atol=1e-9,
        rtol=0.0,
    ):
        raise ValueError("temporally shifted random-PC basis overlaps the shared subspace")
    return basis


def _nondegenerate_episode_mask(rollout: RolloutArtifact) -> torch.Tensor:
    column = "quality.degenerate"
    if column not in rollout.episodes.column_names:
        raise ValueError("rollout episode table lacks degenerate diagnostics")
    return ~torch.tensor(rollout.episodes[column].to_pylist(), dtype=torch.bool)
