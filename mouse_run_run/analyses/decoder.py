"""Diagnostic-only episode-grouped linear decoding of partner events."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from mouse_run_run.analyses import linear_svm
from mouse_run_run.artifacts.rollout import RolloutArtifact
from mouse_run_run.core.progress import ProgressSink, ProgressTracker, TerminalProgress
from mouse_run_run.core.types import AgentId


DECODER_FORMAT = "mrr-zhang-2025-decoder-v1"


@dataclass(frozen=True)
class DecoderConfig:
    folds: int = 5
    shuffled_controls: int = 200
    seed: int = 0
    site: str = "hidden"
    solver: str = "batched"
    """``batched`` solves every label set at once; ``sklearn`` fits them one by one."""

    def validate(self) -> None:
        if self.folds < 2 or self.shuffled_controls < 1:
            raise ValueError("decoder folds and shuffled controls must be positive")
        if self.solver not in ("batched", "sklearn"):
            raise ValueError("decoder solver must be batched or sklearn")


def run_diagnostic_decoder(
    rollout: RolloutArtifact,
    output: Path,
    *,
    config: DecoderConfig = DecoderConfig(),
    progress: ProgressSink | None = None,
) -> pa.Table:
    config.validate()
    if output.exists():
        raise FileExistsError(f"decoder output already exists: {output}")
    output.mkdir(parents=True)
    episode_mask = _nondegenerate_episode_mask(rollout)
    rows: list[dict[str, object]] = []
    jobs = [
        (agent_id, target, source)
        for agent_id in rollout.trajectory.agent_ids
        for target, source in _targets(agent_id).items()
    ]
    tracker = ProgressTracker(
        "analyze:diagnostic-decoder",
        len(jobs),
        progress or TerminalProgress(),
    )
    for index, (agent_id, target, source) in enumerate(jobs):
        rows.append(
            _decode_one(
                rollout,
                agent_id=agent_id,
                target=target,
                event_source=source,
                episode_mask=episode_mask,
                config=config,
            )
        )
        tracker.emit(index + 1)
    table = pa.Table.from_pylist(rows).replace_schema_metadata(
        {b"mrr_format": DECODER_FORMAT.encode()}
    )
    pq.write_table(table, output / "results.parquet", compression="zstd")
    tracker.emit(len(jobs), state="completed")
    return table


def episode_grouped_splits(
    labels: torch.Tensor,
    groups: torch.Tensor,
    *,
    folds: int,
    seed: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    placeholder = torch.zeros(labels.numel(), 1).numpy()
    return [
        (torch.from_numpy(train), torch.from_numpy(test))
        for train, test in splitter.split(placeholder, labels.numpy(), groups.numpy())
    ]


def _decode_one(
    rollout: RolloutArtifact,
    *,
    agent_id: AgentId,
    target: str,
    event_source: str,
    episode_mask: torch.Tensor,
    config: DecoderConfig,
) -> dict[str, object]:
    hidden = rollout.trajectory.agents[agent_id].activations.get(config.site)
    labels = rollout.trajectory.events.get(event_source)
    if hidden is None or labels is None:
        return _insufficient_row(agent_id, target, event_source, "missing_input")
    mask = rollout.trajectory.active.cpu() & episode_mask.unsqueeze(0)
    values = hidden.detach().cpu()[mask].float()
    binary = labels.detach().cpu()[mask].bool().long()
    group_grid = torch.arange(rollout.trajectory.batch_size).expand(
        rollout.trajectory.horizon, -1
    )
    groups = group_grid[mask]
    class_counts = torch.bincount(binary, minlength=2)
    positive_groups = torch.unique(groups[binary == 1]).numel()
    negative_groups = torch.unique(groups[binary == 0]).numel()
    if (
        int(class_counts.min()) < config.folds
        or min(positive_groups, negative_groups) < config.folds
    ):
        return _insufficient_row(
            agent_id,
            target,
            event_source,
            "insufficient_episode_grouped_events",
            samples=int(binary.numel()),
            positives=int(class_counts[1]),
        )
    splits = episode_grouped_splits(
        binary,
        groups,
        folds=config.folds,
        seed=config.seed,
    )
    generator = torch.Generator().manual_seed(config.seed)
    label_sets = torch.stack(
        [binary]
        + [
            binary[torch.randperm(binary.numel(), generator=generator)]
            for _ in range(config.shuffled_controls)
        ]
    )
    scores = (
        _sklearn_scores(values, label_sets, splits)
        if config.solver == "sklearn"
        else _batched_scores(values, label_sets, splits)
    )
    observed = float(scores[0])
    null = scores[1:].double()
    p_value = float((1 + (null >= observed).sum()) / (config.shuffled_controls + 1))
    return {
        "agent": str(agent_id),
        "target": target,
        "event_source": event_source,
        "status": "ok",
        "reason": None,
        "samples": int(binary.numel()),
        "positives": int(class_counts[1]),
        "episodes": int(torch.unique(groups).numel()),
        "balanced_accuracy": observed,
        "shuffle_mean": float(null.mean()),
        "shuffle_p_value": p_value,
        "folds": config.folds,
        "shuffled_controls": config.shuffled_controls,
        "solver": config.solver,
    }


def _batched_scores(
    values: torch.Tensor,
    label_sets: torch.Tensor,
    splits: list[tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    """Cross-validated balanced accuracy for every label set in one pass.

    The folds and the design matrix are shared, so each fold fits all label sets
    together instead of repeating the same standardisation and factorisation per
    shuffle.
    """

    features = values.double()
    predictions = torch.empty_like(label_sets)
    degenerate = torch.zeros(label_sets.shape[0], dtype=torch.bool)
    for train, test in splits:
        train_labels = label_sets[:, train]
        degenerate |= train_labels.amin(dim=1) == train_labels.amax(dim=1)
        train_features, test_features = linear_svm.standardize(
            features[train], features[test]
        )
        train_features = linear_svm.with_intercept(train_features)
        test_features = linear_svm.with_intercept(test_features)
        weights = linear_svm.balanced_weights(train_labels, 1.0)
        coefficients = linear_svm.fit_squared_hinge(train_features, train_labels, weights)
        predictions[:, test] = linear_svm.predict(test_features, coefficients)
    scores = linear_svm.balanced_accuracy(label_sets, predictions).double()
    # A fold whose training half holds one class cannot be fitted; the score for
    # that label set is chance, exactly as the per-fit path reported.
    return torch.where(degenerate, torch.full_like(scores, 0.5), scores)


def _sklearn_scores(
    values: torch.Tensor,
    label_sets: torch.Tensor,
    splits: list[tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    return torch.tensor(
        [_cross_validated_score(values, labels, splits) for labels in label_sets],
        dtype=torch.float64,
    )


def _cross_validated_score(
    values: torch.Tensor,
    labels: torch.Tensor,
    splits: list[tuple[torch.Tensor, torch.Tensor]],
) -> float:
    predictions = torch.empty_like(labels)
    for train, test in splits:
        if torch.unique(labels[train]).numel() < 2:
            return 0.5
        model = make_pipeline(
            StandardScaler(),
            LinearSVC(class_weight="balanced", dual="auto", random_state=0),
        )
        model.fit(values[train].numpy(), labels[train].numpy())
        predictions[test] = torch.from_numpy(model.predict(values[test].numpy()))
    return float(balanced_accuracy_score(labels.numpy(), predictions.numpy()))


def _targets(agent_id: AgentId) -> dict[str, str]:
    if str(agent_id) == "chaser":
        return {"collision": "collision", "partner_escape": "explorer_escape"}
    if str(agent_id) == "explorer":
        return {"collision": "collision", "partner_approach": "chaser_approach"}
    return {"collision": "collision"}


def _insufficient_row(
    agent_id: AgentId,
    target: str,
    event_source: str,
    reason: str,
    *,
    samples: int = 0,
    positives: int = 0,
) -> dict[str, object]:
    return {
        "agent": str(agent_id),
        "target": target,
        "event_source": event_source,
        "status": "insufficient",
        "reason": reason,
        "samples": samples,
        "positives": positives,
        "episodes": 0,
        "balanced_accuracy": None,
        "shuffle_mean": None,
        "shuffle_p_value": None,
        "folds": None,
        "shuffled_controls": None,
        "solver": None,
    }


def _nondegenerate_episode_mask(rollout: RolloutArtifact) -> torch.Tensor:
    column = "quality.degenerate"
    if column not in rollout.episodes.column_names:
        raise ValueError("rollout lacks degenerate episode diagnostics")
    return ~torch.tensor(rollout.episodes[column].to_pylist(), dtype=torch.bool)
