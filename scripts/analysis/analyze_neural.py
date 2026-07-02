"""Network-activation analysis over paper analysis rollouts.

Reads rollout safetensors (the ``collect-paper-marl-rollouts`` output, which
stores per-step RNN hidden states for both agents), excludes paper-style
degenerate episodes, and produces per-checkpoint figures plus a social vs
non_social aggregate:

- PCA of pooled hidden states: variance spectrum, participation ratio, and the
  PC1-PC2 embedding colored by chaser-explorer distance.
- Peak-sorted activation rasters for an example episode.
- Collision-triggered average of hidden-state speed ``||h_t - h_{t-1}||``.
- Partner-visibility tuning: per-unit activation difference between
  partner-visible and partner-hidden steps.

Usage:

    uv run python scripts/analysis/analyze_neural.py runs/<experiment>/paper_rollouts \
        --output-root runs/reports/<experiment>/neural
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from mouse_run_run.provenance import collect_provenance, hash_file, write_json_atomic
from mouse_run_run.serialization import ROLLOUT_FORMAT, read_metadata

AGENTS = ("chaser", "explorer")
TASK_COLORS = {"social": "#d84f45", "non_social": "#2189a8"}
AGENT_COLORS = {"chaser": "#d84f45", "explorer": "#2189a8"}
FIGURE_DPI = 150
POOL_SAMPLE_LIMIT = 20_000
EVENT_WINDOW = 10
# Cohen's-d style threshold for calling a unit visibility-modulated.
MODULATION_THRESHOLD = 0.2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        type=Path,
        nargs="+",
        help="Rollout safetensors files or directories to search recursively.",
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--include-degenerate",
        action="store_true",
        help="Keep paper-style degenerate episodes instead of excluding them.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        help="Analyze at most this many rollout files (sorted by path).",
    )
    args = parser.parse_args()

    rollout_paths = _discover_rollouts(args.paths)
    if args.max_files is not None:
        rollout_paths = rollout_paths[: args.max_files]
    if not rollout_paths:
        raise SystemExit("no rollout safetensors found")

    output_root = args.output_root or _default_output_root(args.paths)
    figures_dir = output_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")

    records = []
    for path in rollout_paths:
        record = _analyze_rollout(
            path,
            figures_dir=figures_dir,
            include_degenerate=args.include_degenerate,
        )
        records.append(record)
        print(f"analyzed={path} status={record['status']}", flush=True)

    aggregate = _aggregate_by_task(records)
    aggregate_figure = _plot_aggregate(aggregate, figures_dir)

    summary = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "analysis_protocol": "neural_activation_v1",
        "include_degenerate": args.include_degenerate,
        "modulation_threshold": MODULATION_THRESHOLD,
        "event_window": EVENT_WINDOW,
        "rollouts": records,
        "aggregate_by_task": aggregate,
        "figures": [record["figure"] for record in records if record.get("figure")]
        + ([aggregate_figure] if aggregate_figure else []),
    }
    summary_path = output_root / "neural_summary.json"
    write_json_atomic(summary_path, summary)

    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "inputs": [str(path) for path in rollout_paths],
        "input_hashes": {str(path): hash_file(path) for path in rollout_paths},
        "outputs": {
            "summary": str(summary_path),
            "figures": summary["figures"],
        },
        "analysis_script": str(Path(__file__).resolve()),
        "provenance": collect_provenance(cwd=Path.cwd()),
    }
    write_json_atomic(output_root / "MANIFEST.json", manifest)
    print(json.dumps({"summary": str(summary_path), "rollouts": len(records)}, indent=2))


def _discover_rollouts(paths: list[Path]) -> list[Path]:
    found: list[Path] = []
    for path in paths:
        if path.is_file():
            candidates = [path]
        elif path.is_dir():
            candidates = sorted(path.rglob("*.safetensors"))
        else:
            raise FileNotFoundError(str(path))
        for candidate in candidates:
            try:
                metadata = read_metadata(candidate)
            except Exception:
                continue
            if metadata.get("format") == ROLLOUT_FORMAT:
                found.append(candidate)
    return sorted(set(found))


def _default_output_root(paths: list[Path]) -> Path:
    name = paths[0].stem if paths[0].is_file() else paths[0].name
    return Path("runs/reports") / name / "neural"


# ---------------------------------------------------------------------------
# Per-rollout analysis
# ---------------------------------------------------------------------------


def _analyze_rollout(
    path: Path,
    *,
    figures_dir: Path,
    include_degenerate: bool,
) -> dict[str, Any]:
    from safetensors.torch import load_file

    metadata = json.loads(read_metadata(path)["metadata"])
    tensors = {
        key.removeprefix("rollout."): value
        for key, value in load_file(str(path), device="cpu").items()
    }
    task = _rollout_task(metadata)
    record: dict[str, Any] = {
        "rollout": str(path),
        "task": task,
        "checkpoint": metadata.get("checkpoint"),
        "unit_seed": (metadata.get("checkpoint_config") or {}).get("seed"),
        "episodes": int(tensors["chaser_hidden"].shape[1]),
        "max_steps": int(tensors["chaser_hidden"].shape[0]),
    }

    degenerate = tensors.get("episode_degenerate")
    if degenerate is None or include_degenerate:
        valid = torch.ones(tensors["chaser_hidden"].shape[1], dtype=torch.bool)
    else:
        valid = ~degenerate.bool()
    record["valid_episodes"] = int(valid.sum())
    record["excluded_episodes"] = int((~valid).sum())
    if not bool(valid.any()):
        record["status"] = "all_episodes_degenerate"
        return record

    steps = tensors["chaser_hidden"].shape[0]
    distance = _state_series(tensors["distance"], steps)[:, valid]
    collision = tensors["collision"].bool()[:, valid]

    agents: dict[str, Any] = {}
    panels: dict[str, Any] = {}
    for agent in AGENTS:
        hidden = tensors[f"{agent}_hidden"][:, valid].float()  # (T, Ev, H)
        visible = _state_series(tensors[f"{agent}_partner_visible"], steps)[:, valid].bool()
        stats, panel = _analyze_agent(hidden, visible, distance, collision)
        agents[agent] = stats
        panels[agent] = panel
    record["agents"] = agents

    figure_path = figures_dir / f"neural__{path.stem}.png"
    _plot_rollout(panels, distance, task=task, stem=path.stem, output=figure_path)
    record["figure"] = str(figure_path)
    record["status"] = "ok"
    return record


def _analyze_agent(
    hidden: torch.Tensor,
    visible: torch.Tensor,
    distance: torch.Tensor,
    collision: torch.Tensor,
) -> tuple[dict[str, Any], dict[str, Any]]:
    steps, episode_count, units = hidden.shape
    pooled = hidden.reshape(-1, units)
    sample = _subsample_rows(pooled, POOL_SAMPLE_LIMIT)

    centered = sample - sample.mean(dim=0, keepdim=True)
    explained = torch.zeros(min(centered.shape))
    coords = torch.zeros(sample.shape[0], 2)
    if float(centered.abs().max()) > 0.0:
        _, singular, v_rows = torch.linalg.svd(centered, full_matrices=False)
        variance = singular.square()
        explained = variance / variance.sum().clamp_min(1e-12)
        coords = centered @ v_rows[:2].T
    participation = float(
        explained.sum().square() / explained.square().sum().clamp_min(1e-12)
    )
    cumulative = torch.cumsum(explained, dim=0)
    n90 = min(int((cumulative < 0.90).sum()) + 1, explained.numel())

    speed = (hidden[1:] - hidden[:-1]).norm(dim=2)  # (T-1, Ev)
    event_curve, event_count = _event_triggered(speed, collision)

    visible_flat = visible.reshape(-1)
    global_std = pooled.std(dim=0).clamp_min(1e-8)
    modulation = None
    modulated_fraction = None
    if bool(visible_flat.any()) and not bool(visible_flat.all()):
        mean_visible = pooled[visible_flat].mean(dim=0)
        mean_hidden = pooled[~visible_flat].mean(dim=0)
        modulation = (mean_visible - mean_hidden) / global_std
        modulated_fraction = float((modulation.abs() > MODULATION_THRESHOLD).float().mean())

    stats = {
        "units": units,
        "active_units": int((pooled.amax(dim=0) > 1e-8).sum()),
        "explained_top8": [round(float(x), 4) for x in explained[:8].tolist()],
        "participation_ratio": round(participation, 3),
        "components_for_90pct": n90,
        "mean_hidden_speed": round(float(speed.mean()), 4),
        "visibility_modulated_fraction": (
            round(modulated_fraction, 4) if modulated_fraction is not None else None
        ),
        "collision_events": event_count,
    }
    panel = {
        "raster": _example_raster(hidden),
        "coords": coords,
        "explained": explained[:20],
        "event_curve": event_curve,
        "event_count": event_count,
    }
    return stats, panel


def _example_raster(hidden: torch.Tensor) -> torch.Tensor:
    """Peak-sorted, per-unit-normalized (units, T) raster of the first episode."""
    episode = hidden[:, 0, :].T  # (units, T)
    unit_max = episode.amax(dim=1, keepdim=True).clamp_min(1e-8)
    normalized = episode / unit_max
    active = episode.amax(dim=1) > 1e-8
    peak = normalized.argmax(dim=1)
    order = sorted(
        range(episode.shape[0]),
        key=lambda unit: (not bool(active[unit]), int(peak[unit]), unit),
    )
    return normalized[order]


def _event_triggered(
    series: torch.Tensor,
    events: torch.Tensor,
) -> tuple[torch.Tensor | None, int]:
    """Average ``series`` (T-1, E) in a window around event onsets (T, E)."""
    steps = series.shape[0]
    onsets = events & ~torch.cat([torch.zeros_like(events[:1]), events[:-1]], dim=0)
    windows = []
    for t, episode in onsets.nonzero().tolist():
        # series index t-1 is the step arriving at the event state.
        if t - EVENT_WINDOW < 0 or t + EVENT_WINDOW >= steps:
            continue
        windows.append(series[t - EVENT_WINDOW : t + EVENT_WINDOW + 1, episode])
    if not windows:
        return None, 0
    return torch.stack(windows).mean(dim=0), len(windows)


def _state_series(tensor: torch.Tensor, steps: int) -> torch.Tensor:
    """Align a state series with hidden[t] (schema v3 stores max_steps + 1 states)."""
    return tensor[:steps]


def _subsample_rows(tensor: torch.Tensor, limit: int) -> torch.Tensor:
    if tensor.shape[0] <= limit:
        return tensor
    indices = torch.linspace(0, tensor.shape[0] - 1, limit).long()
    return tensor[indices]


def _rollout_task(metadata: dict[str, Any]) -> str:
    env_config = metadata.get("env_config") or {}
    task = env_config.get("task")
    if task is None:
        checkpoint_env = (metadata.get("checkpoint_config") or {}).get("env") or {}
        task = checkpoint_env.get("task")
    return str(task)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _plot_rollout(
    panels: dict[str, Any],
    distance: torch.Tensor,
    *,
    task: str,
    stem: str,
    output: Path,
) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 3, figsize=(16, 8.5), constrained_layout=True)

    for row, agent in enumerate(AGENTS):
        panel = panels[agent]
        raster_axis = axes[row][0]
        image = raster_axis.imshow(
            panel["raster"].numpy(),
            aspect="auto",
            cmap="viridis",
            interpolation="nearest",
        )
        raster_axis.set_title(f"{agent}: units × time (episode 0, peak-sorted)", fontsize=10)
        raster_axis.set_xlabel("timestep", fontsize=8)
        raster_axis.set_ylabel("unit", fontsize=8)
        raster_axis.tick_params(labelsize=8)
        figure.colorbar(image, ax=raster_axis, shrink=0.8, label="norm. activity")

        embed_axis = axes[row][1]
        coords = panel["coords"].numpy()
        color_values = _subsample_rows(distance.reshape(-1, 1), coords.shape[0]).squeeze(1).numpy()
        point_size = max(4.0, min(30.0, 4000.0 / max(1, coords.shape[0])))
        scatter = embed_axis.scatter(
            coords[:, 0],
            coords[:, 1],
            c=color_values,
            cmap="magma",
            s=point_size,
            alpha=0.6,
            linewidths=0,
        )
        explained = panel["explained"]
        embed_axis.set_title(
            f"{agent}: PC1–PC2 "
            f"({explained[0] * 100:.0f}% / {explained[1] * 100:.0f}% var)"
            if explained.numel() >= 2
            else f"{agent}: PC1–PC2",
            fontsize=10,
        )
        embed_axis.set_xlabel("PC1", fontsize=8)
        embed_axis.set_ylabel("PC2", fontsize=8)
        embed_axis.tick_params(labelsize=8)
        figure.colorbar(scatter, ax=embed_axis, shrink=0.8, label="distance")

    scree_axis = axes[0][2]
    for agent in AGENTS:
        explained = panels[agent]["explained"].numpy()
        scree_axis.plot(
            range(1, len(explained) + 1),
            explained,
            marker="o",
            markersize=3,
            label=agent,
            color=AGENT_COLORS[agent],
        )
    scree_axis.set_title("PCA variance spectrum", fontsize=10)
    scree_axis.set_xlabel("component", fontsize=8)
    scree_axis.set_ylabel("explained variance ratio", fontsize=8)
    scree_axis.tick_params(labelsize=8)
    scree_axis.legend(fontsize=8)
    scree_axis.grid(alpha=0.25, linewidth=0.5)

    event_axis = axes[1][2]
    plotted = False
    for agent in AGENTS:
        panel = panels[agent]
        if panel["event_curve"] is None:
            continue
        offsets = range(-EVENT_WINDOW, EVENT_WINDOW + 1)
        event_axis.plot(
            list(offsets),
            panel["event_curve"].numpy(),
            label=f"{agent} (n={panel['event_count']})",
            color=AGENT_COLORS[agent],
        )
        plotted = True
    if plotted:
        event_axis.axvline(0, color="#697177", linewidth=1, linestyle="--")
        event_axis.set_title("hidden-state speed around collisions", fontsize=10)
        event_axis.set_xlabel("steps from collision", fontsize=8)
        event_axis.set_ylabel("||h_t − h_{t−1}||", fontsize=8)
        event_axis.legend(fontsize=8)
        event_axis.grid(alpha=0.25, linewidth=0.5)
    else:
        event_axis.text(0.5, 0.5, "no collision events", ha="center", va="center")
        event_axis.set_axis_off()

    figure.suptitle(f"{task} — {stem}", fontsize=12)
    figure.savefig(output, dpi=FIGURE_DPI)
    plt.close(figure)


def _aggregate_by_task(records: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = (
        "participation_ratio",
        "components_for_90pct",
        "mean_hidden_speed",
        "visibility_modulated_fraction",
    )
    ok_records = [record for record in records if record.get("status") == "ok"]
    aggregate: dict[str, Any] = {}
    for task in sorted({str(record["task"]) for record in ok_records}):
        task_records = [record for record in ok_records if str(record["task"]) == task]
        per_agent: dict[str, Any] = {}
        for agent in AGENTS:
            per_agent[agent] = {
                metric: _mean_std(
                    [
                        float(record["agents"][agent][metric])
                        for record in task_records
                        if record["agents"][agent].get(metric) is not None
                    ]
                )
                for metric in metrics
            }
        aggregate[task] = {"n": len(task_records), "agents": per_agent}
    return aggregate


def _plot_aggregate(aggregate: dict[str, Any], figures_dir: Path) -> str | None:
    import matplotlib.pyplot as plt

    tasks = sorted(aggregate)
    if not tasks:
        return None
    metrics = (
        ("participation_ratio", "PCA participation ratio"),
        ("components_for_90pct", "components for 90% variance"),
        ("mean_hidden_speed", "mean hidden-state speed"),
        ("visibility_modulated_fraction", "visibility-modulated unit fraction"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    bar_width = 0.8 / len(AGENTS)
    for axis, (metric, title) in zip(axes.flat, metrics, strict=True):
        for agent_index, agent in enumerate(AGENTS):
            positions = []
            means = []
            stds = []
            for task_index, task in enumerate(tasks):
                stats = aggregate[task]["agents"][agent].get(metric) or {}
                if stats.get("mean") is None:
                    continue
                positions.append(task_index + agent_index * bar_width)
                means.append(stats["mean"])
                stds.append(stats.get("std") or 0.0)
            if not positions:
                continue
            axis.bar(
                positions,
                means,
                width=bar_width * 0.9,
                yerr=stds,
                capsize=3,
                label=agent,
                color=AGENT_COLORS[agent],
                error_kw={"linewidth": 0.8},
            )
        axis.set_title(title, fontsize=10)
        axis.set_xticks(
            [index + bar_width * (len(AGENTS) - 1) / 2 for index in range(len(tasks))],
            [f"{task}\n(n={aggregate[task]['n']})" for task in tasks],
            fontsize=9,
        )
        axis.tick_params(labelsize=8)
        axis.grid(axis="y", alpha=0.25, linewidth=0.5)
    axes.flat[0].legend(fontsize=8)
    figure.suptitle("Network activation statistics by task (mean ± std across pairs)", fontsize=12)

    path = figures_dir / "neural_aggregate.png"
    figure.savefig(path, dpi=FIGURE_DPI)
    plt.close(figure)
    return str(path)


def _mean_std(values: list[float]) -> dict[str, float | int | None]:
    values = [value for value in values if math.isfinite(value)]
    if not values:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


if __name__ == "__main__":
    main()
