"""Zhang et al. (2025) Figure 6 rendered from this repository's study artifacts.

Panels follow the published layout: an example raster (a), decoder accuracy
(b-e), example shared dimensions (f), shared-dimension counts and correlations
(g,h), loading structure (i,j), intra-brain disruption (k), partner variance
(l-n), partner representation over training and against collisions (p-r), and
the causal removals (s-u). Panel o is animal data and has no counterpart here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pyarrow.parquet as pq
import torch
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from mouse_run_run.analyses.paper_plsc import load_paper_plsc
from mouse_run_run.analyses.shared_structure import (
    DisruptionConfig,
    disrupted_cross_covariance,
    loading_comparison,
    unique_dimension_basis,
)
from mouse_run_run.analyses.statistics import permutation_comparison
from mouse_run_run.artifacts.rollout import load_rollout
from mouse_run_run.figures import sources
from mouse_run_run.figures.fig5 import (
    NON_SOCIAL_COLOR,
    SOCIAL_COLOR,
    _apply_style,
    _empty_panel,
    _panel_letter,
)


FIGURE_FORMAT = "mrr-fig6-v1"
RASTER_EVENTS = (
    ("chaser_new_field", "New fields", "#3fb8ae"),
    ("collision", "Collision", "#ef5350"),
    ("chaser_approach", "Approach", "#f5c243"),
    ("explorer_escape_close", "Escape (close)", "#f9d3e8"),
    ("explorer_escape_near", "Escape (near)", "#ef9ad0"),
    ("explorer_escape_far", "Escape (far)", "#d94fa8"),
)
DECODER_PANELS = (
    ("b", "chaser", "collision", "Collision", "Chasers:"),
    ("c", "chaser", "partner_escape", "Partner's escape", "Chasers:"),
    ("d", "explorer", "collision", "Collision", "Explorer:"),
    ("e", "explorer", "partner_approach", "Partner's approach", "Explorer:"),
)
CAUSAL_PANELS = (
    ("s", "event.collision", "Number of collisions", 1.0),
    ("t", "event_fraction.chaser_partner_visible", "Time partner in vision (%)", 100.0),
    ("u", "world.distance_mean", "Average distance to partner (units)", 1.0),
)
CAUSAL_ORDER = (
    ("shared_readout", "PLSCs", SOCIAL_COLOR),
    ("shifted_random_pc_readout", "Random PCs", "#9e9e9e"),
    ("baseline", "Original", NON_SOCIAL_COLOR),
)


@dataclass(frozen=True)
class Fig6Config:
    shared_rank: int = 10
    disruption_shuffles: int = 100
    permutations: int = 10_000
    seed: int = 0
    raster_units: int = 6
    raster_steps: int = 120
    formats: tuple[str, ...] = ("png", "svg")


@dataclass
class Fig6Result:
    figure: list[Path] = field(default_factory=list)
    source_data: list[Path] = field(default_factory=list)
    manifest: Path | None = None
    notes: list[str] = field(default_factory=list)


def render_fig6(
    study_output: Path,
    output: Path,
    *,
    config: Fig6Config | None = None,
) -> Fig6Result:
    config = config or Fig6Config()
    output.mkdir(parents=True, exist_ok=True)
    result = Fig6Result()
    _apply_style()
    figure = plt.figure(figsize=(24.0, 17.0))
    grid = GridSpec(
        4,
        12,
        figure=figure,
        hspace=0.62,
        wspace=1.15,
        left=0.045,
        right=0.99,
        top=0.93,
        bottom=0.05,
    )
    tables: dict[str, list[dict[str, Any]]] = {}
    example = _example_unit(study_output)
    tables.update(_raster_panels(figure, grid, example, config, result))
    tables.update(_decoder_panels(figure, grid, study_output, config, result))
    tables.update(_shared_panels(figure, grid, study_output, config, result))
    tables.update(_loading_panels(figure, grid, example, config, result))
    tables.update(_variance_panels(figure, grid, study_output, config, result))
    tables.update(_training_panels(figure, grid, study_output, config, result))
    tables.update(_causal_panels(figure, grid, study_output, config, result))
    _legend(figure)

    for suffix in config.formats:
        path = output / f"fig6.{suffix}"
        figure.savefig(path, dpi=170 if suffix == "png" else None)
        result.figure.append(path)
    plt.close(figure)
    for name, rows in tables.items():
        if rows:
            result.source_data.append(
                sources.write_source_table(
                    output / "source_data" / f"{name}.parquet", rows, f"{FIGURE_FORMAT}-{name}"
                )
            )
    result.manifest = output / "manifest.json"
    result.manifest.write_text(
        json.dumps(
            {
                "format": FIGURE_FORMAT,
                "study_output": str(study_output),
                "config": {
                    "shared_rank": config.shared_rank,
                    "disruption_shuffles": config.disruption_shuffles,
                    "permutations": config.permutations,
                    "seed": config.seed,
                },
                "figures": [str(path) for path in result.figure],
                "source_data": [str(path) for path in result.source_data],
                "notes": result.notes,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return result


def _example_unit(study_output: Path) -> Path | None:
    for unit in sources.discover_units(study_output):
        if unit.condition != "social":
            continue
        for checkpoint in sorted((unit.root / "neural").glob("update_*")):
            candidate = checkpoint / "partial"
            if (candidate / "plsc").is_dir() and (candidate / "rollout").is_dir():
                return candidate
    return None


def _raster_panels(
    figure: Figure,
    grid: GridSpec,
    example: Path | None,
    config: Fig6Config,
    result: Fig6Result,
) -> dict[str, list[dict[str, Any]]]:
    axes = figure.add_subplot(grid[0, 0:5])
    shared_axis = figure.add_subplot(grid[0, 9:12])
    if example is None:
        _empty_panel(axes, "a", "", "no neural rollout")
        _empty_panel(shared_axis, "f", "", "no neural rollout")
        result.notes.append("panels a and f need a social neural rollout")
        return {}
    rollout = load_rollout(example / "rollout")
    fitted = load_paper_plsc(example / "plsc")
    trajectory = rollout.trajectory
    steps = min(config.raster_steps, trajectory.horizon)
    _panel_letter(axes, "a")
    axes.set_title("Chaser (top) and explorer (bottom) units", fontsize=10, pad=26)
    rows = []
    for offset, agent_id in enumerate(("chaser", "explorer")):
        activity = trajectory.agents[agent_id].activations["hidden"][:steps, 0]
        strongest = activity.std(dim=0).argsort(descending=True)[: config.raster_units]
        for index, unit in enumerate(strongest.tolist()):
            trace = activity[:, unit]
            scale = float(trace.abs().max()) or 1.0
            base = -(offset * (config.raster_units + 1) + index)
            axes.plot(range(steps), base + 0.8 * trace / scale, linewidth=0.8, color="#4a4a4a")
    for name, label, color in RASTER_EVENTS:
        if name not in trajectory.events:
            continue
        active = trajectory.events[name][:steps, 0].bool().nonzero().flatten().tolist()
        for step in active:
            axes.axvspan(step - 0.5, step + 0.5, color=color, alpha=0.35, linewidth=0)
        rows.extend({"step": step, "event": name, "episode": 0} for step in active)
    axes.set_xlim(0, steps)
    axes.set_yticks([])
    axes.set_xlabel("Timestep")
    axes.legend(
        handles=[plt.Rectangle((0, 0), 1, 1, color=color, alpha=0.5) for _, _, color in RASTER_EVENTS],
        labels=[label for _, label, _ in RASTER_EVENTS],
        fontsize=7,
        ncol=6,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.005),
        columnspacing=1.0,
        handlelength=1.2,
    )

    _panel_letter(shared_axis, "f")
    shared_axis.set_title("Shared dimensions (PLSC 1-4)", fontsize=10)
    projections = {}
    for agent_id, color in (("chaser", SOCIAL_COLOR), ("explorer", NON_SOCIAL_COLOR)):
        activity = trajectory.agents[agent_id].activations["hidden"][:steps, 0].double()
        standardized = (activity - fitted.means[agent_id]) / fitted.scales[agent_id]
        scores = standardized @ fitted.bases[agent_id][:, :4]
        projections[agent_id] = scores
        for dimension in range(scores.shape[1]):
            trace = scores[:, dimension]
            scale = float(trace.abs().max()) or 1.0
            shared_axis.plot(
                range(steps),
                -dimension + 0.4 * trace / scale,
                color=color,
                linewidth=0.9,
            )
    shared_axis.set_yticks([-index for index in range(4)])
    shared_axis.set_yticklabels([f"{index + 1}" for index in range(4)])
    shared_axis.set_ylabel("Shared dimension")
    shared_axis.set_xlabel("Timestep")
    shared_axis.set_xlim(0, steps)
    shared_rows = [
        {
            "step": step,
            "dimension": dimension + 1,
            "chaser": float(projections["chaser"][step, dimension]),
            "explorer": float(projections["explorer"][step, dimension]),
        }
        for dimension in range(4)
        for step in range(steps)
    ]
    return {"fig6_raster_events": rows, "fig6_shared_traces": shared_rows}


def _decoder_panels(
    figure: Figure,
    grid: GridSpec,
    study_output: Path,
    config: Fig6Config,
    result: Fig6Result,
) -> dict[str, list[dict[str, Any]]]:
    table = study_output / "tables" / "decoder.parquet"
    positions = {"b": (0, 5, 6), "c": (0, 6, 7), "d": (0, 7, 8), "e": (0, 8, 9)}
    if not table.is_file():
        for letter, (row, start, end) in positions.items():
            _empty_panel(figure.add_subplot(grid[row, start:end]), letter, "", "no decoder table")
        result.notes.append("panels b-e need tables/decoder.parquet")
        return {}
    rows = pq.read_table(table).to_pylist()
    collected: list[dict[str, Any]] = []
    for letter, agent, target, title, prefix in DECODER_PANELS:
        row, start, end = positions[letter]
        axis = figure.add_subplot(grid[row, start:end])
        groups = {
            condition: [
                float(item["balanced_accuracy"])
                for item in rows
                if item["agent"] == agent
                and item["target"] == target
                and item["condition"] == condition
                and item["balanced_accuracy"] is not None
            ]
            for condition in ("social", "non_social")
        }
        _panel_letter(axis, letter)
        axis.set_title(title, fontsize=10)
        axis.set_ylabel("Balanced accuracy of decoder", fontsize=8)
        _paired_box(axis, groups, config, labels=("Social", "Non-social"), xlabel=prefix)
        axis.set_ylim(0.4, 1.05)
        collected.extend(
            {"panel": letter, "agent": agent, "target": target, "condition": condition, "value": value}
            for condition, values in groups.items()
            for value in values
        )
    return {"fig6_decoder": collected}


def _shared_panels(
    figure: Figure,
    grid: GridSpec,
    study_output: Path,
    config: Fig6Config,
    result: Fig6Result,
) -> dict[str, list[dict[str, Any]]]:
    table = study_output / "tables" / "plsc_seed_summary.parquet"
    axes = {
        "g": figure.add_subplot(grid[1, 0:2]),
        "h": figure.add_subplot(grid[1, 2:4]),
    }
    if not table.is_file():
        for letter, axis in axes.items():
            _empty_panel(axis, letter, "", "no PLSC summary")
        result.notes.append("panels g and h need tables/plsc_seed_summary.parquet")
        return {}
    rows = pq.read_table(table).to_pylist()
    groups_for = lambda metric: {  # noqa: E731 - a local table lookup
        label: [
            float(item[metric])
            for item in rows
            if item["condition"] == condition
            and item["visibility"] == visibility
            and item["control"] == "primary"
        ]
        for label, condition, visibility in (
            ("Social", "social", "partial"),
            ("Non-social", "non_social", "none"),
            ("Non-social\n(vision)", "non_social", "full"),
        )
    }
    collected: list[dict[str, Any]] = []
    for letter, metric, ylabel in (
        ("g", "significant_dimensions_mean", "Number of shared dimensions"),
        ("h", "delta_pcc_mean", "ΔPCC of PLSC1"),
    ):
        axis = axes[letter]
        groups = groups_for(metric)
        _panel_letter(axis, letter)
        axis.set_ylabel(ylabel, fontsize=8)
        _multi_box(axis, groups, config, colors=(SOCIAL_COLOR, NON_SOCIAL_COLOR, "#9e9e9e"))
        collected.extend(
            {"panel": letter, "group": label, "metric": metric, "value": value}
            for label, values in groups.items()
            for value in values
        )
    return {"fig6_shared_dimensions": collected}


def _loading_panels(
    figure: Figure,
    grid: GridSpec,
    example: Path | None,
    config: Fig6Config,
    result: Fig6Result,
) -> dict[str, list[dict[str, Any]]]:
    axes = {
        "i": figure.add_subplot(grid[1, 4:6]),
        "j": figure.add_subplot(grid[1, 6:8]),
        "k": figure.add_subplot(grid[1, 8:12]),
    }
    if example is None:
        for letter, axis in axes.items():
            _empty_panel(axis, letter, "", "no neural rollout")
        return {}
    fitted = load_paper_plsc(example / "plsc")
    rollout = load_rollout(example / "rollout")
    unique = unique_dimension_basis(
        fitted, rollout, agent_id="chaser", shared_rank=config.shared_rank
    )
    basis = fitted.bases["chaser"]
    collected: list[dict[str, Any]] = []
    for letter, second, second_label in (
        ("i", unique[:, 0], "U1"),
        ("j", basis[:, 1], "PLSC2"),
    ):
        axis = axes[letter]
        comparison = loading_comparison(basis[:, 0], second)
        _panel_letter(axis, letter)
        both = comparison["first_significant"] & comparison["second_significant"]
        neither = ~comparison["first_significant"] & ~comparison["second_significant"]
        edges = torch.linspace(
            float(comparison["difference"].min()),
            float(comparison["difference"].max()),
            33,
        ).tolist()
        for mask, color, label in (
            (neither, "#d8d8d8", "Neither"),
            (comparison["second_significant"] & ~both, "#9aa7e8", second_label),
            (comparison["first_significant"] & ~both, "#f0a6a0", "PLSC1"),
            (both, "#5b2b8a", "Both"),
        ):
            values = comparison["difference"][mask]
            if values.numel():
                axis.hist(values.tolist(), bins=edges, color=color, alpha=0.9, label=label)
        axis.set_xlabel(f"|W$_{{PLSC1}}$| - |W$_{{{second_label}}}$|", fontsize=8)
        axis.set_ylabel("Unit count", fontsize=8)
        axis.legend(fontsize=7, frameon=False)
        collected.extend(
            {
                "panel": letter,
                "unit": index,
                "difference": float(comparison["difference"][index]),
                "first_significant": bool(comparison["first_significant"][index]),
                "second_significant": bool(comparison["second_significant"][index]),
                "comparison": second_label,
            }
            for index in range(basis.shape[0])
        )
    disruption = disrupted_cross_covariance(
        fitted, rollout, config=DisruptionConfig(shuffles=config.disruption_shuffles)
    )
    axis = axes["k"]
    _panel_letter(axis, "k")
    labels = ["Original", "Remove intraset\ncorrelation", "Remove intraset\ncoupling"]
    values = [
        disruption["original"],
        disruption["intraset_correlation"],
        disruption["intraset_coupling"],
    ]
    axis.bar(range(3), values, color=[SOCIAL_COLOR, "#9e9e9e", "#9e9e9e"], edgecolor="black", linewidth=0.6)
    axis.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    axis.set_xticks(range(3))
    axis.set_xticklabels(labels, fontsize=7)
    axis.set_ylabel("Cov of PLSC1 (observed - shuffle)", fontsize=8)
    collected_k = [
        {"panel": "k", "condition": label.replace("\n", " "), "value": value}
        for label, value in zip(labels, values)
    ]
    return {"fig6_loadings": collected, "fig6_disruption": collected_k}


def _variance_panels(
    figure: Figure,
    grid: GridSpec,
    study_output: Path,
    config: Fig6Config,
    result: Fig6Result,
) -> dict[str, list[dict[str, Any]]]:
    axes = {
        "l": figure.add_subplot(grid[2, 0:2]),
        "m": figure.add_subplot(grid[2, 2:4]),
        "n": figure.add_subplot(grid[2, 4:6]),
        "o": figure.add_subplot(grid[2, 6:8]),
    }
    _panel_letter(axes["o"], "o")
    axes["o"].set_title("Animals", fontsize=10)
    axes["o"].text(
        0.5,
        0.5,
        "animal recordings\nnot part of this replication",
        ha="center",
        va="center",
        fontsize=9,
        color="#8a8a8a",
    )
    axes["o"].set_xticks([])
    axes["o"].set_yticks([])
    for spine in axes["o"].spines.values():
        spine.set_visible(False)
    result.notes.append("panel o is animal data and has no counterpart in this repository")

    table = study_output / "tables" / "partner_variance.parquet"
    if not table.is_file():
        for letter in ("l", "m", "n"):
            _empty_panel(axes[letter], letter, "", "run `mrr figures variance`")
        result.notes.append("panels l-n need tables/partner_variance.parquet")
        return {}
    rows = pq.read_table(table).to_pylist()
    collected: list[dict[str, Any]] = []
    for letter, space, title in (("l", "full", "Full space"), ("m", "plsc1", "PLSC1")):
        axis = axes[letter]
        groups = {
            condition: [
                float(item["partner_variance"])
                for item in rows
                if item["space"] == space
                and item["condition"] == condition
                and item["agent"] == "chaser"
            ]
            for condition in ("social", "non_social")
        }
        _panel_letter(axis, letter)
        axis.set_title(title, fontsize=10)
        axis.set_ylabel("Neural variance explained by partner (%)", fontsize=8)
        _paired_box(axis, groups, config, labels=("Social", "Non-social"))
        collected.extend(
            {"panel": letter, "space": space, "condition": condition, "value": value}
            for condition, values in groups.items()
            for value in values
        )
    axis = axes["n"]
    _panel_letter(axis, "n")
    axis.set_title("Artificial agents", fontsize=10)
    agents = {
        agent: [
            float(item["partner_variance"])
            for item in rows
            if item["space"] == "action" and item["agent"] == agent and item["condition"] == "social"
        ]
        for agent in ("explorer", "chaser")
    }
    means = [sum(values) / len(values) if values else 0.0 for values in agents.values()]
    errors = [
        (torch.tensor(values).std(unbiased=True) / max(len(values), 1) ** 0.5).item()
        if len(values) > 1
        else 0.0
        for values in agents.values()
    ]
    axis.bar(
        range(2),
        means,
        yerr=errors,
        color=["#f3a9a2", SOCIAL_COLOR],
        edgecolor="black",
        linewidth=0.6,
        capsize=4,
    )
    axis.set_xticks(range(2))
    axis.set_xticklabels(["Explorer", "Chaser"], fontsize=8)
    axis.set_ylabel("Neural variance explained by partner (%)", fontsize=8)
    collected.extend(
        {"panel": "n", "space": "action", "condition": "social", "agent": agent, "value": value}
        for agent, values in agents.items()
        for value in values
    )
    return {"fig6_partner_variance": collected}


def _training_panels(
    figure: Figure,
    grid: GridSpec,
    study_output: Path,
    config: Fig6Config,
    result: Fig6Result,
) -> dict[str, list[dict[str, Any]]]:
    axes = {
        "p": figure.add_subplot(grid[2, 8:10]),
        "q": figure.add_subplot(grid[2, 10:12]),
        "r": figure.add_subplot(grid[3, 0:3]),
    }
    table = study_output / "tables" / "partner_representation.parquet"
    if not table.is_file():
        for letter, axis in axes.items():
            _empty_panel(axis, letter, "", "run `mrr figures sweep`")
        result.notes.append("panels p-r need tables/partner_representation.parquet")
        return {}
    rows = [item for item in pq.read_table(table).to_pylist() if item["agent"] == "chaser"]
    axis = axes["p"]
    _panel_letter(axis, "p")
    axis.set_xlabel("Training iteration")
    axis.set_ylabel("Neural variance explained by partner (%)", fontsize=8)
    summary: list[dict[str, Any]] = []
    for condition, color in (("social", SOCIAL_COLOR), ("non_social", NON_SOCIAL_COLOR)):
        points: dict[int, list[float]] = {}
        for item in rows:
            if item["condition"] == condition:
                points.setdefault(int(item["checkpoint_update"]), []).append(
                    float(item["partner_representation"])
                )
        updates = sorted(points)
        means = [sum(points[update]) / len(points[update]) for update in updates]
        errors = [
            float(torch.tensor(points[update]).std(unbiased=True) / len(points[update]) ** 0.5)
            if len(points[update]) > 1
            else 0.0
            for update in updates
        ]
        axis.plot(updates, means, color=color, linewidth=1.6)
        axis.fill_between(
            updates,
            [mean - error for mean, error in zip(means, errors)],
            [mean + error for mean, error in zip(means, errors)],
            color=color,
            alpha=0.25,
            linewidth=0,
        )
        summary.extend(
            {"panel": "p", "condition": condition, "update": update, "mean": mean, "sem": error}
            for update, mean, error in zip(updates, means, errors)
        )
    for letter, condition, color, title in (
        ("q", "social", SOCIAL_COLOR, "Social agents"),
        ("r", "non_social", NON_SOCIAL_COLOR, "Non-social agents"),
    ):
        axis = axes[letter]
        selected = [item for item in rows if item["condition"] == condition]
        x = torch.tensor([float(item["collision_time_percent"]) for item in selected])
        y = torch.tensor([float(item["partner_representation"]) for item in selected])
        _panel_letter(axis, letter)
        axis.set_title(title, fontsize=10)
        axis.scatter(x.tolist(), y.tolist(), s=8, color=color, alpha=0.7, linewidths=0)
        axis.set_xlabel("Collision time (%)")
        axis.set_ylabel("Neural variance explained by partner (%)", fontsize=8)
        r_squared = _fit_line(axis, x, y)
        axis.text(
            0.05,
            0.95,
            f"$R^2$ = {r_squared:.3f}",
            transform=axis.transAxes,
            fontsize=8,
            va="top",
        )
        summary.extend(
            {
                "panel": letter,
                "condition": condition,
                "collision_time_percent": float(item["collision_time_percent"]),
                "partner_representation": float(item["partner_representation"]),
                "r_squared": r_squared,
            }
            for item in selected
        )
    return {"fig6_partner_representation": summary}


def _causal_panels(
    figure: Figure,
    grid: GridSpec,
    study_output: Path,
    config: Fig6Config,
    result: Fig6Result,
) -> dict[str, list[dict[str, Any]]]:
    table = study_output / "tables" / "causal.parquet"
    axes = {
        "s": figure.add_subplot(grid[3, 3:6]),
        "t": figure.add_subplot(grid[3, 6:9]),
        "u": figure.add_subplot(grid[3, 9:12]),
    }
    if not table.is_file():
        for letter, axis in axes.items():
            _empty_panel(axis, letter, "", "no causal table")
        result.notes.append("panels s-u need tables/causal.parquet")
        return {}
    rows = pq.read_table(table).to_pylist()
    removed = {
        name: [float(item["removed_variance"]) for item in rows if item["intervention"] == name]
        for name, _, _ in CAUSAL_ORDER
    }
    shared = removed.get("shared_readout") or [0.0]
    control = removed.get("shifted_random_pc_readout") or [0.0]
    result.notes.append(
        "panels s-u: the removal takes "
        f"{100 * sum(shared) / len(shared):.1f}% of the variance and its control takes "
        f"{100 * sum(control) / len(control):.1f}%"
    )
    collected: list[dict[str, Any]] = []
    for letter, metric, ylabel, scale in CAUSAL_PANELS:
        axis = axes[letter]
        groups: dict[str, list[float]] = {}
        for name, label, _ in CAUSAL_ORDER:
            per_seed: dict[int, list[float]] = {}
            for item in rows:
                if item["metric"] == metric and item["intervention"] == name:
                    per_seed.setdefault(int(item["seed"]), []).append(float(item["value"]))
            groups[label] = [
                scale * sum(values) / len(values) for values in per_seed.values()
            ]
        _panel_letter(axis, letter)
        axis.set_ylabel(ylabel, fontsize=8)
        _multi_box(axis, groups, config, colors=[color for _, _, color in CAUSAL_ORDER])
        collected.extend(
            {"panel": letter, "metric": metric, "intervention": label, "value": value}
            for label, values in groups.items()
            for value in values
        )
    return {"fig6_causal": collected}


def _paired_box(
    axis: Any,
    groups: dict[str, list[float]],
    config: Fig6Config,
    *,
    labels: tuple[str, str],
    xlabel: str = "",
) -> None:
    left, right = list(groups.values())
    if not left or not right:
        axis.axis("off")
        return
    plot = axis.boxplot([left, right], widths=0.55, patch_artist=True,
                        medianprops={"color": "black"}, flierprops={"marker": "o", "markersize": 3})
    for patch, color in zip(plot["boxes"], (SOCIAL_COLOR, NON_SOCIAL_COLOR)):
        patch.set_facecolor(color)
        patch.set_edgecolor("black")
    axis.set_xticks([1, 2])
    axis.set_xticklabels(labels, fontsize=7, rotation=25, ha="right")
    if xlabel:
        axis.set_xlabel(xlabel, fontsize=8)
    comparison = permutation_comparison(
        left, right, permutations=config.permutations, seed=config.seed
    )
    _significance_bar(axis, 1, 2, left + right, comparison.stars)


def _multi_box(
    axis: Any,
    groups: dict[str, list[float]],
    config: Fig6Config,
    *,
    colors: Sequence[str],
) -> None:
    labels = [label for label, values in groups.items() if values]
    values = [groups[label] for label in labels]
    if len(values) < 2:
        axis.axis("off")
        return
    plot = axis.boxplot(values, widths=0.55, patch_artist=True,
                        medianprops={"color": "black"}, flierprops={"marker": "o", "markersize": 3})
    for patch, color in zip(plot["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor("black")
    axis.set_xticks(range(1, len(labels) + 1))
    axis.set_xticklabels(labels, fontsize=7, rotation=20, ha="right")
    flattened = [value for group in values for value in group]
    for index in range(1, len(values)):
        comparison = permutation_comparison(
            values[0], values[index], permutations=config.permutations, seed=config.seed
        )
        _significance_bar(axis, 1, index + 1, flattened, comparison.stars, level=index)


def _significance_bar(
    axis: Any,
    left: float,
    right: float,
    values: Sequence[float],
    stars: str,
    *,
    level: int = 1,
) -> None:
    if not stars or not values:
        return
    top, bottom = max(values), min(values)
    span = max(top - bottom, 1e-9)
    height = top + span * (0.08 + 0.12 * level)
    axis.plot([left, left, right, right], [height, height + span * 0.03, height + span * 0.03, height],
              color="black", linewidth=0.9)
    axis.text((left + right) / 2, height + span * 0.04, stars, ha="center", va="bottom", fontsize=9)
    axis.set_ylim(bottom - span * 0.1, height + span * 0.18)


def _fit_line(axis: Any, x: torch.Tensor, y: torch.Tensor) -> float:
    if x.numel() < 3:
        return float("nan")
    design = torch.stack([x, torch.ones_like(x)], dim=1).double()
    solution = torch.linalg.lstsq(design, y.double().unsqueeze(1)).solution
    fitted = (design @ solution).squeeze(1)
    residual = (y.double() - fitted).square().sum()
    total = (y.double() - y.double().mean()).square().sum()
    order = x.argsort()
    axis.plot(x[order].tolist(), fitted[order].tolist(), color="black", linewidth=1.0)
    return float(1.0 - residual / total.clamp_min(1e-12))


def _legend(figure: Figure) -> None:
    figure.suptitle(
        "Figure 6 replication — shared neural dynamics across artificial agents",
        fontsize=14,
        y=0.985,
    )
