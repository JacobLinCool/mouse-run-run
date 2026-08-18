"""Zhang et al. (2025) Figure 5 rendered from this repository's study artifacts.

Panels follow the published layout: training curves for agents in their own
environment (c-h), evaluation against a uniform-random opponent (i-n), and the
chaser's movement geometry relative to a visible partner (o-r). Every panel
writes the numbers it draws to a source-data table next to the figure.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from math import radians
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from mouse_run_run.analyses.statistics import permutation_comparison, significance_stars
from mouse_run_run.figures import sources
from mouse_run_run.figures.sources import CHASER_MATCHUP, EXPLORER_MATCHUP


SOCIAL_COLOR = "#d33d2f"
NON_SOCIAL_COLOR = "#4b91d6"
EARLY_COLOR = "#b5b5b5"
LATE_COLOR = "#e79a37"
BAND_COLOR = "#e9e9e9"
FIGURE_FORMAT = "mrr-fig5-v1"


@dataclass(frozen=True)
class TrainingPanel:
    letter: str
    metric: str
    ylabel: str
    title: str
    scale: float = 1.0


@dataclass(frozen=True)
class BehaviorPanel:
    letter: str
    matchup: str
    metric: str
    ylabel: str
    title: str
    scale: float = 1.0


TRAINING_PANELS = (
    TrainingPanel("c", "chaser.return_mean", "Rewards", "Chaser"),
    TrainingPanel("d", "explorer.return_mean", "Reward", "Explorer"),
    TrainingPanel("e", "event.collision", "Number of collisions", "Chaser"),
    TrainingPanel(
        "f",
        "event_fraction.chaser_partner_visible",
        "Time partner in vision (%)",
        "Chaser",
        scale=100.0,
    ),
    TrainingPanel("g", "event.chaser_new_field", "Number of new fields", "Chaser"),
    TrainingPanel("h", "event.explorer_new_field", "Number of new fields", "Explorer"),
)

BEHAVIOR_PANELS = (
    BehaviorPanel("i", CHASER_MATCHUP, "event.collision", "Number of collisions", "Chaser"),
    BehaviorPanel(
        "j",
        CHASER_MATCHUP,
        "event_fraction.chaser_partner_visible",
        "Time partner in vision (%)",
        "Chaser",
        scale=100.0,
    ),
    BehaviorPanel(
        "k",
        CHASER_MATCHUP,
        "world.distance_mean",
        "Average distance to opponent (units)",
        "Chaser",
    ),
    BehaviorPanel("l", CHASER_MATCHUP, "event.chaser_new_field", "Number of new fields", "Chaser"),
    BehaviorPanel(
        "m",
        EXPLORER_MATCHUP,
        "event.explorer_new_field",
        "Number of new fields",
        "Explorer",
    ),
    BehaviorPanel(
        "n",
        EXPLORER_MATCHUP,
        "world.distance_mean",
        "Average distance to opponent (units)",
        "Explorer",
    ),
)


@dataclass(frozen=True)
class Fig5Config:
    box_checkpoint: int | None = None
    early_checkpoint: int | None = None
    late_checkpoint: int | None = None
    x_style: str = "log"
    vision_radius: int = 3
    polar_bins: int = 12
    angle_bin_width: float = 30.0
    permutations: int = 10_000
    seed: int = 0
    formats: tuple[str, ...] = ("png", "svg")


@dataclass
class Fig5Result:
    figure: list[Path] = field(default_factory=list)
    source_data: list[Path] = field(default_factory=list)
    manifest: Path | None = None
    notes: list[str] = field(default_factory=list)
    """Panels that were skipped or degraded, reported to the caller."""

    rendering: dict[str, Any] = field(default_factory=dict)
    """Drawing choices worth recording, such as the shared flow arrow scale."""


def render_fig5(
    study_output: Path,
    output: Path,
    *,
    config: Fig5Config | None = None,
) -> Fig5Result:
    """Render Figure 5 and its source data from one study output directory."""

    config = config or Fig5Config()
    if config.x_style not in ("log", "linear"):
        raise ValueError(f"unsupported x_style {config.x_style!r}")
    output.mkdir(parents=True, exist_ok=True)
    result = Fig5Result()
    _apply_style()
    figure = plt.figure(figsize=(24.0, 12.5))
    grid = GridSpec(
        3,
        12,
        figure=figure,
        hspace=0.55,
        wspace=1.05,
        left=0.04,
        right=0.99,
        top=0.90,
        bottom=0.07,
    )

    training = _training_panels(figure, grid, study_output, config, result)
    behavior = _behavior_panels(figure, grid, study_output, config, result)
    movement = _movement_panels(figure, grid, study_output, config, result)

    _condition_legend(figure)
    for suffix in config.formats:
        path = output / f"fig5.{suffix}"
        figure.savefig(path, dpi=200 if suffix == "png" else None)
        result.figure.append(path)
    plt.close(figure)

    tables = {**training, **behavior, **movement}
    for name, rows in tables.items():
        if not rows:
            continue
        result.source_data.append(
            sources.write_source_table(
                output / "source_data" / f"{name}.parquet", rows, f"{FIGURE_FORMAT}-{name}"
            )
        )
    manifest = {
        "format": FIGURE_FORMAT,
        "study_output": str(study_output),
        "config": {
            "box_checkpoint": config.box_checkpoint,
            "early_checkpoint": config.early_checkpoint,
            "late_checkpoint": config.late_checkpoint,
            "x_style": config.x_style,
            "vision_radius": config.vision_radius,
            "polar_bins": config.polar_bins,
            "angle_bin_width": config.angle_bin_width,
            "permutations": config.permutations,
            "seed": config.seed,
        },
        "figures": [str(path) for path in result.figure],
        "source_data": [str(path) for path in result.source_data],
        "notes": result.notes,
        "rendering": result.rendering,
        "matplotlib": matplotlib.__version__,
    }
    result.manifest = output / "manifest.json"
    result.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def _training_panels(
    figure: Figure,
    grid: GridSpec,
    study_output: Path,
    config: Fig5Config,
    result: Fig5Result,
) -> dict[str, list[dict[str, Any]]]:
    axes = [figure.add_subplot(grid[0, index * 2 : index * 2 + 2]) for index in range(6)]
    try:
        rows = sources.training_rows(study_output, [panel.metric for panel in TRAINING_PANELS])
    except FileNotFoundError as error:
        result.notes.append(f"training panels skipped: {error}")
        for panel, axis in zip(TRAINING_PANELS, axes):
            _empty_panel(axis, panel.letter, panel.title, "no training metrics")
        return {}
    summary = sources.summarize_curves(rows)
    for panel, axis in zip(TRAINING_PANELS, axes):
        panel_rows = [row for row in summary if row["metric"] == panel.metric]
        _panel_letter(axis, panel.letter)
        axis.set_title(panel.title, fontsize=11)
        axis.set_xlabel("Training iteration")
        axis.set_ylabel(panel.ylabel)
        if not panel_rows:
            _empty_panel(axis, panel.letter, panel.title, f"missing {panel.metric}")
            result.notes.append(f"panel {panel.letter}: metric {panel.metric} not recorded")
            continue
        updates = sorted({int(row["update"]) for row in panel_rows})
        _training_axis(axis, updates, config.x_style)
        for condition, color in (("social", SOCIAL_COLOR), ("non_social", NON_SOCIAL_COLOR)):
            selected = sorted(
                (row for row in panel_rows if row["condition"] == condition),
                key=lambda row: row["update"],
            )
            if not selected:
                continue
            _curve(
                axis,
                [row["update"] for row in selected],
                [row["mean"] * panel.scale for row in selected],
                [row["sem"] * panel.scale for row in selected],
                color,
            )
    return {
        "fig5_training": [
            {
                **row,
                "scale": _scale_for(row["metric"]),
                "plotted_mean": row["mean"] * _scale_for(row["metric"]),
                "plotted_sem": row["sem"] * _scale_for(row["metric"]),
            }
            for row in summary
        ]
    }


def _behavior_panels(
    figure: Figure,
    grid: GridSpec,
    study_output: Path,
    config: Fig5Config,
    result: Fig5Result,
) -> dict[str, list[dict[str, Any]]]:
    layout = {
        "i": (1, 0, 2, 3),
        "j": (1, 3, 5, 6),
        "k": (1, 6, 8, 9),
        "l": (2, 0, 2, 3),
        "m": (2, 3, 5, 6),
        "n": (2, 6, 8, 9),
    }
    try:
        table = sources.behavior_table(study_output)
    except FileNotFoundError as error:
        result.notes.append(f"random-opponent panels skipped: {error}")
        for panel in BEHAVIOR_PANELS:
            row, start, split, end = layout[panel.letter]
            axis = figure.add_subplot(grid[row, start:split])
            _empty_panel(axis, panel.letter, panel.title, "no behavior table")
            figure.add_subplot(grid[row, split:end]).axis("off")
        return {}
    summary_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    for panel in BEHAVIOR_PANELS:
        row, start, split, end = layout[panel.letter]
        curve_axis = figure.add_subplot(grid[row, start:split])
        box_axis = figure.add_subplot(grid[row, split:end])
        panel_seed_rows = sources.behavior_seed_rows(
            table, matchup=panel.matchup, metric=panel.metric, scale=panel.scale
        )
        _panel_letter(curve_axis, panel.letter)
        curve_axis.set_title(panel.title, fontsize=11)
        curve_axis.set_xlabel("Training iteration")
        curve_axis.set_ylabel(panel.ylabel)
        if not panel_seed_rows:
            _empty_panel(curve_axis, panel.letter, panel.title, f"missing {panel.metric}")
            box_axis.axis("off")
            result.notes.append(f"panel {panel.letter}: metric {panel.metric} absent from behavior table")
            continue
        checkpoints = sorted({int(row["checkpoint_update"]) for row in panel_seed_rows})
        summary = sources.summarize_behavior(panel_seed_rows)
        _training_axis(curve_axis, checkpoints, config.x_style)
        for condition, color in (("social", SOCIAL_COLOR), ("non_social", NON_SOCIAL_COLOR)):
            selected = sorted(
                (item for item in summary if item["condition"] == condition),
                key=lambda item: item["checkpoint_update"],
            )
            if not selected:
                continue
            _curve(
                curve_axis,
                [item["checkpoint_update"] for item in selected],
                [item["mean"] for item in selected],
                [item["sem"] for item in selected],
                color,
            )
        checkpoint = config.box_checkpoint if config.box_checkpoint is not None else checkpoints[-1]
        comparison = sources.compare_conditions(
            panel_seed_rows,
            checkpoint=checkpoint,
            permutations=config.permutations,
            seed=config.seed,
        )
        groups = {
            condition: [
                row["value"]
                for row in panel_seed_rows
                if row["condition"] == condition
                and int(row["checkpoint_update"]) == checkpoint
            ]
            for condition in ("social", "non_social")
        }
        _box_panel(box_axis, groups, comparison, panel.ylabel)
        summary_rows.extend({**item, "panel": panel.letter, "metric": panel.metric} for item in summary)
        seed_rows.extend({**item, "panel": panel.letter} for item in panel_seed_rows)
        comparison_rows.append({**comparison, "panel": panel.letter, "metric": panel.metric})
    return {
        "fig5_random_opponent": summary_rows,
        "fig5_random_opponent_seeds": seed_rows,
        "fig5_random_opponent_comparisons": comparison_rows,
    }


def _movement_panels(
    figure: Figure,
    grid: GridSpec,
    study_output: Path,
    config: Fig5Config,
    result: Fig5Result,
) -> dict[str, list[dict[str, Any]]]:
    flow_cells = grid[1, 9:12].subgridspec(1, 2, wspace=0.35)
    tail_cells = grid[2, 9:12].subgridspec(1, 2, wspace=0.45)
    early_axis = figure.add_subplot(flow_cells[0, 0])
    late_axis = figure.add_subplot(flow_cells[0, 1])
    polar_axis = figure.add_subplot(tail_cells[0, 0], projection="polar")
    angle_axis = figure.add_subplot(tail_cells[0, 1])
    try:
        table = sources.behavior_table(study_output)
    except FileNotFoundError as error:
        for axis, letter, title in (
            (early_axis, "o", "Early stage of training"),
            (late_axis, "p", "Late stage of training"),
            (angle_axis, "r", ""),
        ):
            _empty_panel(axis, letter, title, "no behavior rollouts")
        _panel_letter(polar_axis, "q")
        polar_axis.set_axis_off()
        result.notes.append(f"movement panels skipped: {error}")
        return {}
    checkpoints = sorted({int(value) for value in table.column("checkpoint_update").to_pylist()})
    early = config.early_checkpoint if config.early_checkpoint is not None else checkpoints[0]
    late = config.late_checkpoint if config.late_checkpoint is not None else checkpoints[-1]
    stages: dict[str, tuple[int, dict[int, Any]]] = {}
    for stage, checkpoint in (("early", early), ("late", late)):
        try:
            stages[stage] = (
                checkpoint,
                sources.movement_samples_by_seed(study_output, checkpoint=checkpoint),
            )
        except (FileNotFoundError, ValueError) as error:
            result.notes.append(f"movement stage {stage} skipped: {error}")
    if "early" not in stages or "late" not in stages:
        for axis, letter, title in (
            (early_axis, "o", "Early stage of training"),
            (late_axis, "p", "Late stage of training"),
            (angle_axis, "r", ""),
        ):
            _empty_panel(axis, letter, title, "missing early/late rollouts")
        _panel_letter(polar_axis, "q")
        polar_axis.set_axis_off()
        return {}
    if early == late:
        result.notes.append(
            f"early and late panels use the same checkpoint {early}; "
            "run the behavior stage on more checkpoints to separate them"
        )

    flow_rows: list[dict[str, Any]] = []
    polar_rows: list[dict[str, Any]] = []
    angle_rows: list[dict[str, Any]] = []
    pooled: dict[str, Any] = {}
    for stage, (checkpoint, per_seed) in stages.items():
        pooled[stage] = sources.pooled_samples(per_seed.values())
        flow_rows.extend(
            sources.flow_rows(
                pooled[stage],
                stage=stage,
                checkpoint=checkpoint,
                vision_radius=config.vision_radius,
            )
        )
        polar_rows.extend(
            sources.polar_stage_rows(
                pooled[stage], stage=stage, checkpoint=checkpoint, bins=config.polar_bins
            )
        )
        angle_rows.extend(
            sources.angle_rows_by_seed(
                per_seed, stage=stage, checkpoint=checkpoint, bin_width=config.angle_bin_width
            )
        )
    # One arrow scale for both stages, so early and late stay comparable.
    longest = max(
        (row["movement_x"] ** 2 + row["movement_y"] ** 2) ** 0.5 for row in flow_rows
    )
    arrow_scale = max(longest, 1e-6) / 0.85
    for axis, stage, letter, title in (
        (early_axis, "early", "o", "Early stage of training"),
        (late_axis, "late", "p", "Late stage of training"),
    ):
        _flow_panel(
            axis,
            [row for row in flow_rows if row["stage"] == stage],
            letter=letter,
            title=title,
            vision_radius=config.vision_radius,
            arrow_scale=arrow_scale,
        )
    _polar_panel(polar_axis, polar_rows)
    angle_comparisons = _angle_panel(angle_axis, angle_rows, config)
    result.rendering["flow_arrow_longest_step"] = longest
    result.rendering["flow_arrow_display_cells"] = 0.85
    return {
        "fig5_flow_field": flow_rows,
        "fig5_polar": polar_rows,
        "fig5_angles": angle_rows,
        "fig5_angle_comparisons": angle_comparisons,
    }


def _curve(
    axis: Any,
    x: Sequence[float],
    mean: Sequence[float],
    sem: Sequence[float],
    color: str,
) -> None:
    # A single evaluation checkpoint has no line to draw, so mark the point.
    marker = "o" if len(x) == 1 else None
    axis.plot(x, mean, color=color, linewidth=1.6, marker=marker, markersize=5, zorder=3)
    lower = [value - error for value, error in zip(mean, sem)]
    upper = [value + error for value, error in zip(mean, sem)]
    axis.fill_between(x, lower, upper, color=color, alpha=0.25, linewidth=0, zorder=2)


def _training_axis(axis: Any, updates: Sequence[int], x_style: str) -> None:
    if not updates:
        return
    if x_style == "log" and min(updates) > 0 and min(updates) != max(updates):
        axis.set_xscale("log")
    low, high = min(updates), max(updates)
    if low == high:
        low, high = low - 0.5, high + 0.5
    axis.set_xlim(low, high)
    ticks = _axis_ticks(updates, x_style)
    axis.set_xticks(ticks)
    axis.set_xticklabels([f"{tick:,}" for tick in ticks], rotation=90, fontsize=7)
    axis.minorticks_off()
    for index in range(0, len(ticks) - 1, 2):
        axis.axvspan(ticks[index], ticks[index + 1], color=BAND_COLOR, zorder=0, linewidth=0)


def _axis_ticks(updates: Sequence[int], x_style: str) -> list[int]:
    unique = sorted(set(updates))
    if len(unique) <= 12:
        return unique
    if x_style == "linear":
        step = max(1, len(unique) // 10)
        return unique[::step]
    lowest, highest = unique[0], unique[-1]
    targets = [lowest * (highest / lowest) ** (index / 11) for index in range(12)]
    ticks: list[int] = []
    for target in targets:
        nearest = min(unique, key=lambda value: abs(value - target))
        if nearest not in ticks:
            ticks.append(nearest)
    return ticks


def _box_panel(
    axis: Any,
    groups: dict[str, list[float]],
    comparison: dict[str, Any],
    ylabel: str,
) -> None:
    labels = ["Social", "Non-social"]
    values = [groups["social"], groups["non_social"]]
    if not all(values):
        axis.axis("off")
        return
    plot = axis.boxplot(
        values,
        widths=0.55,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.2},
        whiskerprops={"color": "black"},
        capprops={"color": "black"},
        flierprops={"marker": "o", "markersize": 3, "markerfacecolor": "black"},
    )
    for patch, color in zip(plot["boxes"], (SOCIAL_COLOR, NON_SOCIAL_COLOR)):
        patch.set_facecolor(color)
        patch.set_edgecolor("black")
        patch.set_linewidth(1.0)
    axis.set_xticks([1, 2])
    axis.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axis.set_ylabel(ylabel, fontsize=8)
    axis.tick_params(axis="y", labelsize=8)
    stars = comparison.get("stars") or ""
    if stars:
        top = max(max(values[0]), max(values[1]))
        bottom = min(min(values[0]), min(values[1]))
        span = max(top - bottom, 1e-9)
        bar = top + span * 0.12
        axis.plot([1, 1, 2, 2], [bar, bar + span * 0.04, bar + span * 0.04, bar], color="black", linewidth=1.0)
        axis.text(1.5, bar + span * 0.06, stars, ha="center", va="bottom", fontsize=10)
        axis.set_ylim(bottom - span * 0.15, bar + span * 0.30)


def _flow_panel(
    axis: Any,
    rows: Sequence[dict[str, Any]],
    *,
    letter: str,
    title: str,
    vision_radius: int,
    arrow_scale: float,
) -> None:
    _panel_letter(axis, letter)
    axis.set_title(title, fontsize=11)
    axis.set_xlabel("Relative x distance")
    axis.set_ylabel("Relative y distance")
    axis.set_xlim(-vision_radius - 0.6, vision_radius + 0.6)
    axis.set_ylim(-vision_radius - 0.6, vision_radius + 0.6)
    axis.set_xticks(range(-vision_radius, vision_radius + 1))
    axis.set_yticks(range(-vision_radius, vision_radius + 1))
    axis.set_aspect("equal")
    drawn = [row for row in rows if row["samples"] > 0]
    if drawn:
        axis.quiver(
            [row["offset_x"] for row in drawn],
            [row["offset_y"] for row in drawn],
            [row["movement_x"] for row in drawn],
            [row["movement_y"] for row in drawn],
            color=SOCIAL_COLOR,
            angles="xy",
            scale_units="xy",
            scale=arrow_scale,
            width=0.012,
        )
    axis.plot([0], [0], marker="o", color="#2f4fbf", markersize=6, zorder=4)


def _polar_panel(axis: Any, rows: Sequence[dict[str, Any]]) -> None:
    _panel_letter(axis, "q")
    axis.set_theta_zero_location("E")
    axis.set_theta_direction(1)
    axis.tick_params(labelsize=7, pad=0.5)
    axis.set_rlabel_position(112.5)
    for stage, color, width_scale, zorder in (
        ("early", EARLY_COLOR, 0.92, 2),
        ("late", LATE_COLOR, 0.55, 3),
    ):
        selected = [row for row in rows if row["stage"] == stage]
        if not selected:
            continue
        axis.bar(
            [radians(row["bin_center"]) for row in selected],
            [row["probability"] for row in selected],
            width=radians(selected[0]["bin_width"]) * width_scale,
            color=color,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
            zorder=zorder,
        )
    axis.set_yticklabels([])


def _angle_panel(
    axis: Any,
    rows: Sequence[dict[str, Any]],
    config: Fig5Config,
) -> list[dict[str, Any]]:
    _panel_letter(axis, "r")
    axis.set_xlabel("Angle (degrees)")
    axis.set_ylabel("Probability")
    labels = sorted({row["label"] for row in rows}, key=lambda label: float(label.split("-")[0]))
    comparisons: list[dict[str, Any]] = []
    width = 0.38
    for index, label in enumerate(labels):
        for offset, (stage, color) in zip(
            (-width / 2, width / 2), (("early", EARLY_COLOR), ("late", LATE_COLOR))
        ):
            values = [
                row["probability"] for row in rows if row["label"] == label and row["stage"] == stage
            ]
            if not values:
                continue
            axis.bar(
                index + offset,
                sum(values) / len(values),
                width=width,
                color=color,
                edgecolor="black",
                linewidth=0.6,
                zorder=2,
            )
            axis.scatter(
                [index + offset] * len(values),
                values,
                s=8,
                color="black",
                zorder=3,
                linewidths=0,
            )
        early = [row["probability"] for row in rows if row["label"] == label and row["stage"] == "early"]
        late = [row["probability"] for row in rows if row["label"] == label and row["stage"] == "late"]
        if early and late:
            comparison = permutation_comparison(
                late, early, permutations=config.permutations, seed=config.seed
            )
            comparisons.append(
                {
                    "label": label,
                    "early_mean": comparison.right_mean,
                    "late_mean": comparison.left_mean,
                    "difference": comparison.difference,
                    "p_value": comparison.p_value,
                    "stars": comparison.stars,
                    "permutations": comparison.permutations,
                    "exact": comparison.exact,
                    "early_n": comparison.right_n,
                    "late_n": comparison.left_n,
                }
            )
            stars = significance_stars(comparison.p_value)
            if stars != "ns":
                height = max(max(early), max(late))
                axis.text(index, height * 1.08, stars, ha="center", va="bottom", fontsize=9)
    axis.set_xticks(range(len(labels)))
    axis.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
    axis.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color=EARLY_COLOR),
            plt.Rectangle((0, 0), 1, 1, color=LATE_COLOR),
        ],
        labels=["Early training", "Late training"],
        fontsize=8,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
    )
    return comparisons


def _condition_legend(figure: Figure) -> None:
    figure.legend(
        handles=[
            plt.Line2D([0], [0], color=SOCIAL_COLOR, linewidth=2),
            plt.Line2D([0], [0], color=NON_SOCIAL_COLOR, linewidth=2),
        ],
        labels=["Social environment / agents", "Non-social environment / agents"],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.975),
        ncol=2,
        frameon=False,
        fontsize=11,
    )
    figure.suptitle(
        "Figure 5 replication — emergent social interactions in artificial agents",
        fontsize=14,
        y=0.995,
    )


def _panel_letter(axis: Any, letter: str) -> None:
    axis.annotate(
        letter,
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(-38, 16),
        textcoords="offset points",
        fontsize=15,
        fontweight="bold",
        va="top",
    )


def _empty_panel(axis: Any, letter: str, title: str, message: str) -> None:
    _panel_letter(axis, letter)
    if title:
        axis.set_title(title, fontsize=11)
    axis.text(0.5, 0.5, message, ha="center", va="center", fontsize=9, color="#8a8a8a")
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def _scale_for(metric: str) -> float:
    for panel in TRAINING_PANELS:
        if panel.metric == metric:
            return panel.scale
    return 1.0


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelsize": 9,
            "axes.titlesize": 11,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 9,
            "font.size": 9,
            "savefig.facecolor": "white",
        }
    )
