from __future__ import annotations

import argparse
import json
import math
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from _common import REPO_ROOT, REPORTS_ROOT
from mouse_run_run.provenance import collect_provenance, hash_file, read_jsonl, write_json_atomic


# Match the viewer palette so figures and the interactive viewer read the same.
TASK_COLORS = {"social": "#d84f45", "non_social": "#2189a8"}
FIGURE_DPI = 150


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tables_root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip matplotlib figure generation (report tables only).",
    )
    args = parser.parse_args()

    tables_root = args.tables_root
    output_root = args.output_root or REPORTS_ROOT / tables_root.resolve().name
    output_root.mkdir(parents=True, exist_ok=True)

    runs = read_jsonl(tables_root / "runs.jsonl")
    evaluations = read_jsonl(tables_root / "evaluations.jsonl")
    exclusions = read_jsonl(tables_root / "exclusions.jsonl")

    figures = [] if args.no_figures else _generate_figures(output_root, runs, evaluations)

    summary = {
        "schema_version": 2,
        "generated_at": datetime.now(UTC).isoformat(),
        "tables_root": str(tables_root),
        "valid_pairs_by_task": _valid_pairs_by_task(runs),
        "failed_attempts": [row for row in runs if row.get("status") == "failed"],
        "training_metrics": _training_metrics(runs),
        "evaluation_metrics": _evaluation_metrics(evaluations),
        "rollout_exclusions": _rollout_exclusions(exclusions),
        "figures": figures,
    }
    summary_path = output_root / "summary.json"
    report_path = output_root / "report.md"
    write_json_atomic(summary_path, summary)
    report_path.write_text(_render_report(summary), encoding="utf-8")

    manifest = {
        "schema_version": 2,
        "generated_at": datetime.now(UTC).isoformat(),
        "input_tables": {
            name: str(tables_root / name)
            for name in ("runs.jsonl", "evaluations.jsonl", "rollouts.jsonl", "exclusions.jsonl")
        },
        "input_hashes": {
            name: hash_file(tables_root / name)
            for name in ("runs.jsonl", "evaluations.jsonl", "rollouts.jsonl", "exclusions.jsonl")
            if (tables_root / name).exists()
        },
        "outputs": {
            "summary": str(summary_path),
            "report": str(report_path),
            "figures": [figure["path"] for figure in figures],
        },
        "output_hashes": {
            figure["name"]: hash_file(Path(figure["path"]))
            for figure in figures
            if Path(figure["path"]).exists()
        },
        "analysis_script": str(Path(__file__).resolve()),
        "provenance": collect_provenance(cwd=REPO_ROOT),
    }
    write_json_atomic(output_root / "MANIFEST.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


def _valid_pairs_by_task(runs: list[dict[str, Any]]) -> dict[str, int]:
    valid_units: dict[str, set[str]] = {}
    for row in runs:
        if row.get("status") != "completed" or not row.get("healthy") or not row.get("checkpoint_ok"):
            continue
        task = str(row.get("task"))
        valid_units.setdefault(task, set()).add(str(row.get("unit_id")))
    return {task: len(units) for task, units in sorted(valid_units.items())}


TRAINING_METRIC_KEYS = (
    "collisions_per_episode",
    "chaser_return",
    "explorer_return",
    "chaser_partner_vision",
    "explorer_partner_vision",
    "chaser_new_fields",
    "explorer_new_fields",
    "final_distance",
    "policy_loss",
    "value_loss",
    "entropy",
    "approx_kl",
    "chaser_grad_norm",
    "explorer_grad_norm",
    "max_param_abs",
)

TRAINING_CURVE_KEYS = (
    "collisions_per_episode",
    "chaser_return",
    "explorer_return",
    "final_distance",
    "chaser_partner_vision",
    "explorer_partner_vision",
    "entropy",
    "value_loss",
)

EVALUATION_METRIC_KEYS = (
    "collisions_per_episode",
    "chaser_return",
    "explorer_return",
    "chaser_partner_vision",
    "explorer_partner_vision",
    "chaser_new_fields",
    "explorer_new_fields",
    "final_distance",
    "average_distance",
    "degenerate_fraction",
)


def _training_metrics(runs: list[dict[str, Any]]) -> dict[str, Any]:
    # One row per unit: the latest successful attempt, materialized by
    # build_tables. Averaging over every completed attempt would double-count
    # units that completed more than once.
    metrics = [row for row in runs if row.get("is_latest_successful")]
    output: dict[str, Any] = {}
    for task in sorted({str(row.get("task")) for row in metrics}):
        task_rows = [row for row in metrics if row.get("task") == task]
        output[task] = {
            key: _mean_std(
                [
                    float(row.get("metrics", {}).get(key))
                    for row in task_rows
                    if row.get("metrics", {}).get(key) is not None
                ]
            )
            for key in TRAINING_METRIC_KEYS
        }
        output[task]["n"] = len(task_rows)
    return output


def _evaluation_metrics(evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    # The research question is social vs non_social, so evaluations are
    # grouped by task first and opponent mode second; pooling tasks would
    # answer nothing.
    output: dict[str, Any] = {}
    ok_rows = [row for row in evaluations if row.get("status") == "ok"]
    for task in sorted({_row_task(row) for row in ok_rows}):
        task_rows = [row for row in ok_rows if _row_task(row) == task]
        task_output: dict[str, Any] = {}
        for mode in sorted({str(row.get("opponent_mode")) for row in task_rows}):
            rows = [row for row in task_rows if row.get("opponent_mode") == mode]
            task_output[mode] = {
                key: _mean_std(
                    [
                        float(row.get("metrics", {}).get(key))
                        for row in rows
                        if row.get("metrics", {}).get(key) is not None
                    ]
                )
                for key in EVALUATION_METRIC_KEYS
            }
            task_output[mode]["n"] = len(rows)
        output[task] = task_output
    output["failed_count"] = len([row for row in evaluations if row.get("status") != "ok"])
    return output


def _row_task(row: dict[str, Any]) -> str:
    """Evaluation rows before schema v2 lacked the identity columns; fall back
    to the task recorded in the checkpoint's own config metadata."""
    task = row.get("task")
    if task is None:
        task = ((row.get("checkpoint_config") or {}).get("env") or {}).get("task")
    return str(task)


def _rollout_exclusions(exclusions: list[dict[str, Any]]) -> dict[str, Any]:
    excluded = sum(int(row.get("excluded_count", 0)) for row in exclusions)
    total = sum(int(row.get("total_count", 0)) for row in exclusions)
    return {
        "excluded_degenerate_episodes": excluded,
        "episodes_with_exclusion_records": total,
        "records": len(exclusions),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _generate_figures(
    output_root: Path,
    runs: list[dict[str, Any]],
    evaluations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    import matplotlib

    matplotlib.use("Agg")

    figures_dir = output_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    figures = []
    for figure in (
        _plot_training_curves(runs, figures_dir),
        _plot_evaluation_comparison(evaluations, figures_dir),
    ):
        if figure is not None:
            figures.append(figure)
    return figures


def _plot_training_curves(
    runs: list[dict[str, Any]],
    figures_dir: Path,
) -> dict[str, Any] | None:
    """Mean +/- std training curves across valid pairs, social vs non_social."""
    import matplotlib.pyplot as plt

    # task -> metric -> update -> values across pairs
    histories: dict[str, dict[str, dict[int, list[float]]]] = {}
    pairs_read = 0
    for row in runs:
        if not row.get("is_latest_successful"):
            continue
        run_dir = Path(str(row.get("run_dir", "")))
        if not run_dir.is_absolute():
            # Tables built before run_dir normalization store repo-root-relative
            # paths; resolve them so figures do not silently lose data when the
            # script runs from another cwd.
            run_dir = REPO_ROOT / run_dir
        metrics_path = run_dir / "metrics.jsonl"
        if not metrics_path.exists():
            continue
        pairs_read += 1
        task = str(row.get("task"))
        for record in read_jsonl(metrics_path):
            update = record.get("update")
            metrics = record.get("metrics") or {}
            if update is None:
                continue
            for key in TRAINING_CURVE_KEYS:
                value = metrics.get(key)
                if value is None or not math.isfinite(float(value)):
                    continue
                histories.setdefault(task, {}).setdefault(key, {}).setdefault(
                    int(update), []
                ).append(float(value))
    if not histories:
        return None

    figure, axes = plt.subplots(2, 4, figsize=(18, 7.5), constrained_layout=True)
    for axis, key in zip(axes.flat, TRAINING_CURVE_KEYS, strict=True):
        for task, metric_map in sorted(histories.items()):
            series = metric_map.get(key)
            if not series:
                continue
            updates = sorted(series)
            means = [statistics.fmean(series[update]) for update in updates]
            stds = [
                statistics.pstdev(series[update]) if len(series[update]) > 1 else 0.0
                for update in updates
            ]
            color = TASK_COLORS.get(task)
            axis.plot(updates, means, label=task, color=color, linewidth=1.3)
            axis.fill_between(
                updates,
                [mean - std for mean, std in zip(means, stds, strict=True)],
                [mean + std for mean, std in zip(means, stds, strict=True)],
                color=color,
                alpha=0.18,
                linewidth=0,
            )
        axis.set_title(key.replace("_", " "), fontsize=10)
        axis.set_xlabel("update", fontsize=8)
        axis.tick_params(labelsize=8)
        axis.grid(alpha=0.25, linewidth=0.5)
    axes.flat[0].legend(fontsize=8)
    figure.suptitle("Training curves (mean ± std across valid pairs)", fontsize=12)

    path = figures_dir / "training_curves.png"
    figure.savefig(path, dpi=FIGURE_DPI)
    plt.close(figure)
    return {
        "name": "training_curves",
        "path": str(path),
        "caption": f"Training curves, mean ± std across {pairs_read} valid pairs.",
    }


def _plot_evaluation_comparison(
    evaluations: list[dict[str, Any]],
    figures_dir: Path,
) -> dict[str, Any] | None:
    """Random-opponent evaluation metrics as task-grouped bars per opponent mode."""
    import matplotlib.pyplot as plt

    ok_rows = [row for row in evaluations if row.get("status") == "ok"]
    if not ok_rows:
        return None
    tasks = sorted({_row_task(row) for row in ok_rows})
    modes = sorted({str(row.get("opponent_mode")) for row in ok_rows})
    if not tasks or not modes:
        return None

    figure, axes = plt.subplots(2, 5, figsize=(20, 7.5), constrained_layout=True)
    bar_width = 0.8 / len(tasks)
    for axis, key in zip(axes.flat, EVALUATION_METRIC_KEYS, strict=True):
        for task_index, task in enumerate(tasks):
            positions = []
            means = []
            stds = []
            for mode_index, mode in enumerate(modes):
                values = [
                    float(row.get("metrics", {}).get(key))
                    for row in ok_rows
                    if _row_task(row) == task
                    and str(row.get("opponent_mode")) == mode
                    and row.get("metrics", {}).get(key) is not None
                ]
                values = [value for value in values if math.isfinite(value)]
                if not values:
                    continue
                positions.append(mode_index + task_index * bar_width)
                means.append(statistics.fmean(values))
                stds.append(statistics.pstdev(values) if len(values) > 1 else 0.0)
            if not positions:
                continue
            axis.bar(
                positions,
                means,
                width=bar_width * 0.92,
                yerr=stds,
                capsize=2,
                label=task,
                color=TASK_COLORS.get(task),
                error_kw={"linewidth": 0.8},
            )
        axis.set_title(key.replace("_", " "), fontsize=10)
        axis.set_xticks(
            [index + bar_width * (len(tasks) - 1) / 2 for index in range(len(modes))],
            [mode.replace("_", " ") for mode in modes],
            fontsize=8,
            rotation=12,
        )
        axis.tick_params(labelsize=8)
        axis.grid(axis="y", alpha=0.25, linewidth=0.5)
    axes.flat[0].legend(fontsize=8)
    figure.suptitle(
        "Random-opponent evaluation (mean ± std across valid pairs)", fontsize=12
    )

    path = figures_dir / "evaluation_comparison.png"
    figure.savefig(path, dpi=FIGURE_DPI)
    plt.close(figure)
    return {
        "name": "evaluation_comparison",
        "path": str(path),
        "caption": (
            f"Evaluation metrics over {len(ok_rows)} successful evaluations,"
            f" grouped by opponent mode and task."
        ),
    }


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------


def _render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# MARL Experiment Summary",
        "",
        f"Generated: `{summary['generated_at']}`",
        f"Tables: `{summary['tables_root']}`",
        "",
        "## Valid Pairs",
        "",
        "| Task | Valid pairs |",
        "| --- | ---: |",
    ]
    for task, count in summary["valid_pairs_by_task"].items():
        lines.append(f"| `{task}` | {count} |")

    lines.extend(["", "## Training Metrics (final update, mean ± std across pairs)", ""])
    lines.extend(_metric_table(summary["training_metrics"], TRAINING_METRIC_KEYS))

    lines.extend(["", "## Random-Opponent Evaluation"])
    evaluation = summary["evaluation_metrics"]
    tasks = [task for task in sorted(evaluation) if task != "failed_count"]
    modes = sorted({mode for task in tasks for mode in evaluation[task]})
    for mode in modes:
        lines.extend(["", f"### Opponent: `{mode}`", ""])
        per_task = {
            task: evaluation[task][mode] for task in tasks if mode in evaluation[task]
        }
        lines.extend(_metric_table(per_task, EVALUATION_METRIC_KEYS))
    lines.append("")
    lines.append(f"Failed evaluations: {evaluation.get('failed_count', 0)}")

    figures = summary.get("figures") or []
    if figures:
        lines.extend(["", "## Figures"])
        for figure in figures:
            relative = Path(figure["path"]).name
            lines.extend(
                [
                    "",
                    f"### {figure['name'].replace('_', ' ')}",
                    "",
                    f"![{figure['name']}](figures/{relative})",
                    "",
                    figure["caption"],
                ]
            )

    exclusions = summary["rollout_exclusions"]
    lines.extend(
        [
            "",
            "## Degenerate Episode Exclusions",
            "",
            "| Excluded episodes | Episodes with records | Records |",
            "| ---: | ---: | ---: |",
            (
                f"| {exclusions['excluded_degenerate_episodes']}"
                f" | {exclusions['episodes_with_exclusion_records']}"
                f" | {exclusions['records']} |"
            ),
            "",
            "## Failed Attempts",
            f"- count: {len(summary['failed_attempts'])}",
        ]
    )
    for row in summary["failed_attempts"]:
        error = str(row.get("error") or "unknown error").splitlines()[0][:160]
        lines.append(f"- `{row.get('unit_id')}` `{row.get('attempt_id')}`: {error}")
    return "\n".join(lines) + "\n"


def _metric_table(
    per_task: dict[str, Any],
    metric_keys: tuple[str, ...],
) -> list[str]:
    tasks = [task for task in sorted(per_task) if isinstance(per_task[task], dict)]
    if not tasks:
        return ["(no data)"]
    header = "| Metric | " + " | ".join(f"`{task}` (n={per_task[task].get('n', 0)})" for task in tasks) + " |"
    divider = "| --- |" + " ---: |" * len(tasks)
    rows = [header, divider]
    for key in metric_keys:
        cells = [_format_stats(per_task[task].get(key)) for task in tasks]
        rows.append(f"| {key.replace('_', ' ')} | " + " | ".join(cells) + " |")
    return rows


def _format_stats(stats: Any) -> str:
    if not isinstance(stats, dict) or stats.get("mean") is None:
        return "-"
    mean = _format_number(stats["mean"])
    std = _format_number(stats.get("std") or 0.0)
    return f"{mean} ± {std}"


def _format_number(value: float) -> str:
    if value == 0:
        return "0"
    if abs(value) >= 1000 or abs(value) < 0.001:
        return f"{value:.3g}"
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _mean_std(values: list[float]) -> dict[str, float | int | None]:
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
