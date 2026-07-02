"""Shared neural dimensions between paired agents via PLSC (paper Fig. 5/6, C3).

Partial Least Squares Correlation (PLSC) is the framework Zhang et al. (2025)
use for the inter-brain / inter-agent shared-dynamics analysis. For one trained
pair it takes the two agents' time-aligned RNN hidden states, z-scores each
unit, forms the cross-covariance matrix ``R = X_chaser^T X_explorer / (n - 1)``,
and takes its SVD. The singular values measure the strength of the shared
dimensions; the paired latent variables ``L_c = X_chaser U`` and
``L_e = X_explorer V`` are the agents' projections onto those shared dimensions.

This is the cross-agent counterpart to ``analyze_neural.py``: PCA there is an
SVD of one agent's data matrix (maximum variance within a network); PLSC here is
an SVD of the cross-covariance matrix (maximum covariance between networks). PCA
cannot see shared structure that does not lie along either network's own
high-variance directions.

Significance follows the paper: a temporal permutation breaks the moment-to-
moment pairing between the two agents (one agent's timepoints are shuffled) and
the SVD is recomputed many times to build a null distribution of singular values
per rank. A shared dimension is significant when its observed singular value
exceeds the null percentile for that rank. The paper's C3 claim is that social
pairs have more significant shared dimensions and a higher top-dimension
correlation than non-social controls; this script reports both, per pair and
aggregated by task.

Usage:

    uv run python scripts/analysis/analyze_shared_neural.py runs/<experiment>/paper_rollouts \
        --output-root runs/reports/<experiment>/shared_neural
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

from mouse_run_run.plsc import TOP_K, PLSCResult, compute_plsc
from mouse_run_run.provenance import collect_provenance, hash_file, write_json_atomic
from mouse_run_run.serialization import ROLLOUT_FORMAT, read_metadata

TASK_COLORS = {"social": "#d84f45", "non_social": "#2189a8"}
FIGURE_DPI = 150
DEFAULT_PERMUTATIONS = 200
DEFAULT_ALPHA = 0.05
MAX_SAMPLES = 20_000


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
        "--permutations",
        type=int,
        default=DEFAULT_PERMUTATIONS,
        help="Temporal-permutation samples for the null distribution.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help="Significance level; a dimension is significant when its singular"
        " value exceeds the (1 - alpha) null percentile for its rank.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=MAX_SAMPLES,
        help="Cap on pooled paired timesteps (deterministic subsample) to bound"
        " the permutation cost.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Permutation RNG seed; fixed so results are reproducible.",
    )
    parser.add_argument(
        "--include-degenerate",
        action="store_true",
        help="Keep paper-style degenerate episodes instead of excluding them.",
    )
    parser.add_argument("--max-files", type=int)
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
        record = _analyze_pair(
            path,
            figures_dir=figures_dir,
            permutations=args.permutations,
            alpha=args.alpha,
            max_samples=args.max_samples,
            seed=args.seed,
            include_degenerate=args.include_degenerate,
        )
        records.append(record)
        print(
            f"analyzed={path} status={record['status']}"
            f" significant_dims={record.get('significant_dims')}"
            f" top_corr={record.get('top_dim_correlation')}",
            flush=True,
        )

    aggregate = _aggregate_by_task(records)
    aggregate_figure = _plot_aggregate(aggregate, records, figures_dir)

    summary = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "analysis_protocol": "plsc_shared_neural_v1",
        "method": (
            "z-score each unit; R = X_chaser^T X_explorer / (n - 1); SVD(R);"
            " temporal-permutation null per singular-value rank"
        ),
        "permutations": args.permutations,
        "alpha": args.alpha,
        "max_samples": args.max_samples,
        "seed": args.seed,
        "include_degenerate": args.include_degenerate,
        "pairs": records,
        "aggregate_by_task": aggregate,
        "figures": [record["figure"] for record in records if record.get("figure")]
        + ([aggregate_figure] if aggregate_figure else []),
    }
    summary_path = output_root / "shared_neural_summary.json"
    write_json_atomic(summary_path, summary)

    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "inputs": [str(path) for path in rollout_paths],
        "input_hashes": {str(path): hash_file(path) for path in rollout_paths},
        "outputs": {"summary": str(summary_path), "figures": summary["figures"]},
        "analysis_script": str(Path(__file__).resolve()),
        "provenance": collect_provenance(cwd=Path.cwd()),
    }
    write_json_atomic(output_root / "MANIFEST.json", manifest)
    print(json.dumps({"summary": str(summary_path), "pairs": len(records)}, indent=2))


# ---------------------------------------------------------------------------
# PLSC per trained pair
# ---------------------------------------------------------------------------


def _analyze_pair(
    path: Path,
    *,
    figures_dir: Path,
    permutations: int,
    alpha: float,
    max_samples: int,
    seed: int,
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
    chaser = tensors["chaser_hidden"][:, valid].float().reshape(-1, tensors["chaser_hidden"].shape[2])
    explorer = tensors["explorer_hidden"][:, valid].float().reshape(-1, tensors["explorer_hidden"].shape[2])
    distance = _state_series(tensors["distance"], steps)[:, valid].reshape(-1)

    chaser, explorer, distance = _subsample(chaser, explorer, distance, limit=max_samples)
    if chaser.shape[0] < 8 or _degenerate_matrix(chaser) or _degenerate_matrix(explorer):
        record["status"] = "insufficient_variation"
        return record

    result = compute_plsc(
        chaser,
        explorer,
        permutations=permutations,
        alpha=alpha,
        seed=seed,
    )
    record.update(
        {
            "status": "ok",
            "pooled_samples": int(chaser.shape[0]),
            "hidden_size": int(chaser.shape[1]),
            "singular_values": [round(float(x), 5) for x in result.singular_values[:TOP_K]],
            "null_p95": [round(float(x), 5) for x in result.null_percentile[:TOP_K]],
            "covariance_explained": [round(float(x), 5) for x in result.covariance_explained[:TOP_K]],
            "latent_correlations": [round(float(x), 5) for x in result.latent_correlations[:TOP_K]],
            "dimension_p_values": [round(float(x), 5) for x in result.p_values[:TOP_K]],
            "significant_dims": result.significant_dims,
            "leading_significant_dims": result.leading_significant_dims,
            "top_dim_correlation": round(float(result.latent_correlations[0]), 5),
        }
    )

    figure_path = figures_dir / f"plsc__{path.stem}.png"
    _plot_pair(result, distance, task=task, stem=path.stem, output=figure_path)
    record["figure"] = str(figure_path)
    return record


def _degenerate_matrix(matrix: torch.Tensor) -> bool:
    return bool((matrix.std(dim=0) < 1e-8).all())


def _subsample(
    *tensors: torch.Tensor,
    limit: int,
) -> tuple[torch.Tensor, ...]:
    rows = tensors[0].shape[0]
    if rows <= limit:
        return tensors
    indices = torch.linspace(0, rows - 1, limit).long()
    return tuple(tensor[indices] for tensor in tensors)


def _state_series(tensor: torch.Tensor, steps: int) -> torch.Tensor:
    return tensor[:steps]


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _aggregate_by_task(records: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = (
        "significant_dims",
        "leading_significant_dims",
        "top_dim_correlation",
    )
    ok_records = [record for record in records if record.get("status") == "ok"]
    aggregate: dict[str, Any] = {}
    for task in sorted({str(record["task"]) for record in ok_records}):
        task_records = [record for record in ok_records if str(record["task"]) == task]
        aggregate[task] = {
            "n": len(task_records),
            "metrics": {
                metric: _mean_std([float(record[metric]) for record in task_records])
                for metric in metrics
            },
            "mean_covariance_explained_top5": _mean_std(
                [
                    float(sum(record["covariance_explained"][:5]))
                    for record in task_records
                ]
            ),
        }
    return aggregate


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _plot_pair(
    result: PLSCResult,
    distance: torch.Tensor,
    *,
    task: str,
    stem: str,
    output: Path,
) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(17, 5.2), constrained_layout=True)
    color = TASK_COLORS.get(task, "#202426")
    top = min(TOP_K, result.singular_values.shape[0])
    components = range(1, top + 1)

    scree = axes[0]
    singular = result.singular_values[:top].numpy()
    null = result.null[:, :top].numpy()
    null_low = null.min(axis=0)
    null_p95 = result.null_percentile[:top].numpy()
    scree.fill_between(
        list(components),
        null_low,
        null_p95,
        color="#697177",
        alpha=0.2,
        linewidth=0,
        label=f"permutation null (≤{int((1 - result.alpha) * 100)}%)",
    )
    scree.plot(list(components), singular, marker="o", markersize=4, color=color, label="observed")
    significant_mask = singular > null_p95
    if significant_mask.any():
        significant_components = [c for c, flag in zip(components, significant_mask) if flag]
        scree.scatter(
            significant_components,
            singular[significant_mask],
            s=70,
            facecolors="none",
            edgecolors=color,
            linewidths=1.6,
            label="significant",
        )
    scree.set_title(
        f"shared-dimension spectrum ({result.significant_dims} significant)",
        fontsize=10,
    )
    scree.set_xlabel("shared dimension", fontsize=9)
    scree.set_ylabel("cross-covariance singular value", fontsize=9)
    scree.tick_params(labelsize=8)
    scree.legend(fontsize=8)
    scree.grid(alpha=0.25, linewidth=0.5)

    scatter_axis = axes[1]
    latent_c = result.latent_c0.numpy()
    latent_e = result.latent_e0.numpy()
    scatter = scatter_axis.scatter(
        latent_c,
        latent_e,
        c=distance.numpy(),
        cmap="magma",
        s=max(4.0, min(24.0, 4000.0 / max(1, latent_c.shape[0]))),
        alpha=0.55,
        linewidths=0,
    )
    scatter_axis.set_title(
        f"top shared dimension (r = {result.latent_correlations[0]:.2f})",
        fontsize=10,
    )
    scatter_axis.set_xlabel("chaser latent variable 1", fontsize=9)
    scatter_axis.set_ylabel("explorer latent variable 1", fontsize=9)
    scatter_axis.tick_params(labelsize=8)
    figure.colorbar(scatter, ax=scatter_axis, shrink=0.85, label="distance")

    corr_axis = axes[2]
    correlations = result.latent_correlations[:top].numpy()
    corr_axis.bar(list(components), correlations, color=color, alpha=0.85)
    corr_axis.set_title("paired latent-variable correlation", fontsize=10)
    corr_axis.set_xlabel("shared dimension", fontsize=9)
    corr_axis.set_ylabel("Pearson r", fontsize=9)
    corr_axis.tick_params(labelsize=8)
    corr_axis.grid(axis="y", alpha=0.25, linewidth=0.5)

    figure.suptitle(f"{task} — PLSC shared neural dimensions — {stem}", fontsize=12)
    figure.savefig(output, dpi=FIGURE_DPI)
    plt.close(figure)


def _plot_aggregate(
    aggregate: dict[str, Any],
    records: list[dict[str, Any]],
    figures_dir: Path,
) -> str | None:
    import matplotlib.pyplot as plt

    tasks = sorted(aggregate)
    if not tasks:
        return None

    figure, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    bar_metrics = (
        ("significant_dims", "significant shared dimensions"),
        ("top_dim_correlation", "top-dimension correlation"),
    )
    for axis, (metric, title) in zip(axes[:2], bar_metrics, strict=True):
        means = []
        stds = []
        colors = []
        labels = []
        for task in tasks:
            stats = aggregate[task]["metrics"][metric]
            means.append(stats["mean"] or 0.0)
            stds.append(stats.get("std") or 0.0)
            colors.append(TASK_COLORS.get(task, "#202426"))
            labels.append(f"{task}\n(n={aggregate[task]['n']})")
        axis.bar(range(len(tasks)), means, yerr=stds, capsize=4, color=colors, error_kw={"linewidth": 0.9})
        axis.set_xticks(range(len(tasks)), labels, fontsize=9)
        axis.set_title(title, fontsize=10)
        axis.tick_params(labelsize=8)
        axis.grid(axis="y", alpha=0.25, linewidth=0.5)

    spectrum_axis = axes[2]
    for task in tasks:
        task_records = [
            record
            for record in records
            if record.get("status") == "ok" and str(record["task"]) == task
        ]
        if not task_records:
            continue
        length = min(len(record["singular_values"]) for record in task_records)
        stacked = torch.tensor(
            [record["singular_values"][:length] for record in task_records]
        )
        mean = stacked.mean(dim=0).numpy()
        components = range(1, length + 1)
        spectrum_axis.plot(
            list(components),
            mean,
            marker="o",
            markersize=3,
            color=TASK_COLORS.get(task, "#202426"),
            label=f"{task} (n={len(task_records)})",
        )
    spectrum_axis.set_title("mean shared-dimension spectrum", fontsize=10)
    spectrum_axis.set_xlabel("shared dimension", fontsize=9)
    spectrum_axis.set_ylabel("singular value", fontsize=9)
    spectrum_axis.tick_params(labelsize=8)
    spectrum_axis.legend(fontsize=8)
    spectrum_axis.grid(alpha=0.25, linewidth=0.5)

    figure.suptitle(
        "PLSC shared neural dimensions by task (mean ± std across pairs)", fontsize=12
    )
    path = figures_dir / "shared_neural_aggregate.png"
    figure.savefig(path, dpi=FIGURE_DPI)
    plt.close(figure)
    return str(path)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


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
    return Path("runs/reports") / name / "shared_neural"


def _rollout_task(metadata: dict[str, Any]) -> str:
    env_config = metadata.get("env_config") or {}
    task = env_config.get("task")
    if task is None:
        checkpoint_env = (metadata.get("checkpoint_config") or {}).get("env") or {}
        task = checkpoint_env.get("task")
    return str(task)


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
