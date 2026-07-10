from __future__ import annotations

import base64
import json
import math
import time
from pathlib import Path

import torch
from safetensors import SafetensorError
from torch.nn import functional as F

from mouse_run_run.checkpoint_loading import load_policy_pair
from mouse_run_run.degenerate import episode_degeneracy
from mouse_run_run.env import BatchedChaseEnv
from mouse_run_run.evaluate import OpponentMode
from mouse_run_run.plsc import compute_plsc, project_norms, shared_variance_fraction
from mouse_run_run.rollout import collect_batch
from mouse_run_run.serialization import CHECKPOINT_FORMAT, read_metadata
from mouse_run_run.training_config import select_device


ACTION_LABELS = ("U", "R", "D", "L")


def checkpoint_listing(runs_root: Path, project_root: Path) -> list[dict[str, object]]:
    if not runs_root.exists():
        return []
    checkpoints = []
    for path in runs_root.rglob("*.safetensors"):
        try:
            metadata = read_metadata(path)
        except (OSError, ValueError, SafetensorError):
            continue
        if metadata.get("format") != CHECKPOINT_FORMAT:
            continue
        experiment = _infer_experiment(path, runs_root)
        # Show only the study experiments in the explorer; stray smoke/ad-hoc
        # runs would collide with real architectures in the architecture -> seed
        # -> step cascade.
        if experiment not in ARCHITECTURE_LABELS:
            continue
        config = _loads_metadata(metadata, "config")
        metrics = _loads_metadata(metadata, "metrics")
        stat = path.stat()
        architecture = config.get("architecture", "rnn")
        checkpoints.append(
            {
                "path": _display_path(path.resolve(), project_root),
                "name": path.name,
                "experiment": experiment,
                "architecture": architecture,
                "architecture_label": ARCHITECTURE_LABELS.get(
                    experiment, (architecture.upper(), "")
                )[0],
                "task": (config.get("env") or {}).get("task"),
                "seed": _infer_seed(path),
                "update": _infer_update(path),
                "is_latest": path.stem in ("latest", "final"),
                "size_bytes": stat.st_size,
                "modified_unix": stat.st_mtime,
                "modified": time.strftime(
                    "%Y-%m-%d %H:%M:%S",
                    time.localtime(stat.st_mtime),
                ),
                "metrics": _json_finite(metrics),
            }
        )
    checkpoints.sort(key=lambda item: float(item["modified_unix"]), reverse=True)
    return checkpoints


ARCHITECTURE_LABELS = {
    "mouse-run-run-1": ("RNN", "vanilla ReLU RNN (paper)"),
    "mouse-run-run-2-mlp": ("MLP", "8-frame-window MLP (finite-history floor)"),
    "mouse-run-run-2-ssm": ("SSM", "gated linear recurrence"),
    "mouse-run-run-2-transformer": ("Transformer", "causal transformer"),
}
ARCHITECTURE_ORDER = ("RNN", "MLP", "SSM", "Transformer")


def build_report(runs_root: Path, project_root: Path) -> dict[str, object]:
    """Aggregate the pre-computed experiment/analysis artifacts into a paper-
    aligned report payload: behavioral (Fig. 5), neural (Fig. 6), and the
    cross-architecture comparison. Reads only JSON/markdown/PNG already on
    disk; computes nothing."""
    reports_root = runs_root / "reports"
    analyses_root = runs_root / "analyses"
    experiments = []
    for exp_dir in sorted(p for p in reports_root.glob("*") if p.is_dir()) if reports_root.exists() else []:
        name = exp_dir.name
        # The report is the paper-aligned cross-architecture study; skip
        # unrelated experiments (smoke tests, ad-hoc runs).
        if name not in ARCHITECTURE_LABELS:
            continue
        summary = _read_json_or_none(exp_dir / "summary.json")
        if summary is None:
            continue
        neural = _read_json_or_none(analyses_root / name / "neural_summary.json")
        label, description = ARCHITECTURE_LABELS.get(name, (name, ""))
        experiments.append(
            {
                "experiment": name,
                "architecture": label,
                "description": description,
                "valid_pairs": summary.get("valid_pairs_by_task", {}),
                "behavior": _behavior_rows(summary.get("evaluation_metrics", {})),
                "training": _training_rows(summary.get("training_metrics", {})),
                "neural": _neural_rows(neural),
                "exclusions": summary.get("rollout_exclusions", {}),
            }
        )
    experiments.sort(key=lambda e: _arch_rank(e["architecture"]))

    cross = {
        "cka": _read_json_or_none(analyses_root / "cross_arch_cka.json"),
        "per_unit": _read_json_or_none(analyses_root / "cross_arch_perunit.json"),
        "architecture_order": list(ARCHITECTURE_ORDER),
    }
    figures = {}
    figures_dir = analyses_root / "figures"
    if figures_dir.exists():
        for png in sorted(figures_dir.glob("*.png")):
            figures[png.stem] = "data:image/png;base64," + base64.b64encode(png.read_bytes()).decode()

    report_md = None
    report_path = project_root / "docs" / "paper-notes" / "cross-architecture-representation-study.md"
    if report_path.exists():
        report_md = report_path.read_text(encoding="utf-8")

    alignment_md = None
    alignment_path = project_root / "docs" / "paper-notes" / "paper-claim-alignment.md"
    if alignment_path.exists():
        alignment_md = alignment_path.read_text(encoding="utf-8")

    paper_reference = _read_json_or_none(
        project_root / "docs" / "paper-notes" / "original-paper-reference.json"
    )

    return {
        "experiments": experiments,
        "cross_architecture": cross,
        "figures": figures,
        "paper_reference": paper_reference,
        "report_markdown": report_md,
        "claim_alignment": _claim_alignment(analyses_root),
        "alignment_markdown": alignment_md,
    }


CLAIM_ALIGNMENT = [
    ("C1", "Social rewards produce social behavior", "reproduces",
     "Across all four architectures, social chasers collide more and social explorers evade more effectively than non-social controls."),
    ("C2", "The network encodes social events", "partial",
     "Social-event decoding is strong, but our non-social collision decoder is also above chance (0.765 and 0.747), unlike the paper's chance-level control."),
    ("C3", "Shared dimensions emerge between the two agents", "partial",
     "The social/non-social gap in leading PLSC correlation reproduces (0.743 vs 0.067), but the significant-dimension count does not (183 vs 108)."),
    ("C4", "The chaser represents its partner; it predicts performance", "partial",
     "Partner information is much stronger in social chasers and correlates with collisions (r=+0.64), but it decreases rather than increases over training."),
    ("C5", "Shared dimensions selectively drive social behavior", "inconclusive",
     "Shared-dimension removal impairs behavior, but the nominal random-basis control is equally or more disruptive and its construction is not fully paper-faithful."),
]


def _claim_alignment(analyses_root: Path) -> dict[str, object]:
    c4 = _read_json_or_none(analyses_root / "c4_mouse-run-run-1.json")
    c5 = {}
    c5_rnn_detail = {}
    for name, exp in (("RNN", "mouse-run-run-1"), ("MLP", "mouse-run-run-2-mlp"),
                      ("SSM", "mouse-run-run-2-ssm"), ("Transformer", "mouse-run-run-2-transformer")):
        data = _read_json_or_none(analyses_root / f"c5_{exp}.json")
        if data and data.get("summary"):
            summary = data["summary"]
            coll = summary.get("collisions_per_episode", {})
            c5[name] = {
                "unperturbed": coll.get("unperturbed", {}).get("mean"),
                "shared_removed": coll.get("shared_removed", {}).get("mean"),
                "random_removed": coll.get("random_removed", {}).get("mean"),
            }
            if name == "RNN":
                for metric in (
                    "collisions_per_episode",
                    "partner_in_vision",
                    "average_distance",
                ):
                    block = summary.get(metric)
                    if not isinstance(block, dict):
                        continue
                    c5_rnn_detail[metric] = {
                        condition: {
                            "mean": block.get(condition, {}).get("mean"),
                            "sd": block.get(condition, {}).get("std"),
                        }
                        for condition in (
                            "unperturbed",
                            "shared_removed",
                            "random_removed",
                        )
                    }
    return {
        "claims": [
            {"id": cid, "claim": claim, "verdict": verdict, "detail": detail}
            for cid, claim, verdict, detail in CLAIM_ALIGNMENT
        ],
        "c4": (c4 or {}).get("summary"),
        "c5": c5,
        "c5_rnn_detail": c5_rnn_detail,
    }


def _arch_rank(label: str) -> int:
    return ARCHITECTURE_ORDER.index(label) if label in ARCHITECTURE_ORDER else len(ARCHITECTURE_ORDER)


def _behavior_rows(evaluation: dict) -> dict[str, object]:
    rows: dict[str, object] = {}
    for task in ("social", "non_social"):
        modes = evaluation.get(task)
        if not isinstance(modes, dict):
            continue
        entry = {}
        for mode in ("random_explorer", "random_chaser"):
            metrics = modes.get(mode)
            if isinstance(metrics, dict):
                entry[mode] = {}
                for key in (
                    "collisions_per_episode",
                    "chaser_partner_vision",
                    "average_distance",
                    "chaser_new_fields",
                    "explorer_new_fields",
                ):
                    stat = metrics.get(key, {})
                    entry[mode][key] = stat.get("mean")
                    entry[mode][f"{key}_sd"] = stat.get("std")
                entry[mode]["n"] = metrics.get("collisions_per_episode", {}).get("n")
        rows[task] = entry
    return rows


def _training_rows(training: dict) -> dict[str, object]:
    rows = {}
    for task in ("social", "non_social"):
        block = training.get(task)
        if isinstance(block, dict):
            rows[task] = {
                key: block.get(key, {}).get("mean")
                for key in ("collisions_per_episode", "chaser_return", "explorer_return", "value_loss")
            }
    return rows


def _neural_rows(neural: dict | None) -> dict[str, object] | None:
    if not neural:
        return None
    by_task = neural.get("by_task", {})
    rows = {}
    for task in ("social", "non_social"):
        block = by_task.get(task)
        if not isinstance(block, dict):
            continue
        keys = {
            "plsc_top_dim_correlation": "plsc_top_dim_correlation",
            "plsc_n_significant": "plsc_n_significant",
            "decode_collision": "decode_chaser_collision",
            "decode_explorer_collision": "decode_explorer_collision",
            "decode_escape": "decode_chaser_partner_escape",
            "decode_approach": "decode_explorer_partner_approach",
            "decode_collision_shuffled": "decode_chaser_collision_shuffled",
        }
        rows[task] = {
            "n": block.get("n"),
            "n_degenerate": block.get("n_degenerate"),
        }
        for output_key, source_key in keys.items():
            stat = block.get(source_key, {})
            rows[task][output_key] = stat.get("mean")
            rows[task][f"{output_key}_sd"] = stat.get("std")
            rows[task][f"{output_key}_n"] = stat.get("n")
    return rows


def _read_json_or_none(path: Path) -> object | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


@torch.no_grad()
def compute_shared_subspace(
    checkpoint: Path,
    *,
    episodes: int = 8,
    max_steps: int | None = None,
    permutations: int = 100,
    alpha: float = 0.05,
    seed: int = 0,
    device_name: str = "auto",
    heatmap_size: int = 64,
    example_top_k: int = 6,
) -> dict[str, object]:
    """Compute the PLSC shared/unique subspace decomposition for one checkpoint.

    Collects a small self-play pool, excludes paper-style degenerate episodes,
    runs PLSC (cross-covariance SVD + temporal-permutation significance), and
    returns the spectrum, significance, cross-covariance heatmap, per-agent
    shared vs unique variance split, and one example episode's shared/unique
    norm decomposition over time.
    """
    torch.manual_seed(seed)
    device = select_device(device_name)
    pair = load_policy_pair(checkpoint, device=device, max_steps=max_steps)
    config = pair.config
    env_config = pair.env_config
    chaser = pair.chaser
    explorer = pair.explorer

    batch = collect_batch(
        env_config=env_config,
        batch_size=episodes,
        chaser=chaser,
        explorer=explorer,
        device=device,
        deterministic=False,
        opponent_mode="self_play",
        degenerate_threshold_fraction=0.01,
    )
    steps = int(batch["chaser_hidden"].shape[0])
    degenerate = batch["episode_degenerate"].bool()
    valid = ~degenerate
    if not bool(valid.any()):  # every pooled episode degenerate: keep them all.
        valid = torch.ones_like(degenerate)

    chaser_hidden = batch["chaser_hidden"][:, valid].float()  # (T, Ev, H)
    explorer_hidden = batch["explorer_hidden"][:, valid].float()
    hidden_size = chaser_hidden.shape[2]
    pooled_chaser = chaser_hidden.reshape(-1, hidden_size)
    pooled_explorer = explorer_hidden.reshape(-1, hidden_size)
    if pooled_chaser.shape[0] < 8:
        raise ValueError("not enough non-degenerate samples for PLSC")

    result = compute_plsc(
        pooled_chaser,
        pooled_explorer,
        permutations=permutations,
        alpha=alpha,
        seed=seed,
    )
    shared_dims = result.significant_dims

    zc = (pooled_chaser - result.chaser_mean) / result.chaser_std
    ze = (pooled_explorer - result.explorer_mean) / result.explorer_std
    chaser_shared = shared_variance_fraction(zc, result.chaser_basis, shared_dims)
    explorer_shared = shared_variance_fraction(ze, result.explorer_basis, shared_dims)

    # Example episode 0 from the pool, decomposed into shared vs unique norm.
    chaser_example = (chaser_hidden[:, 0] - result.chaser_mean) / result.chaser_std
    explorer_example = (explorer_hidden[:, 0] - result.explorer_mean) / result.explorer_std
    chaser_shared_norm, chaser_unique_norm = project_norms(
        chaser_example, result.chaser_basis, shared_dims
    )
    explorer_shared_norm, explorer_unique_norm = project_norms(
        explorer_example, result.explorer_basis, shared_dims
    )
    example_top = min(example_top_k, result.chaser_basis.shape[1])
    chaser_latent = chaser_example @ result.chaser_basis[:, :example_top]
    explorer_latent = explorer_example @ result.explorer_basis[:, :example_top]

    warnings: list[str] = []
    samples = int(pooled_chaser.shape[0])
    if samples < hidden_size:
        warnings.append(
            f"Only {samples} pooled samples for {hidden_size} units: the"
            " cross-covariance is rank-deficient, so nearly every dimension"
            " looks significant. Increase episodes/steps or use a"
            " longer-horizon checkpoint (the paper uses 25 x 500)."
        )
    if int(valid.sum()) < episodes:
        warnings.append(
            f"{int(degenerate.sum())} of {episodes} pooled episodes were"
            " degenerate."
        )

    top = min(20, result.singular_values.shape[0])
    return {
        "checkpoint": str(checkpoint),
        "task": (config.get("env") or {}).get("task"),
        "device": str(device),
        "warnings": warnings,
        "pool": {
            "episodes_requested": episodes,
            "episodes_used": int(valid.sum()),
            "excluded_degenerate": int(degenerate.sum()),
            "steps": steps,
            "samples": int(pooled_chaser.shape[0]),
        },
        "params": {
            "permutations": permutations,
            "alpha": alpha,
            "seed": seed,
            "hidden_size": hidden_size,
        },
        "significant_dims": shared_dims,
        "leading_significant_dims": result.leading_significant_dims,
        "unique_dims": hidden_size - shared_dims,
        "top_dim_correlation": round(float(result.latent_correlations[0]), 4),
        "singular_values": _round_list(result.singular_values[:top]),
        "null_p95": _round_list(result.null_percentile[:top]),
        "null_low": _round_list(result.null[:, :top].min(dim=0).values),
        "covariance_explained": _round_list(result.covariance_explained[:top]),
        "latent_correlations": _round_list(result.latent_correlations[:top]),
        "variance": {
            "chaser": {"shared": round(chaser_shared, 4), "unique": round(1.0 - chaser_shared, 4)},
            "explorer": {"shared": round(explorer_shared, 4), "unique": round(1.0 - explorer_shared, 4)},
        },
        "cross_covariance": _downsample_heatmap(result.cross_covariance, heatmap_size),
        "example_episode": {
            "steps": steps,
            "top_k": example_top,
            "chaser": {
                "shared_norm": _round_list(chaser_shared_norm),
                "unique_norm": _round_list(chaser_unique_norm),
                "latent": _round_matrix(chaser_latent),
            },
            "explorer": {
                "shared_norm": _round_list(explorer_shared_norm),
                "unique_norm": _round_list(explorer_unique_norm),
                "latent": _round_matrix(explorer_latent),
            },
        },
    }


def _downsample_heatmap(matrix: torch.Tensor, size: int) -> dict[str, object]:
    """Block-average a square matrix down to at most size x size for display."""
    rows = matrix.shape[0]
    target = min(size, rows)
    pooled = F.adaptive_avg_pool2d(matrix.abs().unsqueeze(0).unsqueeze(0), target).squeeze()
    return {
        "size": int(target),
        "source_size": int(rows),
        "max_abs": round(float(pooled.max()), 5),
        "values": [[round(float(x), 4) for x in row] for row in pooled.tolist()],
    }


def _round_list(tensor: torch.Tensor, digits: int = 4) -> list[float]:
    return [round(float(x), digits) for x in tensor.reshape(-1).tolist()]


def _round_matrix(tensor: torch.Tensor, digits: int = 3) -> list[list[float]]:
    return [[round(float(x), digits) for x in row] for row in tensor.tolist()]


def _clamp_int(raw: str, *, low: int, high: int) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = low
    return max(low, min(high, value))


@torch.no_grad()
def generate_trajectory(
    checkpoint: Path,
    *,
    episode_seed: int = 0,
    deterministic: bool = True,
    opponent_mode: OpponentMode = "self_play",
    device_name: str = "auto",
) -> dict[str, object]:
    torch.manual_seed(episode_seed)
    device = select_device(device_name)
    pair = load_policy_pair(checkpoint, device=device)
    config = pair.config
    checkpoint_metrics = pair.metrics
    env_config = pair.env_config
    chaser = pair.chaser
    explorer = pair.explorer

    env = BatchedChaseEnv(env_config, batch_size=1, device=device)
    chaser_observation, explorer_observation = env.reset()
    chaser_hidden = chaser.initial_hidden(1, device)
    explorer_hidden = explorer.initial_hidden(1, device)

    chaser_return = 0.0
    explorer_return = 0.0
    collisions = 0
    chaser_new_fields = 0
    explorer_new_fields = 0
    chaser_visible_steps = 0
    explorer_visible_steps = 0
    chaser_visited = {_position_key(env.chaser_position)}
    explorer_visited = {_position_key(env.explorer_position)}
    chaser_positions = [env.chaser_position.clone()]
    explorer_positions = [env.explorer_position.clone()]
    chaser_actions = []
    explorer_actions = []
    chaser_collisions = []
    explorer_collisions = []
    chaser_hiddens = []
    explorer_hiddens = []
    chaser_values = []
    explorer_values = []
    frames = [
        _frame(
            t=0,
            env=env,
            chaser_visited=chaser_visited,
            explorer_visited=explorer_visited,
            chaser_return=chaser_return,
            explorer_return=explorer_return,
        )
    ]

    for step in range(env_config.max_steps):
        previous_chaser = _position(env.chaser_position)
        previous_explorer = _position(env.explorer_position)
        chaser_output = chaser(chaser_observation, chaser_hidden)
        explorer_output = explorer(explorer_observation, explorer_hidden)
        chaser_probs = F.softmax(chaser_output.logits, dim=-1)
        explorer_probs = F.softmax(explorer_output.logits, dim=-1)
        chaser_action = _select_action(chaser_probs, deterministic)
        explorer_action = _select_action(explorer_probs, deterministic)
        if opponent_mode == "random_chaser":
            chaser_action = torch.randint(4, (1,), device=device)
        elif opponent_mode == "random_explorer":
            explorer_action = torch.randint(4, (1,), device=device)
        elif opponent_mode != "self_play":
            raise ValueError(f"Unsupported opponent mode: {opponent_mode}")

        result = env.step(chaser_action, explorer_action)
        chaser_actions.append(chaser_action.clone())
        explorer_actions.append(explorer_action.clone())
        chaser_hiddens.append(chaser_output.hidden[0].detach().cpu())
        explorer_hiddens.append(explorer_output.hidden[0].detach().cpu())
        chaser_values.append(_float(chaser_output.value))
        explorer_values.append(_float(explorer_output.value))
        chaser_positions.append(result.chaser_position.clone())
        explorer_positions.append(result.explorer_position.clone())
        chaser_collisions.append(result.chaser_collision.clone())
        explorer_collisions.append(result.explorer_collision.clone())
        chaser_return += _float(result.chaser_reward)
        explorer_return += _float(result.explorer_reward)
        chaser_movement = _movement(
            before=previous_chaser,
            after=_position(result.chaser_position),
            action=int(_long(chaser_action)),
            collision=_bool(result.collision),
            grid_size=env_config.grid_size,
        )
        explorer_movement = _movement(
            before=previous_explorer,
            after=_position(result.explorer_position),
            action=int(_long(explorer_action)),
            collision=_bool(result.collision),
            grid_size=env_config.grid_size,
        )
        collisions += int(_bool(result.collision))
        chaser_new_fields += int(_bool(result.chaser_new_field))
        explorer_new_fields += int(_bool(result.explorer_new_field))
        chaser_visible_steps += int(_bool(result.chaser_partner_visible))
        explorer_visible_steps += int(_bool(result.explorer_partner_visible))
        chaser_visited.add(_position_key(result.chaser_position))
        explorer_visited.add(_position_key(result.explorer_position))

        frames.append(
            _frame(
                t=step + 1,
                env=env,
                chaser_visited=chaser_visited,
                explorer_visited=explorer_visited,
                chaser_return=chaser_return,
                explorer_return=explorer_return,
                chaser_action=int(_long(chaser_action)),
                explorer_action=int(_long(explorer_action)),
                chaser_probabilities=_probabilities(chaser_probs),
                explorer_probabilities=_probabilities(explorer_probs),
                chaser_reward=_float(result.chaser_reward),
                explorer_reward=_float(result.explorer_reward),
                collision=_bool(result.collision),
                chaser_new_field=_bool(result.chaser_new_field),
                explorer_new_field=_bool(result.explorer_new_field),
                chaser_approach=_bool(result.chaser_approach),
                explorer_escape=_bool(result.explorer_escape),
                explorer_escape_close=_bool(result.explorer_escape_close),
                explorer_escape_near=_bool(result.explorer_escape_near),
                explorer_escape_far=_bool(result.explorer_escape_far),
                chaser_movement=chaser_movement,
                explorer_movement=explorer_movement,
                chaser_subspace_norm=_float(
                    chaser.neural_action_subspace(chaser_output.hidden).norm(dim=1)
                ),
                explorer_subspace_norm=_float(
                    explorer.neural_action_subspace(explorer_output.hidden).norm(dim=1)
                ),
            )
        )

        chaser_hidden = chaser_output.state
        explorer_hidden = explorer_output.state
        chaser_observation = result.chaser_observation
        explorer_observation = result.explorer_observation

    horizon = float(env_config.max_steps)
    degeneracy = episode_degeneracy(
        torch.stack(chaser_positions),
        torch.stack(explorer_positions),
        torch.stack(chaser_actions),
        torch.stack(explorer_actions),
        chaser_collisions=torch.stack(chaser_collisions),
        explorer_collisions=torch.stack(explorer_collisions),
        threshold_fraction=0.01,
    )
    return {
        "checkpoint": str(checkpoint),
        "config": config,
        "checkpoint_metrics": _json_finite(checkpoint_metrics),
        "episode_seed": episode_seed,
        "deterministic": deterministic,
        "opponent_mode": opponent_mode,
        "device": str(device),
        "summary": {
            "collisions": collisions,
            "chaser_return": chaser_return,
            "explorer_return": explorer_return,
            "chaser_new_fields": chaser_new_fields,
            "explorer_new_fields": explorer_new_fields,
            "chaser_partner_vision": chaser_visible_steps / horizon,
            "explorer_partner_vision": explorer_visible_steps / horizon,
            "final_distance": frames[-1]["distance"],
            "degenerate": _tensor_bool(degeneracy["degenerate"]),
            "degenerate_threshold_fraction": 0.01,
            "degenerate_threshold_steps": _tensor_float(degeneracy["threshold_steps"]),
            "same_state_run_steps": _tensor_int(degeneracy["same_state_run_steps"]),
            "chaser_stuck_run_steps": _tensor_int(degeneracy["chaser_stuck_run_steps"]),
            "explorer_stuck_run_steps": _tensor_int(degeneracy["explorer_stuck_run_steps"]),
        },
        "action_labels": ACTION_LABELS,
        "frames": frames,
        "neural": {
            "hidden_size": config["hidden_size"],
            "alignment": (
                "neural index t is computed from state s_t and produced action t;"
                " it aligns with frames[t + 1]"
            ),
            "chaser": _neural_payload(chaser_hiddens, chaser_values),
            "explorer": _neural_payload(explorer_hiddens, explorer_values),
        },
    }


def _neural_payload(
    hiddens: list[torch.Tensor],
    values: list[float],
    *,
    pca_components: int = 3,
    explained_components: int = 8,
) -> dict[str, object]:
    """Per-step network activity plus a PCA embedding of the hidden trajectory."""
    activity = torch.stack(hiddens).float()  # (steps, hidden_size)
    unit_max = activity.amax(dim=0)
    active = unit_max > 1e-8
    # Peak-time sort gives the raster its sequence structure; silent units sink
    # to the bottom in index order.
    peak_time = activity.argmax(dim=0)
    order = sorted(
        range(activity.shape[1]),
        key=lambda unit: (not bool(active[unit]), int(peak_time[unit]), unit),
    )

    centered = activity - activity.mean(dim=0, keepdim=True)
    pca: dict[str, object] = {"coords": None, "explained": None}
    if activity.shape[0] >= 2 and float(centered.abs().max()) > 0.0:
        _, singular, v_rows = torch.linalg.svd(centered, full_matrices=False)
        variance = singular.square()
        explained = variance / variance.sum().clamp_min(1e-12)
        coords = centered @ v_rows[:pca_components].T
        pca = {
            "coords": [[round(float(x), 4) for x in row] for row in coords.tolist()],
            "explained": [round(float(x), 4) for x in explained[:explained_components].tolist()],
        }

    return {
        "hidden": [[round(float(x), 3) for x in row] for row in activity.tolist()],
        "value": [round(float(x), 4) for x in values],
        "unit_order": order,
        "active_units": int(active.sum()),
        "pca": pca,
    }


def _frame(
    *,
    t: int,
    env: BatchedChaseEnv,
    chaser_visited: set[tuple[int, int]],
    explorer_visited: set[tuple[int, int]],
    chaser_return: float,
    explorer_return: float,
    chaser_action: int | None = None,
    explorer_action: int | None = None,
    chaser_probabilities: list[float] | None = None,
    explorer_probabilities: list[float] | None = None,
    chaser_reward: float = 0.0,
    explorer_reward: float = 0.0,
    collision: bool = False,
    chaser_new_field: bool = False,
    explorer_new_field: bool = False,
    chaser_approach: bool = False,
    explorer_escape: bool = False,
    explorer_escape_close: bool = False,
    explorer_escape_near: bool = False,
    explorer_escape_far: bool = False,
    chaser_movement: dict[str, object] | None = None,
    explorer_movement: dict[str, object] | None = None,
    chaser_subspace_norm: float = 0.0,
    explorer_subspace_norm: float = 0.0,
) -> dict[str, object]:
    chaser_position = _position(env.chaser_position)
    explorer_position = _position(env.explorer_position)
    chaser_visible = _bool(env._partner_in_fov(env.chaser_position, env.explorer_position))
    explorer_visible = _bool(env._partner_in_fov(env.explorer_position, env.chaser_position))
    return {
        "t": t,
        "chaser": chaser_position,
        "explorer": explorer_position,
        "chaser_visited": _sorted_positions(chaser_visited),
        "explorer_visited": _sorted_positions(explorer_visited),
        "distance": _float(env.distance()),
        "chaser_partner_visible": chaser_visible,
        "explorer_partner_visible": explorer_visible,
        "actions": {
            "chaser": chaser_action,
            "explorer": explorer_action,
            "chaser_label": None if chaser_action is None else ACTION_LABELS[chaser_action],
            "explorer_label": None if explorer_action is None else ACTION_LABELS[explorer_action],
        },
        "probabilities": {
            "chaser": chaser_probabilities,
            "explorer": explorer_probabilities,
        },
        "rewards": {
            "chaser": chaser_reward,
            "explorer": explorer_reward,
        },
        "returns": {
            "chaser": chaser_return,
            "explorer": explorer_return,
        },
        "events": {
            "collision": collision,
            "chaser_new_field": chaser_new_field,
            "explorer_new_field": explorer_new_field,
            "chaser_approach": chaser_approach,
            "explorer_escape": explorer_escape,
            "explorer_escape_close": explorer_escape_close,
            "explorer_escape_near": explorer_escape_near,
            "explorer_escape_far": explorer_escape_far,
        },
        "movement": {
            "chaser": chaser_movement or _empty_movement(),
            "explorer": explorer_movement or _empty_movement(),
        },
        "subspace_norm": {
            "chaser": chaser_subspace_norm,
            "explorer": explorer_subspace_norm,
        },
    }


def _select_action(probabilities: torch.Tensor, deterministic: bool) -> torch.Tensor:
    if deterministic:
        return probabilities.argmax(dim=-1)
    return torch.multinomial(probabilities, num_samples=1).squeeze(1)


def _movement(
    *,
    before: list[int],
    after: list[int],
    action: int,
    collision: bool,
    grid_size: int,
) -> dict[str, object]:
    row_delta, col_delta = ((-1, 0), (0, 1), (1, 0), (0, -1))[action]
    raw = [before[0] + row_delta, before[1] + col_delta]
    moved = before != after
    blocked_by_wall = (
        raw[0] < 0
        or raw[0] >= grid_size
        or raw[1] < 0
        or raw[1] >= grid_size
    )
    blocked_by_collision = (not moved) and (not blocked_by_wall) and collision
    return {
        "moved": moved,
        "blocked": not moved,
        "blocked_by_wall": blocked_by_wall,
        "blocked_by_collision": blocked_by_collision,
        "attempted": raw,
    }


def _empty_movement() -> dict[str, object]:
    return {
        "moved": False,
        "blocked": False,
        "blocked_by_wall": False,
        "blocked_by_collision": False,
        "attempted": None,
    }


def _resolve_checkpoint(raw: str, *, project_root: Path) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = project_root / path
    path = path.resolve()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(str(path))
    if path.suffix != ".safetensors":
        raise ValueError("checkpoint must be a .safetensors file")
    return path


def _display_path(path: Path, project_root: Path) -> str:
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def _loads_metadata(metadata: dict[str, str], key: str) -> dict[str, object]:
    raw = metadata.get(key)
    if not raw:
        return {}
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if isinstance(decoded, dict):
        return decoded
    return {}


def _infer_experiment(path: Path, runs_root: Path) -> str | None:
    try:
        return path.resolve().relative_to(runs_root.resolve()).parts[0]
    except (ValueError, IndexError):
        return None


def _infer_seed(path: Path) -> int | None:
    for part in path.parts:
        if part.startswith("seed_"):
            try:
                return int(part.removeprefix("seed_"))
            except ValueError:
                return None
    return None


def _infer_update(path: Path) -> int | None:
    stem = path.stem
    if stem.startswith("update_"):
        try:
            return int(stem.removeprefix("update_"))
        except ValueError:
            return None
    if stem == "latest" or stem == "final":
        return None
    return None


def _first(query: dict[str, list[str]], key: str, default: str = "") -> str:
    values = query.get(key)
    if not values:
        return default
    return values[0]


def _parse_bool(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "on"}


def _position(tensor: torch.Tensor) -> list[int]:
    return [int(tensor[0, 0].item()), int(tensor[0, 1].item())]


def _position_key(tensor: torch.Tensor) -> tuple[int, int]:
    return (int(tensor[0, 0].item()), int(tensor[0, 1].item()))


def _sorted_positions(values: set[tuple[int, int]]) -> list[list[int]]:
    return [[row, col] for row, col in sorted(values)]


def _probabilities(tensor: torch.Tensor) -> list[float]:
    return [float(value) for value in tensor[0].detach().cpu().tolist()]


def _bool(tensor: torch.Tensor) -> bool:
    return bool(tensor[0].item())


def _float(tensor: torch.Tensor) -> float:
    return float(tensor[0].item())


def _long(tensor: torch.Tensor) -> int:
    return int(tensor[0].item())


def _tensor_bool(tensor: torch.Tensor) -> bool:
    return bool(tensor.reshape(-1)[0].item())


def _tensor_int(tensor: torch.Tensor) -> int:
    return int(tensor.reshape(-1)[0].item())


def _tensor_float(tensor: torch.Tensor) -> float:
    return float(tensor.reshape(-1)[0].item())


def _json_finite(value: object) -> object:
    if isinstance(value, dict):
        return {key: _json_finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_finite(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
