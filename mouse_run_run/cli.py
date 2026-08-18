"""Unified command surface for training, rollouts, analyses, and replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mouse_run_run.analyses.cka import run_linear_cka
from mouse_run_run.analyses.intervention import compare_rollouts
from mouse_run_run.analyses.plsc import (
    FixedRank,
    PLSCConfig,
    PermutationThreshold,
    fit_plsc,
    load_fitted_plsc,
    save_fitted_plsc,
)
from mouse_run_run.artifacts.rollout import load_rollout
from mouse_run_run.core.experiment import RuntimeConfig
from mouse_run_run.core.intervention import InterventionPipeline
from mouse_run_run.core.types import AgentId
from mouse_run_run.interventions import SubspaceIntervention
from mouse_run_run.loading import load_experiment, load_study
from mouse_run_run.rollouts import collect_experiment_rollout
from mouse_run_run.training.runner import train_experiment
from mouse_run_run.studies.runner import plan_study, run_study


DEFAULT_EXPERIMENT = "experiments.chase_grid.experiment:definition"


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    if args.command == "train":
        _train(args)
    elif args.command == "rollout":
        _rollout(args)
    elif args.command == "analyze":
        _analyze(args)
    elif args.command == "replay":
        _replay(args)
    elif args.command == "study":
        _study(args)
    elif args.command == "figures":
        _figures(args)
    else:  # pragma: no cover - argparse guarantees a registered command.
        raise AssertionError(args.command)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mrr")
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="train one typed Python experiment")
    train.add_argument("experiment", nargs="?", default=DEFAULT_EXPERIMENT)
    train.add_argument("--run-dir", type=Path, required=True)
    train.add_argument("--resume-from", type=Path)
    _runtime_arguments(train, default_batch_size=40)
    train.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="PATH=VALUE",
    )

    rollout = commands.add_parser("rollout", help="collect a saved rollout")
    rollout.add_argument("checkpoint", type=Path)
    rollout.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    rollout.add_argument("--output", type=Path, required=True)
    rollout.add_argument("--episodes", type=int, default=128)
    rollout.add_argument("--horizon", type=int)
    rollout.add_argument("--deterministic", action="store_true")
    _runtime_arguments(rollout, default_batch_size=32)
    rollout.add_argument("--set", dest="overrides", action="append", default=[])
    rollout.add_argument("--subspace", type=Path)
    rollout.add_argument("--agent")
    rollout.add_argument(
        "--basis",
        choices=("shared", "top_unique", "random_unique"),
        default="shared",
    )
    rollout.add_argument("--operation", choices=("remove", "keep"), default="remove")
    rollout.add_argument("--target", choices=("readout", "recurrent"))
    rollout.add_argument("--intervention-name", default="subspace")

    analyze = commands.add_parser("analyze", help="analyze a saved rollout")
    analyses = analyze.add_subparsers(dest="analysis", required=True)
    cka = analyses.add_parser("cka")
    cka.add_argument("rollout", type=Path)
    cka.add_argument("--output", type=Path, required=True)
    cka.add_argument("--site", default="hidden")
    plsc = analyses.add_parser("plsc")
    plsc.add_argument("rollout", type=Path)
    plsc.add_argument("--output", type=Path, required=True)
    plsc.add_argument("--agent-a", required=True)
    plsc.add_argument("--agent-b", required=True)
    plsc.add_argument("--site", default="hidden")
    selection = plsc.add_mutually_exclusive_group()
    selection.add_argument("--rank", type=int)
    selection.add_argument("--threshold", action="store_true")
    plsc.add_argument("--alpha", type=float, default=0.05)
    plsc.add_argument("--permutations", type=int, default=200)
    plsc.add_argument(
        "--null-model",
        choices=("time_shuffle", "episode_shuffle", "circular_shift"),
        default="episode_shuffle",
    )
    plsc.add_argument("--seed", type=int, default=0)
    plsc.add_argument("--min-shift", type=int, default=10)
    plsc.add_argument("--control-rank", type=int)
    compare = analyses.add_parser("compare")
    compare.add_argument("baseline", type=Path)
    compare.add_argument("disturbed", type=Path)
    compare.add_argument("--output", type=Path, required=True)

    replay = commands.add_parser("replay", help="view saved rollouts")
    replay.add_argument("rollout", type=Path)
    replay.add_argument("--compare", type=Path, action="append", default=[])
    replay.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    replay.add_argument("--host", default="127.0.0.1")
    replay.add_argument("--port", type=int, default=8765)

    study = commands.add_parser("study", help="plan or run a typed reproduction study")
    study_commands = study.add_subparsers(dest="study_command", required=True)
    study_plan = study_commands.add_parser("plan", help="validate and print workload only")
    study_plan.add_argument("definition")
    study_run = study_commands.add_parser("run", help="run one or more strict study stages")
    study_run.add_argument("definition")
    study_run.add_argument("--output", type=Path, required=True)
    study_run.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    study_run.add_argument("--backend", choices=("torch", "triton"), default="torch")
    study_run.add_argument("--parallelism", type=int, default=1)
    study_run.add_argument(
        "--stage",
        choices=("train", "behavior", "neural", "causal", "all"),
        default="all",
    )
    study_run.add_argument("--resume", action="store_true")

    figures = commands.add_parser("figures", help="render paper figures from study artifacts")
    figure_commands = figures.add_subparsers(dest="figure", required=True)
    sweep = figure_commands.add_parser(
        "sweep", help="measure partner representation at every training checkpoint"
    )
    sweep.add_argument("study_output", type=Path)
    sweep.add_argument("--episodes", type=int, default=10)
    sweep.add_argument("--horizon", type=int, default=500)
    sweep.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="cpu")
    sweep.add_argument("--backend", choices=("torch", "triton"), default="torch")

    variance = figure_commands.add_parser(
        "variance", help="partner variance from the stored neural rollouts"
    )
    variance.add_argument("study_output", type=Path)

    fig6 = figure_commands.add_parser("fig6", help="Zhang et al. (2025) Figure 6")
    fig6.add_argument("study_output", type=Path)
    fig6.add_argument("--output", type=Path, required=True)
    fig6.add_argument("--permutations", type=int, default=10_000)
    fig6.add_argument("--seed", type=int, default=0)
    fig6.add_argument(
        "--format", dest="formats", action="append", choices=("png", "svg", "pdf"), default=[]
    )

    fig5 = figure_commands.add_parser("fig5", help="Zhang et al. (2025) Figure 5")
    fig5.add_argument("study_output", type=Path)
    fig5.add_argument("--output", type=Path, required=True)
    fig5.add_argument("--box-checkpoint", type=int)
    fig5.add_argument("--early-checkpoint", type=int)
    fig5.add_argument("--late-checkpoint", type=int)
    fig5.add_argument("--x-style", choices=("log", "linear"), default="log")
    fig5.add_argument("--vision-radius", type=int, default=3)
    fig5.add_argument("--polar-bins", type=int, default=12)
    fig5.add_argument("--angle-bin-width", type=float, default=30.0)
    fig5.add_argument("--permutations", type=int, default=10_000)
    fig5.add_argument("--seed", type=int, default=0)
    fig5.add_argument(
        "--format",
        dest="formats",
        action="append",
        choices=("png", "svg", "pdf"),
        default=[],
    )
    return parser


def _runtime_arguments(parser: argparse.ArgumentParser, *, default_batch_size: int) -> None:
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--backend", choices=("torch", "triton"), default="torch")
    parser.add_argument("--batch-size", type=int, default=default_batch_size)
    parser.add_argument("--seed", type=int, default=0)


def _runtime(args: argparse.Namespace) -> RuntimeConfig:
    return RuntimeConfig(
        device=args.device,
        environment_backend=args.backend,
        batch_size=args.batch_size,
        seed=args.seed,
    )


def _train(args: argparse.Namespace) -> None:
    experiment = load_experiment(args.experiment, args.overrides)
    result = train_experiment(
        experiment,
        runtime=_runtime(args),
        run_dir=args.run_dir,
        resume_from=args.resume_from,
    )
    print(f"checkpoint={result.checkpoint}")


def _rollout(args: argparse.Namespace) -> None:
    experiment = load_experiment(args.experiment, args.overrides)
    pipeline = InterventionPipeline()
    if args.subspace is not None:
        if args.target is None:
            raise ValueError("--target is required with --subspace")
        fitted = load_fitted_plsc(args.subspace)
        agent_id = fitted.config.agent_a if args.agent is None else AgentId(args.agent)
        intervention = SubspaceIntervention(
            name=args.intervention_name,
            subspace=fitted.subspace(agent_id, args.basis),
            target=args.target,
            operation=args.operation,
        )
        pipeline = InterventionPipeline((intervention,))
    elif args.target is not None or args.agent is not None:
        raise ValueError("subspace intervention options require --subspace")
    artifact = collect_experiment_rollout(
        experiment,
        args.checkpoint,
        args.output,
        runtime=_runtime(args),
        episodes=args.episodes,
        horizon=args.horizon or experiment.training.horizon,
        deterministic=args.deterministic,
        interventions=pipeline,
    )
    print(f"rollout={artifact.path}")


def _analyze(args: argparse.Namespace) -> None:
    if args.analysis == "compare":
        result = compare_rollouts(
            load_rollout(args.baseline),
            load_rollout(args.disturbed),
            args.output,
        )
        print(f"analysis={result.output}")
        return
    rollout = load_rollout(args.rollout)
    if args.analysis == "cka":
        result = run_linear_cka(rollout, args.output, site=args.site)
        print(f"analysis={result.output}")
        return
    if args.rank is not None:
        selection = FixedRank(args.rank)
    else:
        selection = PermutationThreshold(
            alpha=args.alpha,
            permutations=args.permutations,
            null_model=args.null_model,
            seed=args.seed,
            min_shift=args.min_shift,
        )
    fitted = fit_plsc(
        rollout,
        PLSCConfig(
            agent_a=AgentId(args.agent_a),
            agent_b=AgentId(args.agent_b),
            site=args.site,
            selection=selection,
            control_rank=args.control_rank,
            control_seed=args.seed,
        ),
    )
    save_fitted_plsc(args.output, fitted, rollout=rollout.path)
    print(f"analysis={args.output}")


def _replay(args: argparse.Namespace) -> None:
    from mouse_run_run.replay.server import serve_replay

    experiment = load_experiment(args.experiment)
    serve_replay(
        args.rollout,
        compare=args.compare,
        experiment=experiment,
        host=args.host,
        port=args.port,
    )


def _figures(args: argparse.Namespace) -> None:
    # Imported here so plotting dependencies stay off the path of every other
    # command.
    from mouse_run_run.figures.fig5 import Fig5Config, render_fig5

    if args.figure == "sweep":
        from mouse_run_run.figures.sweep import SweepConfig, collect_partner_representation

        path = collect_partner_representation(
            args.study_output,
            config=SweepConfig(
                episodes=args.episodes,
                horizon=args.horizon,
                device=args.device,
                backend=args.backend,
            ),
        )
        print(f"table={path}")
        return
    if args.figure == "variance":
        from mouse_run_run.figures.sweep import collect_partner_variance

        print(f"table={collect_partner_variance(args.study_output)}")
        return
    if args.figure == "fig6":
        from mouse_run_run.figures.fig6 import Fig6Config, render_fig6

        result = render_fig6(
            args.study_output,
            args.output,
            config=Fig6Config(
                permutations=args.permutations,
                seed=args.seed,
                formats=tuple(args.formats) or ("png", "svg"),
            ),
        )
        for path in result.figure:
            print(f"figure={path}")
        print(f"manifest={result.manifest}")
        for note in result.notes:
            print(f"note: {note}")
        return
    if args.figure != "fig5":  # pragma: no cover - argparse guarantees the name.
        raise AssertionError(args.figure)
    config = Fig5Config(
        box_checkpoint=args.box_checkpoint,
        early_checkpoint=args.early_checkpoint,
        late_checkpoint=args.late_checkpoint,
        x_style=args.x_style,
        vision_radius=args.vision_radius,
        polar_bins=args.polar_bins,
        angle_bin_width=args.angle_bin_width,
        permutations=args.permutations,
        seed=args.seed,
        formats=tuple(args.formats) or ("png", "svg"),
    )
    result = render_fig5(args.study_output, args.output, config=config)
    for path in result.figure:
        print(f"figure={path}")
    print(f"source_data={len(result.source_data)} tables in {args.output / 'source_data'}")
    print(f"manifest={result.manifest}")
    for note in result.notes:
        print(f"note: {note}")


def _study(args: argparse.Namespace) -> None:
    definition = load_study(args.definition)
    if args.study_command == "plan":
        print(json.dumps(plan_study(definition), indent=2, sort_keys=True))
        return
    result = run_study(
        args.definition,
        definition,
        output=args.output,
        device=args.device,
        backend=args.backend,
        parallelism=args.parallelism,
        stage=args.stage,
        resume=args.resume,
    )
    print(f"study={result.output}")
    print(f"report={result.report}")


if __name__ == "__main__":
    main()
