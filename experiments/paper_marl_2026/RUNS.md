# Runs

## Launch Gates

```bash
uv run python scripts/verify_triton_env.py --output runs/mouse-run-run-0701/triton_equivalence.json
uv run python scripts/run/launch_gate.py --config experiments/paper_marl_2026/configs/primary.json --triton-equivalence runs/mouse-run-run-0701/triton_equivalence.json
```

## Primary Training

```bash
uv run python scripts/run_local_experiment.py --config experiments/paper_marl_2026/configs/primary.json
```

## Paper Evaluation

```bash
uv run evaluate-paper-marl runs/mouse-run-run-0701 --output runs/mouse-run-run-0701/paper_random_eval.jsonl --device cuda
uv run collect-paper-marl-rollouts runs/mouse-run-run-0701 --output-root runs/mouse-run-run-0701/paper_rollouts --device cuda
```

## Tables And Report

```bash
uv run python scripts/transform/build_tables.py runs/mouse-run-run-0701
uv run python scripts/analysis/summarize_experiment.py runs/tables/mouse-run-run-0701
```

## Hugging Face Artifact Upload

```bash
uv run upload-marl-artifacts runs/mouse-run-run-0701 --repo-id JacobLinCool/mouse-run-run-0701 --repo-type dataset --dry-run
uv run upload-marl-artifacts runs/mouse-run-run-0701 --repo-id JacobLinCool/mouse-run-run-0701 --repo-type dataset
```
