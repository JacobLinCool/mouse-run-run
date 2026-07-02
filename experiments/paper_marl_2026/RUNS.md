# Runs

## Run Log

### mouse-run-run-1 (2026-07-02) — primary 20-pair experiment

- Code: commit `685f46e` (clean tree), host: rented RTX 4070 Ti SUPER (vast.ai, no volume).
- Gates: Triton equivalence regenerated at `685f46e` (`runs/mouse-run-run-1/triton_equivalence.json`), launch gate `status: ok`, zero failed checks (`runs/mouse-run-run-1/launch_gate.json`).
- Config: `configs/primary.json` — 20,000 updates x 40 episodes x 100 steps per pair, seeds 0..9, social + non_social, preset `paper_text`, `value_clip` 10.0, fused agent rollout, Triton env step, concurrency 10, 24 h attempt timeout.
- Training: **20/20 units completed on attempt 1** — zero failed attempts, zero interrupted attempts, all checkpoints passed finite validation. Wall time ~2.5 h at ~16.5k env steps/s per pair (~165k aggregate).
- Health: value_loss bounded 4.7–8.0 throughout (value_clip active), max |param| <= 2.3, no NaN/Inf events.
- Paper evaluation (100 episodes x 100 steps vs standardized random opponent, seed 0, latest successful attempt per unit): 40/40 records ok.
  - social chaser vs random explorer: **8.98 ± 5.70 collisions/episode**, partner-in-vision 60.5%, avg distance 3.78
  - non_social chaser vs random explorer: 3.45 ± 0.32 collisions, avg distance 5.19
  - social explorer vs random chaser: **0.57 ± 0.19 collisions** (learned escape), non_social explorer: 2.87 ± 0.32
- Analysis rollouts: 20 x (25 episodes x 500 steps, self-play, seed 0); degenerate exclusions 82/500 episodes (16.4%) across 9 rollout files.
- Tables/report: `runs/tables/mouse-run-run-1`, `runs/reports/mouse-run-run-1` (summary.json, report.md, figures/).
- Artifacts: mirrored to the local machine and uploaded to Hugging Face dataset `JacobLinCool/mouse-run-run-1` (checkpoints, eval records, rollouts, tables, reports, manifests).

## Commands

### Launch Gates

```bash
uv run python scripts/verify_triton_env.py --output runs/mouse-run-run-1/triton_equivalence.json
uv run python scripts/run/launch_gate.py --config experiments/paper_marl_2026/configs/primary.json --triton-equivalence runs/mouse-run-run-1/triton_equivalence.json
```

### Primary Training

```bash
uv run python scripts/run_local_experiment.py --config experiments/paper_marl_2026/configs/primary.json
```

### Paper Evaluation

```bash
uv run evaluate-paper-marl runs/mouse-run-run-1 --output runs/mouse-run-run-1/paper_random_eval.jsonl --device cuda
uv run collect-paper-marl-rollouts runs/mouse-run-run-1 --output-root runs/mouse-run-run-1/paper_rollouts --device cuda
```

### Tables And Report

```bash
uv run python scripts/transform/build_tables.py runs/mouse-run-run-1
uv run python scripts/analysis/summarize_experiment.py runs/tables/mouse-run-run-1
```

### Hugging Face Artifact Upload

```bash
uv run upload-marl-artifacts runs/mouse-run-run-1 --repo-id JacobLinCool/mouse-run-run-1 --repo-type dataset --dry-run
uv run upload-marl-artifacts runs/mouse-run-run-1 --repo-id JacobLinCool/mouse-run-run-1 --repo-type dataset
```
