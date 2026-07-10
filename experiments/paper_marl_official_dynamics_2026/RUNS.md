# Run Log

## CPU smoke — 2026-07-10

- Config: `configs/smoke.json`
- Experiment: `runs/mouse-run-run-official-dynamics-smoke`
- Result: `1/1` social unit completed, two updates, no failed attempts.
- Checkpoints: both `update_000001.safetensors` and `latest.safetensors`
  passed finite validation.
- Resume payload inspection: independent `chaser` and `explorer` optimizer
  states, both adaptive KL coefficients, CPU action RNG, and minibatch RNG were
  present and readable.
- This smoke reduces PPO epochs, minibatch size, sequence length, batch size,
  and episode length. It validates the mechanism and evidence chain, not the
  paper-scale numerical regime.

## Local MPS engineering diagnostic — 2026-07-10

- Purpose: locate performance bottlenecks before CUDA calibration.
- Resolved official learner settings: `40 x 100` rollout, hidden size 256,
  30 epochs, minibatch 128, sequence length 20, separate policies.
- One full update completed in `11.43 s` wall time on the local MPS host with
  fused rollout and `finite_guard=false`; the same seeded update took `14.79 s`
  with `finite_guard=true` (the planned setting), after the Ray
  collector/minibatch slicing path was fixed to the audited 33-minibatch
  behavior.
- This is a non-reportable development diagnostic: it used MPS rather than the
  planned CUDA host, disabled the safety guard, wrote only a temporary
  checkpoint, and must not be used as the full-run runtime estimate.

## CUDA calibration — 2026-07-10

- Runtime snapshot: clean synthetic commit `aaa0be1`, Python 3.14.2,
  PyTorch 2.12.1+cu130, Triton 3.7.1, and one NVIDIA RTX PRO 6000 Blackwell
  Server Edition GPU. The isolated commit captured the exact local source tree
  without modifying the user's dirty working tree.
- Verification: Ruff passed, all 26 tests passed, and the CUDA Triton
  environment matched the torch reference for social/partial,
  non-social/none, and social/full visibility across 20 seeds and 100 steps per
  seed. The launch gate bound the calibration config and equivalence record to
  the same clean source snapshot.
- Five-update calibration: all updates and checkpoints were finite and healthy.
  Per-update wall times were `30.193`, `26.224`, `26.218`, `26.202`, and
  `26.206 s`. Excluding the first warm update gives `26.212 +/- 0.009 s` per
  update (population SD). Peak device memory was `799 MiB`.
- Finite-guard optimization: reducing CUDA synchronization from every
  minibatch to one accumulated check per policy update improved steady update
  time by `6.55%`. The pre- and post-optimization final checkpoints had the
  same 67 tensor keys and were bit-identical (`max abs diff = 0`); final metrics
  were also identical.
- Multiplex calibration: the initial 1/2/4/8/10-worker sweep showed continued
  throughput scaling through 10 workers. Repeating 10 workers on the final
  code completed without failure at `1,094.97 environment steps/s`.
- Full-panel projection: one L2 panel contains 20 jobs x 20,000 updates x 4,000
  steps = `1.6 billion environment steps`. At the measured 10-worker
  throughput, one RTX PRO 6000 requires approximately `405.90 h` or
  `16.91 days`. This is a capacity projection from three-update workers, not a
  completed full training panel; long-run scheduling overhead and contention
  can change the realized duration.
- A worker at concurrency 10 is projected to need roughly `195-203 h` for
  20,000 updates. Both full-panel configs therefore use a `336 h` (14-day)
  per-attempt timeout, leaving headroom for long-run variance while retaining
  the existing 30-minute checkpoint/resume cadence.
- Retained evidence:
  `evidence/cuda_calibration_20260710.json`. After verification, all remote
  snapshots, virtual environments, checkpoints, TensorBoard files, per-worker
  artifacts, GPU telemetry traces, local smoke runs, and temporary checkpoints
  were deleted to avoid contaminating later experiments.
