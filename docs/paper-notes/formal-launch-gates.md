# Formal Launch Gates

This checklist maps the requested 13 pre-launch items to implementation and verification artifacts.

1. Non-finite training guard
   - Implementation: `mouse_run_run/health.py`, `mouse_run_run/train.py`
   - Evidence: failed attempts are written to per-attempt `status.json` and root `raw_records.jsonl`.

2. Checkpoint completion validation
   - Implementation: `mouse_run_run/health.py`, `validate-marl-checkpoints`
   - Runner rule: an attempt is complete only if `status.json` is `completed` and `latest.safetensors` is finite.

3. Paper-style degenerate episode exclusion
   - Implementation: `mouse_run_run/degenerate.py`, `mouse_run_run/rollout.py`, `scripts/transform/build_tables.py`
   - Rule: raw rollouts are preserved; exclusions are represented in `exclusions.jsonl`.

4. Formal run manifest
   - Implementation: `scripts/run_local_experiment.py`, `mouse_run_run/provenance.py`
   - Output: `runs/{experiment}/manifest.json`

5. Preset decision
   - Source of truth: `experiments/paper_marl_2026/configs/primary.json`
   - Primary preset: `paper_text`

6. Triton env-only equivalence test
   - Implementation: `scripts/verify_triton_env.py`
   - Output: `runs/{experiment}/triton_equivalence.json`

7. Standardized random-opponent eval pipeline
   - Implementation: `evaluate-paper-marl`
   - Output: `runs/{experiment}/paper_random_eval.jsonl`

8. 25 x 500 analysis rollout pipeline
   - Implementation: `collect-paper-marl-rollouts`
   - Output: `runs/{experiment}/paper_rollouts/*.safetensors`

9. TensorBoard health dashboard
   - Implementation: `mouse_run_run/observability.py`
   - Tags include rollout, optimization, RNN subspace, runtime, and health metrics.

10. Rerun policy
    - Implementation: `scripts/run_local_experiment.py`
    - Failed attempts remain raw evidence and retry up to `max_attempts`.

11. Summary report
    - Implementation: `scripts/analysis/summarize_experiment.py`
    - Output: `runs/reports/{experiment}/report.md`

12. Hugging Face artifact upload
    - Implementation: `upload-marl-artifacts`
    - Dry-run: writes `HF_UPLOAD_MANIFEST.json` without uploading.

13. Viewer filtered/raw controls and seed comparison
    - Implementation: `mouse_run_run/viewer.py`, `mouse_run_run/viewer_static/index.html`
    - UI: Analysis selector supports raw and non-degenerate trajectories; Seeds panel compares checkpoint metrics.
