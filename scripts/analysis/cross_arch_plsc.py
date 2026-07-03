"""Fast cross-architecture PLSC on time-mean-subtracted hidden states."""
import json, statistics as st
from pathlib import Path
import numpy as np
from mouse_run_run.analysis import load_rollout, episode_hidden_pairs, plsc_shared_dimensions

EXPS = [("RNN","mouse-run-run-1"), ("MLP","mouse-run-run-2-mlp"),
        ("SSM","mouse-run-run-2-ssm"), ("Transformer","mouse-run-run-2-transformer")]
out = {}
for name, exp in EXPS:
    root = Path(f"runs/{exp}/paper_rollouts")
    if not root.exists(): continue
    per_task = {"social":[], "non_social":[]}
    for f in sorted(root.glob("*.safetensors")):
        meta, t = load_rollout(f)
        cfg = meta.get("checkpoint_config",{}); task = (cfg.get("env") or {}).get("task")
        ch, ex = episode_hidden_pairs(t)
        if len(ch) < 3:
            continue
        # equal-length episodes required; paper rollouts are all 500 steps
        r = plsc_shared_dimensions(ch, ex, permutations=100, seed=0, subtract_time_mean=True)
        per_task[task].append((r.top_dim_correlation, r.n_significant, r.n_episodes))
    out[name] = per_task
    for task in ("social","non_social"):
        vals = per_task[task]
        if vals:
            tr = [v[0] for v in vals]; ns = [v[1] for v in vals]
            print(f"{name:12} {task:11} residualized top_r={st.fmean(tr):.3f}±{st.pstdev(tr):.3f}  n_sig={st.fmean(ns):.0f}  units={len(vals)}", flush=True)
json.dump(out, open("runs/analyses/cross_arch_plsc_residualized.json","w"), indent=1)
print("wrote runs/analyses/cross_arch_plsc_residualized.json")
