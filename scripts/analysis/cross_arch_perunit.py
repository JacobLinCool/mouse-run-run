"""Per-unit residualized PLSC top_r vs behavioral coupling (vision), all archs."""
import json, statistics as st
from pathlib import Path
import numpy as np
from mouse_run_run.analysis import load_rollout, episode_hidden_pairs, plsc_shared_dimensions

EXPS = [("RNN","mouse-run-run-1"), ("MLP","mouse-run-run-2-mlp"),
        ("SSM","mouse-run-run-2-ssm"), ("Transformer","mouse-run-run-2-transformer")]
rows = []
for name, exp in EXPS:
    root = Path(f"runs/{exp}/paper_rollouts")
    for f in sorted(root.glob("*.safetensors")):
        meta, t = load_rollout(f)
        cfg = meta.get("checkpoint_config",{}); env = cfg.get("env") or {}
        task = env.get("task"); seed = cfg.get("seed")
        ch, ex = episode_hidden_pairs(t)
        vis = t["chaser_partner_visible"].float().mean().item()
        coll = t["collision"].float().sum(0).mean().item()
        if len(ch) < 3:
            rows.append(dict(arch=name, task=task, seed=seed, status="degenerate",
                             n_valid=len(ch), vision=vis, collisions=coll, top_r=None))
            continue
        r = plsc_shared_dimensions(ch, ex, permutations=60, seed=0, subtract_time_mean=True)
        rows.append(dict(arch=name, task=task, seed=seed, status="ok", n_valid=len(ch),
                         vision=vis, collisions=coll, top_r=r.top_dim_correlation))
    print(f"{name} done", flush=True)
json.dump(rows, open("runs/analyses/cross_arch_perunit.json","w"), indent=1)

print("\n=== SOCIAL units: residualized top_r vs vision (sorted by vision) ===")
for name,_ in EXPS:
    soc = [r for r in rows if r["arch"]==name and r["task"]=="social" and r["status"]=="ok"]
    soc.sort(key=lambda r: r["vision"])
    lo = [r for r in soc if r["vision"]<0.4]
    hi = [r for r in soc if r["vision"]>=0.4]
    def mm(g): return f"{st.fmean([r['top_r'] for r in g]):.3f} (n={len(g)})" if g else "—"
    ndeg = len([r for r in rows if r["arch"]==name and r["task"]=="social" and r["status"]=="degenerate"])
    print(f"{name:12} low-vision(<40%): {mm(lo):18} high-vision(>=40%): {mm(hi):18} degenerate={ndeg}")
