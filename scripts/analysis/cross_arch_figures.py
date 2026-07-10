"""Figures for the cross-architecture study: sharing-vs-vision scatter and the
linear-CKA similarity matrix, rendered from the cross_arch_* analysis JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from _common import ANALYSES_ROOT, CROSS_ARCH_EXPERIMENTS, write_manifest

COLORS = {"RNN": "#d84f45", "MLP": "#8a8f4f", "SSM": "#c4862b", "Transformer": "#2189a8"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analyses-root", type=Path, default=ANALYSES_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Figure directory (default: <analyses-root>/figures).",
    )
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    perunit_path = args.analyses_root / "cross_arch_perunit.json"
    cka_path = args.analyses_root / "cross_arch_cka.json"
    rows = json.loads(perunit_path.read_text(encoding="utf-8"))
    out = args.output_dir or args.analyses_root / "figures"
    out.mkdir(parents=True, exist_ok=True)

    # Fig 1: residualized top_r vs vision, social units, per architecture
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    for arch in CROSS_ARCH_EXPERIMENTS:
        pts = [
            (r["vision"] * 100, r["top_r"])
            for r in rows
            if r["arch"] == arch and r["task"] == "social" and r["status"] == "ok"
        ]
        if pts:
            xs, ys = zip(*pts)
            ax.scatter(xs, ys, s=70, color=COLORS[arch], label=arch, alpha=0.8, edgecolor="white", linewidth=0.8)
    ax.axvline(40, color="#999", ls="--", lw=1)
    ax.set_xlabel("chaser partner-in-vision (%)")
    ax.set_ylabel("residualized PLSC top-dim correlation")
    ax.set_title("Interaction-driven shared representation vs behavioral coupling\n(social units, time-locked scaffold removed)")
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.2)
    ax.annotate(
        "RNN: high sharing at LOW vision\n(genuine internal coupling)",
        xy=(20, 0.7),
        xytext=(42, 0.85),
        fontsize=9,
        color=COLORS["RNN"],
        arrowprops=dict(arrowstyle="->", color=COLORS["RNN"]),
    )
    fig.tight_layout()
    fig.savefig(out / "sharing_vs_vision.png")
    plt.close(fig)

    # Fig 2: CKA matrix
    cka = json.loads(cka_path.read_text(encoding="utf-8"))
    names = list(CROSS_ARCH_EXPERIMENTS)
    n = len(names)
    M = np.array([[cka[f"{a}|{b}"] for b in names] for a in names])
    fig, ax = plt.subplots(figsize=(5.5, 4.8), dpi=150)
    im = ax.imshow(M, vmin=0.6, vmax=1.0, cmap="magma")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_yticklabels(names)
    for i in range(n):
        for j in range(n):
            ax.text(
                j,
                i,
                f"{M[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if M[i, j] < 0.85 else "black",
                fontsize=11,
            )
    ax.set_title("Cross-architecture representational similarity\n(linear CKA, common game-state probe, chaser)")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out / "cross_arch_cka.png")
    plt.close(fig)

    figures = [out / "sharing_vs_vision.png", out / "cross_arch_cka.png"]
    write_manifest(
        out / "MANIFEST.json",
        script=Path(__file__),
        inputs=[perunit_path, cka_path],
        outputs={"figures": figures},
    )
    print("figures written:", sorted(out.glob("*.png")))


if __name__ == "__main__":
    main()
