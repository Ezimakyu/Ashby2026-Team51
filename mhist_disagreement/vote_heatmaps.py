#!/usr/bin/env python3
"""Confusion heatmaps on the test set: true vs predicted number of pathologists.

Reads results/<tag>/test_predictions.csv from probe.py (no retraining).
    - SSA votes, 8x8 (0..7): for methods that output p(SSA), predicted votes =
      round(7 * p).
    - Conflict, 4x4 (0..3, where conflict = min(votes, 7 - votes)): ridge_conflict's
      prediction rounded, and the conflict implied by logreg_soft's predicted votes.
Rows are the true value. Each cell shows its count; its color is the fraction
of that row, so every row reads as "of the images with this many true votes,
where did the predictions land?"

Also writes vote_heatmap_summary.csv: exact-match rate and within-1 rate per
method, with the constant baseline for reference.

    python vote_heatmaps.py --tag musk
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
VOTE_METHODS = ["logreg_soft", "mlp_soft", "logreg_hard", "knn_soft"]


def confusion(true, pred, n):
    m = np.zeros((n, n), dtype=int)
    for t, p in zip(true, pred):
        m[t, p] += 1
    return m


def draw(ax, m, title, xlabel, ylabel):
    frac = m / np.maximum(m.sum(axis=1, keepdims=True), 1)
    ax.imshow(frac, cmap="Blues", vmin=0, vmax=1, origin="lower")
    n = len(m)
    for i in range(n):
        for j in range(n):
            if m[i, j]:
                ax.text(j, i, str(m[i, j]), ha="center", va="center", fontsize=8,
                        color="white" if frac[i, j] > 0.5 else "black")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)


def rates(true, pred):
    diff = np.abs(np.asarray(true) - np.asarray(pred))
    return float(np.mean(diff == 0)), float(np.mean(diff <= 1))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", default="musk")
    parser.add_argument("--results", default=str(HERE / "results"))
    args = parser.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(args.results) / args.tag
    df = pd.read_csv(out / "test_predictions.csv")
    k, c = df["votes_ssa"].to_numpy(), df["conflict"].to_numpy()
    rows = []

    methods = [m for m in VOTE_METHODS if f"p_{m}" in df]
    fig, axes = plt.subplots(1, len(methods), figsize=(4.6 * len(methods), 4.4), squeeze=False)
    for ax, m in zip(axes.flat, methods):
        pv = np.clip(np.rint(7 * df[f"p_{m}"].to_numpy()), 0, 7).astype(int)
        exact, within1 = rates(k, pv)
        rows.append({"target": "ssa_votes (0-7)", "method": m, "exact": exact, "within_1": within1})
        draw(ax, confusion(k, pv, 8), f"{m}\nexact {exact:.0%}, within 1 vote {within1:.0%}",
             "predicted SSA votes = round(7p)", "true SSA votes")
    fig.suptitle(f"[{args.tag}] test set: true vs predicted SSA votes (color = fraction of row)")
    fig.tight_layout()
    fig.savefig(out / "vote_heatmap.png", dpi=130)
    plt.close(fig)

    panels = []
    if "u_ridge_conflict" in df:
        panels.append(("ridge_conflict", np.clip(np.rint(df["u_ridge_conflict"].to_numpy()), 0, 3).astype(int)))
    if "p_logreg_soft" in df:
        pv = np.clip(np.rint(7 * df["p_logreg_soft"].to_numpy()), 0, 7).astype(int)
        panels.append(("logreg_soft (implied)", np.minimum(pv, 7 - pv)))
    fig, axes = plt.subplots(1, len(panels), figsize=(4.4 * len(panels), 4.2), squeeze=False)
    for ax, (name, pc) in zip(axes.flat, panels):
        exact, within1 = rates(c, pc)
        rows.append({"target": "conflict (0-3)", "method": name, "exact": exact, "within_1": within1})
        draw(ax, confusion(c, pc, 4), f"{name}\nexact {exact:.0%}, within 1 {within1:.0%}",
             "predicted conflict", "true conflict  (0 = 7-0 unanimous, 3 = 4-3 split)")
    fig.tight_layout()
    fig.savefig(out / "conflict_heatmap.png", dpi=130)
    plt.close(fig)

    # Reference: always predict the single most common training value.
    if "p_baseline_constant" in df:
        pv = int(np.clip(np.rint(7 * df["p_baseline_constant"].iloc[0]), 0, 7))
        rows.append({"target": "ssa_votes (0-7)", "method": f"baseline (always {pv})", **dict(zip(("exact", "within_1"), rates(k, np.full_like(k, pv))))})
    for const in range(4):
        e, w = rates(c, np.full_like(c, const))
        rows.append({"target": "conflict (0-3)", "method": f"baseline (always {const})", "exact": e, "within_1": w})

    summary = pd.DataFrame(rows)
    summary.to_csv(out / "vote_heatmap_summary.csv", index=False)
    with pd.option_context("display.float_format", "{:.3f}".format):
        print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
