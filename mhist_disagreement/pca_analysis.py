#!/usr/bin/env python3
"""PCA view of the embeddings: which principal components carry the HP/SSA signal?

PCA (and the standardization before it) is fit on the train split only; every
plot and number is on the test split projected into those axes, so nothing
about the test set shaped them.

Outputs in <out>/<tag>/:
    pca_pairs.png       test images on PC1/PC2, PC3/PC4, ..., colored by SSA votes
    pca_per_pc.png      per-PC separability bars (label AUROC, |Spearman| with votes)
    pca_per_pc.csv      the same numbers, with 95% bootstrap intervals
    pca_cumulative.csv  logistic regression on the first k PCs (train -> test)
    pca_cluster.json    the PC1 two-cluster split: sizes and SSA rates per side

    python pca_analysis.py --annotations data/annotations.csv --embeddings musk=embeddings/musk.csv
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import mhist
from probe import parse_tagged

HERE = Path(__file__).resolve().parent
N_BOOT = 500


def per_pc_stats(Z, y, votes, seed):
    """For each PC alone: direction-free label AUROC (max(auc, 1-auc)), and
    |Spearman| with the vote count, each with a 95% bootstrap interval."""
    rng = np.random.default_rng(seed)
    n, rows = len(y), []
    for j in range(Z.shape[1]):
        z = Z[:, j]
        auc = roc_auc_score(y, z)
        sign = 1 if auc >= 0.5 else -1
        rho = spearmanr(z, votes).statistic
        boot_auc, boot_rho = [], []
        for _ in range(N_BOOT):
            idx = rng.integers(0, n, n)
            if len(np.unique(y[idx])) == 2:
                boot_auc.append(roc_auc_score(y[idx], sign * z[idx]))
            boot_rho.append(abs(spearmanr(z[idx], votes[idx]).statistic))
        rows.append({
            "pc": j + 1,
            "label_auroc": max(auc, 1 - auc),
            "label_auroc_lo": np.percentile(boot_auc, 2.5), "label_auroc_hi": np.percentile(boot_auc, 97.5),
            "ssa_higher_when": "high" if sign > 0 else "low",
            "votes_abs_spearman": abs(rho),
            "votes_abs_spearman_lo": np.percentile(boot_rho, 2.5),
            "votes_abs_spearman_hi": np.percentile(boot_rho, 97.5),
        })
    return pd.DataFrame(rows)


def cumulative(Ztr, ytr, Zte, yte, ks):
    rows = []
    for k in ks:
        clf = LogisticRegression(C=0.1, max_iter=5000).fit(Ztr[:, :k], ytr)
        rows.append({"first_k_pcs": k, "test_label_auroc": roc_auc_score(yte, clf.predict_proba(Zte[:, :k])[:, 1])})
    return pd.DataFrame(rows)


def plot_pairs(Z, votes, ev, n_pairs, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = 5
    rows = int(np.ceil(n_pairs / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.8 * cols, 3.4 * rows), squeeze=False,
                             gridspec_kw={"wspace": 0.4, "hspace": 0.35})
    order = np.argsort(np.abs(votes - 3.5))[::-1]  # draw unanimous first, contested on top
    for i, ax in enumerate(axes.flat):
        if i >= n_pairs:
            ax.axis("off")
            continue
        a, b = 2 * i, 2 * i + 1
        sc = ax.scatter(Z[order, a], Z[order, b], c=votes[order], cmap="coolwarm", vmin=0, vmax=7, s=5, alpha=0.75)
        ax.set_xlabel(f"PC{a + 1} ({ev[a]:.1%})", fontsize=9)
        ax.set_ylabel(f"PC{b + 1} ({ev[b]:.1%})", fontsize=9)
        ax.tick_params(labelsize=7)
    fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.8, label="pathologists voting SSA (of 7)")
    fig.suptitle("Test images in PCA axes fit on train (colored by SSA votes)")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_per_pc(stats, ev, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = stats["pc"].to_numpy()
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].bar(x, ev[: len(x)] * 100, color="gray")
    axes[0].set_ylabel("variance\nexplained (%)")
    for ax, key, base, label in (
        (axes[1], "label_auroc", 0.5, "label AUROC\n(this PC alone)"),
        (axes[2], "votes_abs_spearman", 0.0, "|Spearman|\nwith SSA votes"),
    ):
        v = stats[key].to_numpy()
        err = np.vstack([v - stats[f"{key}_lo"], stats[f"{key}_hi"] - v])
        ax.bar(x, v - base, bottom=base, yerr=err, capsize=2, color="#4a7bb7")
        ax.axhline(base, color="black", lw=0.8)
        ax.set_ylabel(label)
    axes[2].set_xlabel("principal component (fit on train, measured on test)")
    axes[2].set_xticks(x)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def cluster_split(Ztr, Zte, train, test):
    """PC1 often splits the images into two groups. Find the gap with a
    2-means split on train PC1 and report sizes and SSA rates per side."""
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(Ztr[:, :1])
    out = {}
    for name, Z, df in (("train", Ztr, train), ("test", Zte, test)):
        side = km.predict(Z[:, :1])
        out[name] = [{
            "pc1_center": float(km.cluster_centers_[c, 0]),
            "n": int((side == c).sum()),
            "ssa_majority_rate": float(df["y"].to_numpy()[side == c].mean()) if (side == c).any() else None,
            "mean_ssa_votes": float(df["votes_ssa"].to_numpy()[side == c].mean()) if (side == c).any() else None,
        } for c in range(2)]
    return out


def run_one(tag, ann, emb_path, out_dir: Path, n_pcs: int, seed: int):
    df = mhist.join(ann, mhist.load_embeddings(emb_path))
    train, test = df[df["split"] == "train"].reset_index(drop=True), df[df["split"] == "test"].reset_index(drop=True)
    Xtr, Xte = np.stack(train["embedding"]), np.stack(test["embedding"])
    scaler = StandardScaler().fit(Xtr)
    n_fit = min(max(n_pcs, 50), Xtr.shape[1], len(Xtr))
    pca = PCA(n_components=n_fit, random_state=0).fit(scaler.transform(Xtr))
    Ztr, Zte = pca.transform(scaler.transform(Xtr)), pca.transform(scaler.transform(Xte))
    ev = pca.explained_variance_ratio_
    yte, vte = test["y"].to_numpy(), test["votes_ssa"].to_numpy()
    print(f"\n[{tag}] PCA fit on {len(train)} train images, plotted on {len(test)} test images; "
          f"first {n_pcs} PCs explain {ev[:n_pcs].sum():.0%} of variance")

    out_dir.mkdir(parents=True, exist_ok=True)
    stats = per_pc_stats(Zte[:, :n_pcs], yte, vte, seed)
    stats.insert(1, "variance_explained", ev[:n_pcs])
    stats.to_csv(out_dir / "pca_per_pc.csv", index=False)
    ks = [k for k in (1, 2, 5, 10, 20, 50) if k <= n_fit]
    cum = cumulative(Ztr, train["y"].to_numpy(), Zte, yte, ks)
    cum.to_csv(out_dir / "pca_cumulative.csv", index=False)
    clusters = cluster_split(Ztr, Zte, train, test)
    (out_dir / "pca_cluster.json").write_text(json.dumps(clusters, indent=2))
    plot_pairs(Zte, vte, ev, n_pcs // 2, out_dir / "pca_pairs.png")
    plot_per_pc(stats, ev, out_dir / "pca_per_pc.png")

    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(stats[["pc", "variance_explained", "label_auroc", "label_auroc_lo", "label_auroc_hi",
                     "votes_abs_spearman", "ssa_higher_when"]].to_string(index=False))
        print("\nlogistic regression on the first k PCs:")
        print(cum.to_string(index=False))
    print("\nPC1 two-cluster split:", json.dumps(clusters["test"]))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--embeddings", action="append", required=True, help="tag=path; repeatable")
    parser.add_argument("--n-pcs", type=int, default=20, help="PCs to plot and score (even number)")
    parser.add_argument("--out", default=str(HERE / "results"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.n_pcs % 2:
        raise SystemExit("--n-pcs must be even (PCs are plotted in pairs)")

    ann = mhist.load_annotations(args.annotations)
    for tag, path in parse_tagged(args.embeddings, "--embeddings").items():
        run_one(tag, ann, path, Path(args.out) / tag, args.n_pcs, args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
