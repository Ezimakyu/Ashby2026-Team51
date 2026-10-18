#!/usr/bin/env python3
"""3x3 agreement heatmaps: true vs predicted agreement group, per model and head.

Groups by number of SSA votes (of 7). The groups are deliberately unequal:
    HP-confident   0-1 SSA votes  (7-0 or 6-1 for HP)
    contested      2-5 SSA votes  (5-2 HP through 5-2 SSA)
    SSA-confident  6-7 SSA votes  (6-1 or 7-0 for SSA)
Predicted group comes from round(7 * p(SSA)). Rows are the true group; color
is the fraction of the row, cells show count and row %.

Reads test_predictions.csv written by probe.py, no retraining.

    python agreement_bins.py --model musk=results_musk/musk --model conch=results_conch/conch
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
GROUPS = ["HP-confident\n(0-1 SSA)", "contested\n(2-5 SSA)", "SSA-confident\n(6-7 SSA)"]


def to_group(votes):
    votes = np.asarray(votes)
    return np.where(votes <= 1, 0, np.where(votes >= 6, 2, 1))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", action="append", required=True,
                        help="tag=folder containing test_predictions.csv; repeatable")
    parser.add_argument("--heads", default="logreg_hard,logreg_soft")
    parser.add_argument("--out", default=str(HERE / "results_compare" / "agreement_bins.png"))
    args = parser.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = dict(m.split("=", 1) for m in args.model)
    heads = args.heads.split(",")
    fig, axes = plt.subplots(len(models), len(heads), figsize=(5.2 * len(heads), 4.8 * len(models)), squeeze=False)
    rows = []
    for r, (tag, folder) in enumerate(models.items()):
        df = pd.read_csv(Path(folder) / "test_predictions.csv")
        true = to_group(df["votes_ssa"])
        for c, head in enumerate(heads):
            pred = to_group(np.clip(np.rint(7 * df[f"p_{head}"].to_numpy()), 0, 7))
            m = np.zeros((3, 3), dtype=int)
            np.add.at(m, (true, pred), 1)
            frac = m / m.sum(axis=1, keepdims=True)
            ax = axes[r, c]
            ax.imshow(frac, cmap="Blues", vmin=0, vmax=1, origin="lower")
            for i in range(3):
                for j in range(3):
                    ax.text(j, i, f"{m[i, j]}\n{frac[i, j]:.0%}", ha="center", va="center", fontsize=10,
                            color="white" if frac[i, j] > 0.5 else "black")
            ax.set_xticks(range(3), GROUPS, fontsize=8)
            ax.set_yticks(range(3), GROUPS, fontsize=8)
            ax.set_xlabel("predicted group  (round(7p))")
            ax.set_ylabel("true group  (pathologist votes)")
            ax.set_title(f"{tag} · {head}  (correct group {np.trace(m) / m.sum():.0%})", fontsize=10)
            for i in range(3):
                rows.append({"model": tag, "head": head, "true_group": GROUPS[i].split("\n")[0], "n": int(m[i].sum()),
                             **{f"pred_{GROUPS[j].split(chr(10))[0]}": float(frac[i, j]) for j in range(3)}})
    fig.suptitle("Test set: true vs predicted agreement group (color = fraction of row)")
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    table = pd.DataFrame(rows)
    table.to_csv(out.with_suffix(".csv"), index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.2f}".format):
        print(table.to_string(index=False))
    print(f"\nsaved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
