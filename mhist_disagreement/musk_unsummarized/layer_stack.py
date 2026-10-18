#!/usr/bin/env python3
"""Stack per-layer soft heads with one interpretable weight per layer.

Each "base" is one layer's features with its own head:
    logreg   soft-label logistic regression on a layer vector (class token or
             patch average) from musk_tokens.npz, layers 6/12/18/24
    attn     (--with-attention) the attention head on a layer's 6x6 patch grid

The final p(SSA) is a weighted average of the bases' p(SSA), with weights >= 0
that sum to 1, so each weight reads as "how much this layer's opinion counts".

Nothing about the test set is used to build it:
    1. each base's hyperparameter is picked by 5-fold CV on train (as in probe.py;
       attention heads reuse the weight decay their own train-CV picked)
    2. each base predicts every training image from a model that didn't see it
       (5-fold out-of-fold predictions)
    3. the weights minimize the binomial NLL of those out-of-fold predictions
       against the vote fractions
    4. bases are refit on all of train, and the test set is scored once

Writes <out>/metrics.csv, test_predictions.csv, weights.csv and weights.png.

    python layer_stack.py --annotations ../data/annotations.csv
    python layer_stack.py --annotations ../data/annotations.csv --with-attention \
        --npz ../embeddings_full/musk_tokens_grids.npz --out ../results_musk_layers/stack_logreg+attn
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import KFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import mhist  # noqa: E402
import probe  # noqa: E402

C_GRID = [0.001, 0.01, 0.1, 1.0]


def convex_weights(P, target):
    """Weights w >= 0, sum 1, minimizing binomial NLL of P @ w (P: [n, bases])."""
    def loss(z):
        w = np.exp(z - z.max())
        w /= w.sum()
        return probe.binomial_nll(target, P @ w)
    res = minimize(loss, np.zeros(P.shape[1]), method="L-BFGS-B")
    w = np.exp(res.x - res.x.max())
    return w / w.sum()


def oof_and_test(make, Xtr, ptr, Xte, seed):
    """Out-of-fold predictions on train, and test predictions from a full-train fit."""
    oof = np.zeros(len(Xtr))
    for tr, va in KFold(5, shuffle=True, random_state=seed).split(Xtr):
        oof[va] = make().fit(Xtr[tr], ptr[tr]).predict(Xtr[va])
    return oof, make().fit(Xtr, ptr).predict(Xte)


def logreg_bases(data, tr, te, ptr, seed):
    for li, L in enumerate(data["layers"]):
        for ki, kind in enumerate(("cls", "mean")):
            X = data["layer_feats"][:, li, ki].astype(np.float32)
            C, _ = probe.pick_by_cv(probe.SoftLogReg, C_GRID, X[tr], ptr, seed)
            oof, test = oof_and_test(lambda: probe.SoftLogReg(C), X[tr], ptr, X[te], seed)
            name = f"L{L}_{kind}"
            print(f"  {name}: C={C}, out-of-fold NLL {probe.binomial_nll(ptr, oof):.4f}", flush=True)
            yield name, oof, test, {"C": C}


def attention_bases(data, tr, te, ptr, seed, results_dir: Path):
    import attention_head as ah
    for gi, L in enumerate(data["grid_layers"]):
        hp_file = results_dir / f"attention_head_L{L}" / "hyperparameters.json"
        wd = json.loads(hp_file.read_text())["attn_tokens"]["weight_decay"] if hp_file.exists() else 1e-3
        g = data["grids"][:, gi].astype(np.float32)
        g = ah.standardize(g[tr], g)[1]
        oof, test = oof_and_test(lambda: ah.GatedAttention(g.shape[-1], weight_decay=wd, seed=seed),
                                 g[tr], ptr, g[te], seed)
        name = f"L{L}_attn"
        print(f"  {name}: weight decay {wd}, out-of-fold NLL {probe.binomial_nll(ptr, oof):.4f}", flush=True)
        yield name, oof, test, {"weight_decay": wd}


def plot_weights(weights: pd.Series, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(max(5, 0.7 * len(weights) + 2), 3.2))
    ax.bar(weights.index, weights.values, color="#2c6fbb")
    for i, v in enumerate(weights.values):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_ylabel("weight (sums to 1)")
    ax.set_title("How much each layer's head counts in the stacked prediction")
    ax.set_ylim(0, max(0.05, weights.max() * 1.2))
    plt.xticks(rotation=30, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--npz", default=str(HERE.parent / "embeddings_full" / "musk_tokens.npz"))
    parser.add_argument("--with-attention", action="store_true", help="also stack attention heads on grids")
    parser.add_argument("--no-logreg", action="store_true", help="stack only the attention heads")
    parser.add_argument("--results-dir", default=str(HERE.parent / "results_musk_layers"),
                        help="where attention_head_L*/hyperparameters.json live")
    parser.add_argument("--out", default=str(HERE.parent / "results_musk_layers" / "stack_logreg"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    data = np.load(args.npz)
    ann = mhist.load_annotations(args.annotations).set_index("image").loc[list(data["images"])].reset_index()
    tr, te = (ann["split"] == "train").to_numpy(), (ann["split"] == "test").to_numpy()
    test = ann[te].reset_index(drop=True)
    ptr = ann.loc[tr, "p_ssa"].to_numpy()

    bases = []
    if not args.no_logreg:
        print("logistic regression bases:")
        bases += list(logreg_bases(data, tr, te, ptr, args.seed))
    if args.with_attention:
        print("attention bases:")
        bases += list(attention_bases(data, tr, te, ptr, args.seed, Path(args.results_dir)))
    names = [b[0] for b in bases]
    P_oof = np.stack([b[1] for b in bases], 1)
    P_test = np.stack([b[2] for b in bases], 1)

    w = convex_weights(P_oof, ptr)
    weights = pd.Series(w, index=names)
    print("\nlayer weights (learned on train out-of-fold predictions):")
    print(weights.round(3).to_string())

    rows, preds = [], test[["image", "votes_ssa", "conflict", "y"]].copy()
    candidates = {"stack": P_test @ w, "equal_average": P_test.mean(1),
                  **{f"base_{n}": P_test[:, i] for i, n in enumerate(names)}}
    for name, p in candidates.items():
        u = probe.uncertainty_from_p(p)
        rows.append({"embeddings": "musk_layer_stack", "method": name, **probe.point_metrics(test, p, u),
                     **(probe.bootstrap_ci(test, p, u, args.seed) if name in ("stack", "equal_average") else {})})
        preds[f"p_{name}"], preds[f"u_{name}"] = p, u
    table = pd.DataFrame(rows)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "metrics.csv", index=False)
    preds.to_csv(out / "test_predictions.csv", index=False)
    weights.rename("weight").to_csv(out / "weights.csv")
    (out / "bases.json").write_text(json.dumps({b[0]: b[3] for b in bases}, indent=2))
    plot_weights(weights, out / "weights.png")
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print("\n" + table[["method", "label_auroc", "label_acc", "binomial_nll", "disagree_spearman",
                            "disagree_auroc"]].to_string(index=False))
    print(f"\nsaved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
