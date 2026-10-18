#!/usr/bin/env python3
"""Fit the two heads the demo and stain_robustness.py use, on the MHIST train
split only, and save them to one file:

    soft    logreg_soft from probe.py: p(SSA), trained on the vote fraction k/7
    conflict ridge_conflict from probe.py: predicted min(k, 7-k), our best
            disagreement detector; images above `flag_threshold` (the train
            set's 80th percentile) get the "recommend second review" flag

The file also keeps the train embeddings, image names and votes, for the
demo's "most similar training cases". That's per-image MHIST data, so
models/ is gitignored: keep the file on Delta or your laptop, don't push it.

    python heads.py --annotations <annotations.csv> \
        --embeddings embeddings/musk.csv --out models/musk_heads.joblib
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import mhist
import probe

FLAG_PERCENTILE = 80
C_GRID = [0.001, 0.01, 0.1, 1.0]


def fit(train, seed: int = 0) -> dict:
    X = np.stack(train["embedding"])
    p, conflict = train["p_ssa"].to_numpy(), train["conflict"].to_numpy()
    C, _ = probe.pick_by_cv(probe.SoftLogReg, C_GRID, X, p, seed)
    soft = probe.SoftLogReg(C).fit(X, p)
    ridge = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 4, 11))).fit(X, conflict)
    return {
        "soft": soft,
        "conflict": ridge,
        "C": C,
        "flag_threshold": float(np.percentile(ridge.predict(X), FLAG_PERCENTILE)),
        "train_unit": X / np.linalg.norm(X, axis=1, keepdims=True),
        "train_image": train["image"].tolist(),
        "train_votes": train["votes_ssa"].to_numpy(),
    }


def predict(heads: dict, X) -> dict:
    X = np.atleast_2d(np.asarray(X, dtype=np.float32))
    conflict = heads["conflict"].predict(X)
    return {"p_ssa": heads["soft"].predict(X), "conflict": conflict,
            "flag": conflict >= heads["flag_threshold"]}


def neighbours(heads: dict, x, k: int = 5, same_image_sim: float = 0.9999):
    """Indices of the k most cosine-similar train images, and their similarities.
    Skips near-exact matches, so uploading a train image doesn't show its own votes."""
    x = np.asarray(x, dtype=np.float32)
    sims = heads["train_unit"] @ (x / np.linalg.norm(x))
    idx = [i for i in np.argsort(-sims) if sims[i] < same_image_sim][:k]
    return np.asarray(idx, dtype=int), sims[idx]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--embeddings", required=True, help="image_path,embedding CSV, e.g. embeddings/musk.csv")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    df = mhist.join(mhist.load_annotations(args.annotations), mhist.load_embeddings(args.embeddings))
    train, test = df[df["split"] == "train"], df[df["split"] == "test"]
    heads = fit(train.reset_index(drop=True), args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(heads, args.out)

    # Sanity check against probe.py's summary.csv (MUSK: label_auroc 0.879, disagree_auroc 0.764).
    out = predict(heads, np.stack(test["embedding"]))
    conflict = test["conflict"].to_numpy()
    mask = (conflict >= 2) | (conflict == 0)
    print(f"saved {args.out}: C={heads['C']}, flag threshold={heads['flag_threshold']:.3f}")
    print(f"test label AUROC {roc_auc_score(test['y'], out['p_ssa']):.3f}, "
          f"disagree AUROC (conflict head) {roc_auc_score(conflict[mask] >= 2, out['conflict'][mask]):.3f}, "
          f"flagged {out['flag'].mean():.0%} of test images")
    return 0


if __name__ == "__main__":
    sys.exit(main())
