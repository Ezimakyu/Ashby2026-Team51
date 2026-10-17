#!/usr/bin/env python3
"""Train small models on frozen embeddings to predict HP vs SSA and how much
the 7 MHIST pathologists disagreed. Reports numbers only; it makes no call on
whether they're good enough.

Every method yields, per test image, p(SSA) and/or an uncertainty score
(higher = more contested). For p(SSA) methods, uncertainty = 1 - |2p - 1|.

Methods:
    baseline_constant  predicts the train-set mean p(SSA) for everything (reference row)
    zero_shot          p(SSA) from zero_shot.py (only if --zero-shot is given)
    logreg_hard        logistic regression on the majority label
    svm_hard           RBF SVM on the majority label, Platt-scaled probabilities
    logreg_soft        logistic regression on the vote fraction k/7 (binomial likelihood)
    knn_soft           mean vote fraction of the k nearest training images (cosine)
    mlp_soft           1-hidden-layer MLP, BCE on the vote fraction
    ridge_conflict     ridge regression straight onto conflict = min(k, 7-k)

Hyperparameters are picked by 5-fold CV on the train split; the test split is
used once, for the reported metrics.

    python probe.py --annotations annotations.csv \
        --embeddings musk=embeddings/musk.csv \
        --zero-shot musk=embeddings/musk_zeroshot.csv
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score

import mhist

HERE = Path(__file__).resolve().parent
EPS = 1e-6
N_BOOT = 1000


# ---------- soft-label models ----------

def _soft_rows(X, p):
    """Each image becomes an SSA row weighted p and an HP row weighted 1-p, so
    logistic regression's weighted log-loss equals the binomial likelihood of
    the 7 votes (up to a constant factor)."""
    X2 = np.vstack([X, X])
    y2 = np.concatenate([np.ones(len(X)), np.zeros(len(X))])
    w2 = np.concatenate([p, 1 - p])
    return X2, y2, w2


def binomial_nll(p_true, p_hat):
    p_hat = np.clip(p_hat, EPS, 1 - EPS)
    return float(-np.mean(p_true * np.log(p_hat) + (1 - p_true) * np.log(1 - p_hat)))


class SoftLogReg:
    def __init__(self, C):
        self.model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))

    def fit(self, X, p):
        X2, y2, w2 = _soft_rows(X, p)
        self.model.fit(X2, y2, logisticregression__sample_weight=w2)
        return self

    def predict(self, X):
        return self.model.predict_proba(X)[:, 1]


class SoftMLP:
    def __init__(self, weight_decay, hidden=128, epochs=300, lr=1e-3, seed=0):
        self.weight_decay, self.hidden, self.epochs, self.lr, self.seed = weight_decay, hidden, epochs, lr, seed

    def fit(self, X, p):
        import torch
        torch.manual_seed(self.seed)
        self.scaler = StandardScaler().fit(X)
        Xt = torch.tensor(self.scaler.transform(X), dtype=torch.float32)
        pt = torch.tensor(p, dtype=torch.float32)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(X.shape[1], self.hidden), torch.nn.ReLU(), torch.nn.Dropout(0.2),
            torch.nn.Linear(self.hidden, 1),
        )
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        self.net.train()
        for _ in range(self.epochs):  # full batch: ~2k x 512 is tiny
            opt.zero_grad()
            loss = loss_fn(self.net(Xt).squeeze(1), pt)
            loss.backward()
            opt.step()
        return self

    def predict(self, X):
        import torch
        self.net.eval()
        with torch.no_grad():
            Xt = torch.tensor(self.scaler.transform(X), dtype=torch.float32)
            return torch.sigmoid(self.net(Xt).squeeze(1)).numpy()


def pick_by_cv(make, grid, X, p, seed):
    """Return the grid value with the lowest mean held-out binomial NLL."""
    folds = KFold(n_splits=5, shuffle=True, random_state=seed)
    scores = {}
    for value in grid:
        losses = []
        for tr, va in folds.split(X):
            model = make(value).fit(X[tr], p[tr])
            losses.append(binomial_nll(p[va], model.predict(X[va])))
        scores[value] = float(np.mean(losses))
    best = min(scores, key=scores.get)
    return best, scores


# ---------- fitting all methods ----------

def fit_methods(train: pd.DataFrame, test: pd.DataFrame, seed: int):
    """Return {method: (p_ssa or None, uncertainty)} on the test set, plus chosen hyperparameters."""
    Xtr, Xte = np.stack(train["embedding"]), np.stack(test["embedding"])
    ytr, ptr, ctr = train["y"].to_numpy(), train["p_ssa"].to_numpy(), train["conflict"].to_numpy()
    strat = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    out, chosen = {}, {}

    out["baseline_constant"] = np.full(len(test), ptr.mean())

    gs = GridSearchCV(make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
                      {"logisticregression__C": [0.001, 0.01, 0.1, 1.0]}, cv=strat, scoring="roc_auc")
    gs.fit(Xtr, ytr)
    out["logreg_hard"] = gs.predict_proba(Xte)[:, 1]
    chosen["logreg_hard"] = gs.best_params_

    svm = CalibratedClassifierCV(SVC(kernel="rbf"), ensemble=False, cv=5)  # Platt scaling
    gs = GridSearchCV(make_pipeline(StandardScaler(), svm),
                      {"calibratedclassifiercv__estimator__C": [0.3, 1.0, 3.0, 10.0]}, cv=strat, scoring="roc_auc")
    gs.fit(Xtr, ytr)
    out["svm_hard"] = gs.predict_proba(Xte)[:, 1]
    chosen["svm_hard"] = gs.best_params_

    C, scores = pick_by_cv(SoftLogReg, [0.001, 0.01, 0.1, 1.0], Xtr, ptr, seed)
    out["logreg_soft"] = SoftLogReg(C).fit(Xtr, ptr).predict(Xte)
    chosen["logreg_soft"] = {"C": C, "cv_nll": scores}

    gs = GridSearchCV(KNeighborsRegressor(metric="cosine", weights="distance"),
                      {"n_neighbors": [5, 10, 20, 40]}, cv=KFold(5, shuffle=True, random_state=seed),
                      scoring="neg_mean_squared_error")
    gs.fit(Xtr, ptr)
    out["knn_soft"] = gs.predict(Xte)
    chosen["knn_soft"] = gs.best_params_

    try:
        wd, scores = pick_by_cv(lambda v: SoftMLP(v, seed=seed), [1e-4, 1e-3, 1e-2], Xtr, ptr, seed)
        out["mlp_soft"] = SoftMLP(wd, seed=seed).fit(Xtr, ptr).predict(Xte)
        chosen["mlp_soft"] = {"weight_decay": wd, "cv_nll": scores}
    except ImportError:
        print("warning: torch not importable, skipping mlp_soft")

    ridge = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 4, 11)))
    ridge.fit(Xtr, ctr)
    chosen["ridge_conflict"] = {"alpha": float(ridge[-1].alpha_)}

    results = {name: (p, uncertainty_from_p(p)) for name, p in out.items()}
    results["ridge_conflict"] = (None, ridge.predict(Xte))
    return results, chosen


def uncertainty_from_p(p):
    return 1.0 - np.abs(2.0 * np.asarray(p) - 1.0)


# ---------- metrics ----------

def _safe_auc(y, s):
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else float("nan")


def _safe_spearman(a, b):
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(spearmanr(a, b).statistic)


def point_metrics(test: pd.DataFrame, p, u) -> dict:
    y, ptrue, k, conflict = (test[c].to_numpy() for c in ("y", "p_ssa", "votes_ssa", "conflict"))
    m = {}
    if p is not None:
        pred = (p >= 0.5).astype(int)
        m["label_auroc"] = _safe_auc(y, p)
        m["label_acc"] = float(np.mean(pred == y))
        m["label_balanced_acc"] = float(np.mean([np.mean(pred[y == c] == c) for c in (0, 1) if np.any(y == c)]))
        m["vote_spearman"] = _safe_spearman(p, ptrue)
        m["vote_mae"] = float(np.mean(np.abs(p * mhist.N_ANNOTATORS - k)))
        m["binomial_nll"] = binomial_nll(ptrue, p)
    m["disagree_spearman"] = _safe_spearman(u, conflict)
    contested = conflict >= 2          # 2-5 SSA votes
    unanimous = conflict == 0          # 0 or 7
    mask = contested | unanimous
    m["disagree_auroc"] = _safe_auc(contested[mask].astype(int), u[mask])
    split34 = conflict == 3            # 3 or 4 SSA votes
    mask = split34 | unanimous
    m["disagree_auroc_3v4"] = _safe_auc(split34[mask].astype(int), u[mask])
    return m


BOOT_KEYS = ("label_auroc", "label_acc", "vote_spearman", "disagree_spearman", "disagree_auroc")


def bootstrap_ci(test: pd.DataFrame, p, u, seed: int) -> dict:
    """95% percentile intervals from resampling test images with replacement."""
    rng = np.random.default_rng(seed)
    draws = {k: [] for k in BOOT_KEYS}
    n = len(test)
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        m = point_metrics(test.iloc[idx], None if p is None else p[idx], u[idx])
        for key in BOOT_KEYS:
            if key in m and not np.isnan(m[key]):
                draws[key].append(m[key])
    ci = {}
    for key, vals in draws.items():
        if len(vals) > N_BOOT // 2:
            ci[f"{key}_lo"], ci[f"{key}_hi"] = (float(v) for v in np.percentile(vals, [2.5, 97.5]))
    return ci


# ---------- plots ----------

def plot_votes(test, results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [n for n, (p, _) in results.items() if p is not None and n != "baseline_constant"]
    cols = min(3, len(names))
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.4 * rows), squeeze=False)
    k = test["votes_ssa"].to_numpy()
    for ax, name in zip(axes.flat, names):
        p = results[name][0]
        ax.boxplot([p[k == v] for v in range(8)], positions=range(8), widths=0.6, showfliers=False)
        ax.plot(range(8), np.arange(8) / 7, "--", lw=1, color="gray", label="k/7")
        ax.set_title(name)
        ax.set_xlabel("pathologists voting SSA (of 7)")
        ax.set_ylabel("predicted p(SSA)")
        ax.set_ylim(-0.02, 1.02)
    for ax in list(axes.flat)[len(names):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_pca(df, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    X = StandardScaler().fit_transform(np.stack(df["embedding"]))
    pca = PCA(n_components=2, random_state=0).fit(X)
    Z = pca.transform(X)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sc = ax.scatter(Z[:, 0], Z[:, 1], c=df["votes_ssa"], cmap="coolwarm", s=6, alpha=0.7, vmin=0, vmax=7)
    fig.colorbar(sc, ax=ax, label="pathologists voting SSA (of 7)")
    ev = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({ev[0]:.0%} var)")
    ax.set_ylabel(f"PC2 ({ev[1]:.0%} var)")
    ax.set_title("Embeddings, all images")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------- driver ----------

def parse_tagged(values, flag):
    out = {}
    for v in values or []:
        if "=" not in v:
            raise SystemExit(f"{flag} expects tag=path, e.g. musk=embeddings/musk.csv; got {v!r}")
        tag, path = v.split("=", 1)
        out[tag] = path
    return out


def run_one(tag, ann, emb_path, zs_path, out_dir: Path, seed: int) -> pd.DataFrame:
    df = mhist.join(ann, mhist.load_embeddings(emb_path))
    train, test = df[df["split"] == "train"].reset_index(drop=True), df[df["split"] == "test"].reset_index(drop=True)
    print(f"\n[{tag}] {len(train)} train / {len(test)} test, embedding dim {len(df['embedding'][0])}")

    results, chosen = fit_methods(train, test, seed)
    if zs_path:
        zs = pd.read_csv(zs_path)
        zs["image"] = zs["image_path"].map(lambda s: Path(s).name)
        zs_p = test[["image"]].merge(zs[["image", "p_ssa"]], on="image", how="left")["p_ssa"].to_numpy()
        if np.isnan(zs_p).any():
            print(f"warning: zero-shot CSV is missing {int(np.isnan(zs_p).sum())} test images; zero_shot skipped")
        else:
            results = {"baseline_constant": results.pop("baseline_constant"),
                       "zero_shot": (zs_p, uncertainty_from_p(zs_p)), **results}

    rows = []
    for name, (p, u) in results.items():
        row = {"embeddings": tag, "method": name, **point_metrics(test, p, u)}
        if name != "baseline_constant":
            row.update(bootstrap_ci(test, p, u, seed))
        rows.append(row)
    table = pd.DataFrame(rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "metrics.csv", index=False)
    counts = test["votes_ssa"].value_counts().sort_index()
    (out_dir / "metrics.json").write_text(json.dumps({
        "embeddings": tag, "embeddings_csv": emb_path, "zero_shot_csv": zs_path,
        "n_train": len(train), "n_test": len(test),
        "test_ssa_rate": float(test["y"].mean()),
        "test_votes_histogram": {int(k): int(v) for k, v in counts.items()},
        "test_conflict_histogram": {int(k): int(v) for k, v in test["conflict"].value_counts().sort_index().items()},
        "hyperparameters": chosen,
        "metrics": rows,
    }, indent=2, default=float))
    preds = test[["image", "votes_ssa", "conflict", "y"]].copy()
    for name, (p, u) in results.items():
        if p is not None:
            preds[f"p_{name}"] = p
        preds[f"u_{name}"] = u
    preds.to_csv(out_dir / "test_predictions.csv", index=False)
    plot_votes(test, results, out_dir / "votes_boxplot.png")
    plot_pca(df, out_dir / "pca.png")
    return table


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotations", required=True, help="MHIST annotations.csv")
    parser.add_argument("--embeddings", action="append", required=True,
                        help="tag=path to an image_path,embedding CSV; repeat for several models")
    parser.add_argument("--zero-shot", action="append", help="tag=path to zero_shot.py output for that tag")
    parser.add_argument("--out", default=str(HERE / "results"))
    parser.add_argument("--seed", type=int, default=0)
    for key in ("image", "votes", "majority", "split"):
        parser.add_argument(f"--{key}-col", help=f"annotations column for {key} (auto-detected by default)")
    args = parser.parse_args(argv)

    ann = mhist.load_annotations(args.annotations, {k: getattr(args, f"{k}_col") for k in
                                                    ("image", "votes", "majority", "split")})
    embeddings = parse_tagged(args.embeddings, "--embeddings")
    zero_shot = parse_tagged(args.zero_shot, "--zero-shot")
    for tag in zero_shot:
        if tag not in embeddings:
            raise SystemExit(f"--zero-shot tag {tag!r} has no matching --embeddings tag")

    tables = [run_one(tag, ann, path, zero_shot.get(tag), Path(args.out) / tag, args.seed)
              for tag, path in embeddings.items()]
    summary = pd.concat(tables, ignore_index=True)
    summary.to_csv(Path(args.out) / "summary.csv", index=False)

    show = ["embeddings", "method", "label_auroc", "label_acc", "label_balanced_acc",
            "vote_spearman", "disagree_spearman", "disagree_auroc", "disagree_auroc_3v4"]
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.float_format", "{:.3f}".format):
        print("\n" + summary[[c for c in show if c in summary]].to_string(index=False))
    print(f"\nFull table with 95% bootstrap intervals: {Path(args.out) / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
