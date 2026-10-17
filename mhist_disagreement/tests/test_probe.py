"""Synthetic-data checks: no MHIST data or server needed.

    cd mhist_disagreement && python -m pytest tests/
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mhist  # noqa: E402
import probe  # noqa: E402
import zero_shot  # noqa: E402

MHIST_COLUMNS = ["Image Name", "Majority Vote Label", "Number of Annotators who Selected SSA (Out of 7)", "Partition"]


def make_synthetic(tmp_path, n=600, dim=32, seed=0, split_names=("train", "test")):
    """Embeddings where p(SSA) is a logistic function of one hidden direction,
    and 7 simulated pathologists vote from that p."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, dim)).astype(np.float32)
    w = rng.normal(size=dim)
    p = 1 / (1 + np.exp(-(X @ w) / np.sqrt(dim) * 3))
    votes = rng.binomial(7, p)
    names = [f"MHIST_{i:04d}.png" for i in range(n)]
    ann = pd.DataFrame({
        MHIST_COLUMNS[0]: names,
        MHIST_COLUMNS[1]: np.where(votes >= 4, "SSA", "HP"),
        MHIST_COLUMNS[2]: votes,
        MHIST_COLUMNS[3]: np.where(np.arange(n) < int(n * 0.7), split_names[0], split_names[1]),
    })
    ann_path = tmp_path / "annotations.csv"
    ann.to_csv(ann_path, index=False)
    emb = pd.DataFrame({
        "image_path": [f"/data/mhist/images/{nm}" for nm in names],
        "embedding": [json.dumps(x.tolist()) for x in X],
    })
    emb_path = tmp_path / "emb.csv"
    emb.to_csv(emb_path, index=False)
    return ann_path, emb_path, p, votes


def test_load_annotations_detects_mhist_columns(tmp_path):
    ann_path, _, _, votes = make_synthetic(tmp_path, n=50, split_names=("train", "Validate"))
    ann = mhist.load_annotations(str(ann_path))
    assert list(ann["votes_ssa"]) == list(votes)
    assert set(ann["split"]) == {"train", "test"}
    assert (ann["conflict"] == np.minimum(votes, 7 - votes)).all()
    assert (ann["y"] == (votes >= 4)).all()


def test_missing_column_names_the_columns(tmp_path):
    pd.DataFrame({"file": ["a.png"], "label": ["HP"]}).to_csv(tmp_path / "bad.csv", index=False)
    with pytest.raises(SystemExit, match="Columns are"):
        mhist.load_annotations(str(tmp_path / "bad.csv"))


def test_zero_shot_score_conversion():
    results = [{"predicted_label": "SSA", "score": 0.6}, {"predicted_label": "HP", "score": 0.7}]
    assert np.allclose(zero_shot.p_ssa_from_results(results), [0.6, 0.3])


def test_soft_logreg_matches_binomial_target():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(400, 5))
    p = 1 / (1 + np.exp(-2 * X[:, 0]))
    pred = probe.SoftLogReg(C=10.0).fit(X, p).predict(X)
    assert np.corrcoef(pred, p)[0, 1] > 0.95


def test_end_to_end_recovers_signal(tmp_path):
    ann_path, emb_path, _, _ = make_synthetic(tmp_path)
    # A zero-shot CSV that's pure noise: should land near chance.
    rng = np.random.default_rng(2)
    zs = pd.read_csv(emb_path)[["image_path"]]
    zs["p_ssa"] = rng.uniform(0.4, 0.6, len(zs))
    zs.to_csv(tmp_path / "zs.csv", index=False)

    out = tmp_path / "results"
    assert probe.main(["--annotations", str(ann_path), "--embeddings", f"syn={emb_path}",
                       "--zero-shot", f"syn={tmp_path / 'zs.csv'}", "--out", str(out)]) == 0

    table = pd.read_csv(out / "summary.csv").set_index("method")
    expected = {"baseline_constant", "zero_shot", "logreg_hard", "svm_hard", "logreg_soft",
                "knn_soft", "mlp_soft", "ridge_conflict"}
    assert expected <= set(table.index)
    for name in ("logreg_hard", "logreg_soft", "mlp_soft"):
        assert table.loc[name, "label_auroc"] > 0.85, name
        assert table.loc[name, "disagree_spearman"] > 0.15, name
    assert table.loc["logreg_soft", "label_auroc_lo"] < table.loc["logreg_soft", "label_auroc"]
    assert abs(table.loc["zero_shot", "label_auroc"] - 0.5) < 0.15
    assert np.isnan(table.loc["ridge_conflict", "label_auroc"])
    for f in ("metrics.csv", "metrics.json", "test_predictions.csv", "votes_boxplot.png", "pca.png"):
        assert (out / "syn" / f).exists(), f
