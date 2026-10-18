#!/usr/bin/env python3
"""Triage curve: flag the X% most uncertain test images for a second read.

For each uncertainty score, sort test images from most to least uncertain and
flag the top X%. Reports, for X = 0..100%:
    caught      share of all contested images (2-5 SSA votes) that were flagged
    precision   share of flagged images that are contested
    acc_rest    accuracy of the HP/SSA call on the images NOT flagged, vs the
                majority label (call = --classifier's p(SSA) >= 0.5)
Random flagging catches X% of contested images on average (the diagonal).

Reads test_predictions.csv from probe.py, no retraining.

    python triage.py --model musk=results_musk/musk --model conch=results_conch/conch
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SCORES = {"ridge_conflict": "u_ridge_conflict", "logreg_soft": "u_logreg_soft", "logreg_hard": "u_logreg_hard"}
REPORT_AT = (10, 20, 30)


def curve(u, contested, correct):
    order = np.argsort(-u, kind="stable")
    n, n_cont = len(u), contested.sum()
    rows = []
    for pct in range(0, 101):
        k = int(round(n * pct / 100))
        flagged = np.zeros(n, dtype=bool)
        flagged[order[:k]] = True
        rows.append({
            "flag_pct": pct,
            "caught": contested[flagged].sum() / n_cont,
            "precision": contested[flagged].mean() if k else np.nan,
            "acc_rest": correct[~flagged].mean() if k < n else np.nan,
        })
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", action="append", required=True, help="tag=folder with test_predictions.csv")
    parser.add_argument("--classifier", default="logreg_soft", help="head whose p(SSA) makes the HP/SSA call")
    parser.add_argument("--out", default=str(HERE / "results_compare" / "triage"))
    args = parser.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = dict(m.split("=", 1) for m in args.model)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, len(models), figsize=(5.5 * len(models), 8.5), squeeze=False)
    headline, all_curves = [], []
    for c, (tag, folder) in enumerate(models.items()):
        df = pd.read_csv(Path(folder) / "test_predictions.csv")
        contested = (df["conflict"] >= 2).to_numpy()
        correct = ((df[f"p_{args.classifier}"] >= 0.5).astype(int) == df["y"]).to_numpy()
        ax_c, ax_a = axes[0, c], axes[1, c]
        for name, col in SCORES.items():
            if col not in df:
                continue
            cv = curve(df[col].to_numpy(), contested, correct)
            cv.insert(0, "score", name)
            cv.insert(0, "model", tag)
            all_curves.append(cv)
            ax_c.plot(cv["flag_pct"], cv["caught"] * 100, label=name)
            ax_a.plot(cv["flag_pct"], cv["acc_rest"] * 100, label=name)
            for pct in REPORT_AT:
                r = cv[cv["flag_pct"] == pct].iloc[0]
                headline.append({"model": tag, "score": name, "flag_pct": pct,
                                 "contested_caught_pct": 100 * r["caught"],
                                 "random_caught_pct": pct,
                                 "flagged_that_are_contested_pct": 100 * r["precision"],
                                 "acc_on_rest_pct": 100 * r["acc_rest"],
                                 "acc_all_pct": 100 * correct.mean()})
        ax_c.plot([0, 100], [0, 100], "--", color="gray", label="random")
        ax_c.set_title(f"{tag}: contested cases caught ({contested.sum()} of {len(df)} contested)")
        ax_c.set_xlabel("% of test images flagged (most uncertain first)")
        ax_c.set_ylabel("% of contested images caught")
        ax_c.legend(fontsize=8)
        ax_a.axhline(100 * correct.mean(), ls="--", color="gray", label="no flagging")
        ax_a.set_title(f"{tag}: accuracy on unflagged images\n(HP/SSA call from {args.classifier})")
        ax_a.set_xlabel("% of test images flagged")
        ax_a.set_ylabel("accuracy vs majority label (%)")
        ax_a.set_xlim(0, 95)
        ax_a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "triage_curve.png", dpi=130)
    plt.close(fig)

    pd.concat(all_curves).to_csv(out / "triage_curve.csv", index=False)
    table = pd.DataFrame(headline)
    table.to_csv(out / "triage_headline.csv", index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.1f}".format):
        print(table.to_string(index=False))
    summary = write_summary(table, args.classifier, out / "triage_summary.md")
    print("\n" + summary)
    print(f"saved {out}/triage_curve.png and {out}/triage_summary.md")
    return 0


SCORE_NAMES = {
    "ridge_conflict": "the conflict head (ridge_conflict)",
    "logreg_soft": "the soft head (logreg_soft, closeness of P(SSA) to 0.5)",
    "logreg_hard": "the hard head (logreg_hard, closeness of P(SSA) to 0.5)",
}


def sentence(r) -> str:
    ratio = r.contested_caught_pct / r.random_caught_pct
    return (f"Flagging the {r.flag_pct:.0f}% most uncertain cases catches {r.contested_caught_pct:.0f}% of "
            f"contested cases ({ratio:.1f}x random flagging, which catches {r.random_caught_pct:.0f}%); "
            f"accuracy on the rest is {r.acc_on_rest_pct:.0f}% (vs {r.acc_all_pct:.0f}% with no flagging).")


def write_summary(table: pd.DataFrame, classifier: str, path: Path) -> str:
    lines = [
        "# Triage summary",
        "",
        "How to read `triage_curve.png`:",
        "- **Top row:** x = share of test images sent for a second read, most uncertain first. "
        "y = share of all contested images (2-5 of 7 SSA votes) that ended up flagged. "
        "The dashed diagonal is random flagging. The further a curve sits above it, the better "
        "that uncertainty score finds the cases pathologists disputed.",
        f"- **Bottom row:** accuracy of the HP/SSA call ({classifier}) on the images that were *not* "
        "flagged, vs the majority label. The dashed line is accuracy with no flagging. A rising curve "
        "means flagging removes the cases the model gets wrong.",
        "- \"Contested caught\" counts disputed cases, not model errors. A score can be good at one and "
        "not the other: the conflict head finds disputed cases well but barely raises accuracy on the rest.",
        "",
    ]
    for model, group in table.groupby("model", sort=False):
        lines.append(f"## {model}")
        lines.append("")
        for score, rows in group.groupby("score", sort=False):
            lines.append(f"Uncertainty from {SCORE_NAMES.get(score, score)}:")
            lines.extend(f"- {sentence(r)}" for r in rows.itertuples())
            lines.append("")
    text = "\n".join(lines)
    path.write_text(text)
    return text


if __name__ == "__main__":
    sys.exit(main())
