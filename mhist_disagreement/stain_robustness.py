#!/usr/bin/env python3
"""Stain robustness: does the HP/SSA call flip under stain shift or blur, and
do contested images flip more?

For each sampled test image, makes 3 stain-shifted copies (perturb.stain_shift,
--strength, seeds 0-2) and 1 blurred copy (--blur px), embeds all 5 versions
through the hosted server's /embed, and applies the saved soft head from
heads.py. A flip = the call (p(SSA) >= 0.5) differs from the original's.

Reports, per agreement group (7/7, 6/7, 5/7, 4/7 = size of the majority):
    flip_any    share of images where any perturbed copy flipped (Wilson 95% CI)
    flip_rate   share of (image, perturbation) pairs that flipped (bootstrap CI)
    flip_stain / flip_blur   the same, per perturbation type
    mean_dp     mean |p(SSA) change|; doesn't depend on the 0.5 threshold, but the
                sigmoid is steepest at 0.5, so it's still larger near the boundary
    mean_dlogit mean |logit p(SSA) change|: how far the image moves along the
                head's direction, independent of where it starts
    median_margin   median |p_orig - 0.5|: contested images sit nearer the
                threshold, so they can flip from small shifts; if mean_dlogit
                barely rises with conflict, that, not instability, explains the flips
plus a Spearman trend test of conflict vs flips per image.

Rerun the same command to resume (predictions are appended per batch). Takes a
few minutes for 300 images on the shared server; the servers stop Sun 6 pm.

    python stain_robustness.py --images <MHIST images dir> --annotations <annotations.csv> \
        --heads models/musk_heads.joblib --n 300 --out results_stain/musk
"""

import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
from PIL import Image
from scipy.stats import spearmanr

import heads as heads_mod
import hosted
import mhist
from perturb import blur, stain_shift

N_BOOT = 1000
GROUPS = {0: "7/7", 1: "6/7", 2: "5/7", 3: "4/7"}  # conflict -> majority size


def variants(img, strength, seeds, blur_radius):
    out = {"original": img}
    for s in seeds:
        out[f"stain_s{s}"] = stain_shift(img, strength, seed=s)
    out["blur"] = blur(img, blur_radius)
    return out


def post_with_retry(args, route, payload, tries=3):
    for attempt in range(tries):
        try:
            return hosted.post(args, route, payload)
        except requests.RequestException as e:
            if attempt == tries - 1:
                raise
            print(f"  server error ({e}); retrying in 10 s", flush=True)
            time.sleep(10)


def sample_test(test, n, seed):
    """Up to n test images, stratified by conflict so every agreement group keeps its share."""
    if n <= 0 or n >= len(test):
        return test
    frac = n / len(test)
    return (test.groupby("conflict", group_keys=False)
                .apply(lambda g: g.sample(max(1, round(len(g) * frac)), random_state=seed)))


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def boot_ci(values, seed):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = [values[rng.integers(0, len(values), len(values))].mean() for _ in range(N_BOOT)]
    return tuple(np.percentile(means, [2.5, 97.5]))


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def per_image(preds: pd.DataFrame) -> pd.DataFrame:
    """One row per image: original p, flips and |dp| per perturbation."""
    wide = preds.pivot(index="image", columns="variant", values="p_ssa")
    meta = preds.drop_duplicates("image").set_index("image")[["votes_ssa", "conflict"]]
    p0 = wide["original"]
    perturbed = [c for c in wide.columns if c != "original"]
    flips = pd.DataFrame({v: (wide[v] >= 0.5) != (p0 >= 0.5) for v in perturbed})
    dp = pd.DataFrame({v: (wide[v] - p0).abs() for v in perturbed})
    dlogit = pd.DataFrame({v: (_logit(wide[v]) - _logit(p0)).abs() for v in perturbed})
    out = meta.join(p0.rename("p_orig"))
    out["n_flips"] = flips.sum(axis=1)
    out["flip_any"] = flips.any(axis=1)
    out["flip_stain"] = flips[[c for c in perturbed if c.startswith("stain")]].mean(axis=1)
    out["flip_blur"] = flips["blur"].astype(float)
    out["flip_rate"] = flips.mean(axis=1)
    out["mean_dp"] = dp.mean(axis=1)
    out["mean_dlogit"] = dlogit.mean(axis=1)
    return out.reset_index()


def summarise(img: pd.DataFrame, seed: int) -> pd.DataFrame:
    rows = []
    groups = [(GROUPS[c], img[img["conflict"] == c]) for c in sorted(GROUPS)] + [("all", img)]
    for name, g in groups:
        n, k = len(g), int(g["flip_any"].sum())
        row = {"agreement": name, "n_images": n, "flip_any": k / n if n else np.nan}
        row["flip_any_lo"], row["flip_any_hi"] = wilson(k, n)
        for col in ("flip_rate", "flip_stain", "flip_blur", "mean_dp", "mean_dlogit"):
            row[col] = g[col].mean() if n else np.nan
            row[f"{col}_lo"], row[f"{col}_hi"] = boot_ci(g[col], seed)
        row["median_margin"] = (g["p_orig"] - 0.5).abs().median() if n else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def md_table(df) -> str:
    fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def write_report(summary, img, out: Path, args):
    rho_f, p_f = spearmanr(img["conflict"], img["n_flips"])
    rho_d, p_d = spearmanr(img["conflict"], img["mean_dp"])
    rho_l, p_l = spearmanr(img["conflict"], img["mean_dlogit"])
    show = ["agreement", "n_images", "flip_any", "flip_rate", "flip_stain", "flip_blur", "mean_dp", "mean_dlogit",
            "median_margin"]
    lines = [
        "# Stain robustness (MUSK + soft head)",
        "",
        f"{len(img)} test images; perturbations: 3 stain shifts (strength {args.strength}, "
        f"seeds {', '.join(map(str, args.stain_seeds))}) and 1 blur (radius {args.blur} px).",
        "A flip = the HP/SSA call (p(SSA) >= 0.5) differs from the unperturbed image's.",
        "",
        md_table(summary[show]),
        "",
        f"Trend, conflict (0 = 7/7 … 3 = 4/7) vs number of flips per image: Spearman {rho_f:.3f} (p = {p_f:.2g}).",
        f"Trend, conflict vs mean |change in p(SSA)|: Spearman {rho_d:.3f} (p = {p_d:.2g}).",
        f"Trend, conflict vs mean |change in logit p(SSA)|: Spearman {rho_l:.3f} (p = {p_l:.2g}).",
        "",
        "Caveat: contested images have p(SSA) nearer 0.5 (see median_margin), so the same",
        "shift is more likely to cross the threshold, and mean_dp is inflated there too because",
        "the sigmoid is steepest at 0.5. mean_dlogit removes both effects: how much it rises",
        "with conflict is how much more the contested images' embeddings actually move.",
        "95% intervals are in summary.csv (Wilson for flip_any, bootstrap over images otherwise).",
    ]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def plot(summary, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = summary[summary["agreement"] != "all"].reset_index(drop=True)
    x = np.arange(len(s))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for off, col, label in ((-0.2, "flip_stain", "stain shift"), (0.2, "flip_blur", "blur")):
        err = np.array([s[col] - s[f"{col}_lo"], s[f"{col}_hi"] - s[col]])
        axes[0].bar(x + off, s[col], 0.4, yerr=err, capsize=3, label=label)
    axes[0].set_ylabel("share of calls that flipped")
    axes[0].legend()
    err = np.array([s["mean_dp"] - s["mean_dp_lo"], s["mean_dp_hi"] - s["mean_dp"]])
    axes[1].bar(x, s["mean_dp"], 0.6, yerr=err, capsize=3, color="gray")
    axes[1].set_ylabel("mean |change in p(SSA)|")
    for ax in axes:
        ax.set_xticks(x, [f"{a}\n(n={n})" for a, n in zip(s["agreement"], s["n_images"])])
        ax.set_xlabel("pathologists agreeing with the majority")
    fig.tight_layout()
    fig.savefig(out / "flip_by_agreement.png", dpi=130)
    plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    hosted.add_server_args(parser)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--heads", required=True, help="file from heads.py, built from the same model's embeddings")
    parser.add_argument("--n", type=int, default=300, help="test images to use, stratified (0 = all)")
    parser.add_argument("--strength", type=float, default=0.2)
    parser.add_argument("--stain-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--blur", type=float, default=1.5)
    parser.add_argument("--out", default="results_stain/musk")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    hosted.check_server_args(args)

    heads = joblib.load(args.heads)
    ann = mhist.load_annotations(args.annotations)
    paths = {Path(p).name: p for p in hosted.find_images(args.images)}
    test = ann[(ann["split"] == "test") & ann["image"].isin(paths)].reset_index(drop=True)
    test = sample_test(test, args.n, args.seed).reset_index(drop=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pred_path = out / "stain_predictions.csv"
    done = set(pd.read_csv(pred_path)["image"]) if pred_path.exists() else set()
    todo = test[~test["image"].isin(done)]
    n_var = 2 + len(args.stain_seeds)
    print(f"{len(test)} test images ({len(done)} done), {n_var} versions each")

    per_batch = max(1, args.batch_size // n_var)
    for start in range(0, len(todo), per_batch):
        batch = todo.iloc[start:start + per_batch]
        names, kinds, images = [], [], []
        for _, row in batch.iterrows():
            img = Image.open(paths[row["image"]]).convert("RGB")
            for kind, im in variants(img, args.strength, args.stain_seeds, args.blur).items():
                names.append(row["image"])
                kinds.append(kind)
                images.append(hosted.encode_pil(im))
        emb = post_with_retry(args, "embed", {"images": images})["embeddings"]
        res = heads_mod.predict(heads, np.asarray(emb, dtype=np.float32))
        rows = pd.DataFrame({"image": names, "variant": kinds, "p_ssa": res["p_ssa"]})
        rows = rows.merge(batch[["image", "votes_ssa", "conflict"]], on="image")
        rows.to_csv(pred_path, mode="a", header=not pred_path.exists(), index=False)
        print(f"  {min(start + per_batch, len(todo))}/{len(todo)}", flush=True)

    preds = pd.read_csv(pred_path)
    preds = preds[preds["image"].isin(test["image"])].drop_duplicates(["image", "variant"], keep="last")
    img = per_image(preds)
    summary = summarise(img, args.seed)
    summary.to_csv(out / "summary.csv", index=False)
    write_report(summary, img, out, args)
    plot(summary, out)
    print(f"\nWrote {out}/summary.md, summary.csv, flip_by_agreement.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
