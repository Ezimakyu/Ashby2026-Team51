"""P1 extras: per-class metrics, class_weight=balanced, NLL CIs, image quality vs error, standardized zero-shot.
Usage: python extras.py <annotations.csv> <embeddings.csv> <images_dir> <tag>
Reads results/zeroshot_*_set1_test.csv if present. Writes results/extras_<tag>.txt"""
import sys, csv, json, glob, os, numpy as np
from PIL import Image
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, confusion_matrix

csv.field_size_limit(sys.maxsize)
ann_path, emb_path, img_dir, tag = sys.argv[1:5]
ann = {r[0]: (int(r[2]), r[3]) for r in list(csv.reader(open(ann_path)))[1:]}
rows = list(csv.reader(open(emb_path))); header, rows = rows[0], rows[1:]
ni = header.index("image_path") if "image_path" in header else 0; ei = header.index("embedding")
names, X = [], []
for r in rows:
    key = r[ni].split("/")[-1]
    if key in ann: names.append(key); X.append(json.loads(r[ei]))
X = np.array(X, dtype=np.float32); names = np.array(names)
k = np.array([ann[n][0] for n in names]); split = np.array([ann[n][1] for n in names])
y = (k >= 4).astype(float); tr, te = split == "train", split == "test"
X = StandardScaler().fit(X[tr]).transform(X)
out = open(f"results/extras_{tag}.txt", "w")
def say(s=""): print(s); out.write(s + "\n")
rng0 = lambda: np.random.default_rng(0)

def boot_ci(fn, n, B=500):
    rng = rng0(); v = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        try: v.append(fn(i))
        except ValueError: pass
    return np.percentile(v, [2.5, 97.5])

def nll_vec(p, kt):
    p = np.clip(p, 1e-6, 1 - 1e-6); return -(kt * np.log(p) + (7 - kt) * np.log(1 - p)) / 7

def dis_auc(p, kt):
    m = (kt == 0) | (kt == 7) | ((kt >= 2) & (kt <= 5))
    c = ((kt >= 2) & (kt <= 5))[m]; u = -np.abs(p - 0.5)[m]
    return roc_auc_score(c, u), boot_ci(lambda i: roc_auc_score(c[i], u[i]), m.sum())

C = LogisticRegressionCV(Cs=10, cv=5, scoring="neg_log_loss", max_iter=5000).fit(X[tr], y[tr]).C_[0]
tri = np.where(tr)[0]
def fit(target, cw=None):
    Xd = np.vstack([X[tri], X[tri]]); yd = np.r_[np.ones(len(tri)), np.zeros(len(tri))]
    w = np.r_[target, 1 - target]
    if cw == "balanced":
        pos = target.mean(); w = w * np.r_[np.full(len(tri), 0.5 / pos), np.full(len(tri), 0.5 / (1 - pos))]
    keep = w > 0
    return LogisticRegression(C=C, max_iter=5000).fit(Xd[keep], yd[keep], sample_weight=w[keep]).predict_proba(X[te])[:, 1]

kt, yt = k[te], y[te]
heads = {"hard": fit(y[tri]), "soft": fit(k[tri] / 7),
         "hard_balanced": fit(y[tri], "balanced"), "soft_balanced": fit(k[tri] / 7, "balanced")}
say(f"== {tag}: test {te.sum()} images, {int(yt.sum())} SSA majority ==")
say(f"{'head':15s} labelAUC  SSA_sens  HP_spec  [TN FP / FN TP]   NLL [95% CI]          disAUC [95% CI]")
for h, p in heads.items():
    tn, fp, fn, tp = confusion_matrix(yt, p >= 0.5).ravel()
    nv = nll_vec(p, kt); nlo, nhi = boot_ci(lambda i: nv[i].mean(), len(nv))
    d, (dlo, dhi) = dis_auc(p, kt)
    say(f"{h:15s} {roc_auc_score(yt, p):.3f}     {tp/(tp+fn):.3f}     {tn/(tn+fp):.3f}    [{tn} {fp} / {fn} {tp}]   "
        f"{nv.mean():.3f} [{nlo:.3f}-{nhi:.3f}]   {d:.3f} [{dlo:.3f}-{dhi:.3f}]")

say("\n== Image quality (hypothesis 2), test images, soft head ==")
bright, tissue = [], []
for n in names[te]:
    a = np.asarray(Image.open(os.path.join(img_dir, n)).convert("RGB"), dtype=np.float32)
    bright.append(a.mean()); tissue.append((a.mean(axis=2) < 220).mean())
bright, tissue = np.array(bright), np.array(tissue)
p = heads["soft"]; err = np.abs(p - yt); conflict = np.minimum(kt, 7 - kt); unc = -np.abs(p - 0.5)
for qn, q in (("brightness", bright), ("tissue_fraction", tissue)):
    for tn_, t in (("model_error", err), ("pathologist_conflict", conflict), ("model_uncertainty", unc)):
        r, pv = spearmanr(q, t); say(f"{qn:16s} vs {tn_:21s} rho={r:+.3f}  p={pv:.3g}")
say(f"brightness mean {bright.mean():.1f} (sd {bright.std():.1f}); tissue fraction mean {tissue.mean():.2f}")

say("\n== Zero-shot, standard eval (disputed = 2-5 votes vs 0/7) ==")
say(f"{'file':40s} labelAUC [95% CI]        disAUC [95% CI]       NLL")
for f in sorted(glob.glob("results/zeroshot_*_set1_test.csv")):
    zr = list(csv.DictReader(open(f)))
    zk = np.array([int(r["votes_ssa"]) for r in zr]); zp = np.array([float(r["p_ssa"]) for r in zr]); zy = (zk >= 4).astype(int)
    la = roc_auc_score(zy, zp); llo, lhi = boot_ci(lambda i: roc_auc_score(zy[i], zp[i]), len(zy))
    d, (dlo, dhi) = dis_auc(zp, zk)
    say(f"{os.path.basename(f):40s} {la:.3f} [{llo:.3f}-{lhi:.3f}]   {d:.3f} [{dlo:.3f}-{dhi:.3f}]   {nll_vec(zp, zk).mean():.3f}")
out.close()
