"""Controls for the soft-label claim (Kun Wang's suggestions).
Usage: python controls.py <annotations.csv> <embeddings.csv> <tag>
All logistic variants share the C chosen by CV on hard labels; all are evaluated on the full test split."""
import sys, csv, json, numpy as np, torch
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

csv.field_size_limit(sys.maxsize)
ann_path, emb_path, tag = sys.argv[1], sys.argv[2], sys.argv[3]
ann = {r[0]: (int(r[2]), r[3]) for r in list(csv.reader(open(ann_path)))[1:]}

rows = list(csv.reader(open(emb_path)))
header, rows = rows[0], rows[1:]
ni = header.index("image_path") if "image_path" in header else 0
ei = header.index("embedding")
names, X = [], []
for r in rows:
    key = r[ni].split("/")[-1]
    if key in ann:
        names.append(key); X.append(json.loads(r[ei]))
X = np.array(X, dtype=np.float32)
k = np.array([ann[n][0] for n in names]); split = np.array([ann[n][1] for n in names])
y = (k >= 4).astype(float); tr, te = split == "train", split == "test"
print(f"{len(names)} images, {X.shape[1]} dims, train {tr.sum()}, test {te.sum()}")
sc = StandardScaler().fit(X[tr]); X = sc.transform(X).astype(np.float32)

def metrics(p):
    p = np.clip(p, 1e-6, 1 - 1e-6); kt = k[te]; yt = y[te]
    label_auc = roc_auc_score(yt, p)
    m = (kt == 0) | (kt == 7) | ((kt >= 2) & (kt <= 5))
    contested = ((kt >= 2) & (kt <= 5))[m]; unc = -np.abs(p - 0.5)[m]
    dis_auc = roc_auc_score(contested, unc)
    rng = np.random.default_rng(0); boot = []
    for _ in range(500):
        i = rng.integers(0, m.sum(), m.sum())
        if contested[i].min() != contested[i].max(): boot.append(roc_auc_score(contested[i], unc[i]))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    nll = -np.mean((kt * np.log(p) + (7 - kt) * np.log(1 - p)) / 7)
    return label_auc, dis_auc, lo, hi, nll

C = LogisticRegressionCV(Cs=10, cv=5, scoring="neg_log_loss", max_iter=5000).fit(X[tr], y[tr]).C_[0]
print("C chosen on hard labels:", C)

def fit_logreg(idx, target):
    Xd = np.vstack([X[idx], X[idx]]); yd = np.r_[np.ones(len(idx)), np.zeros(len(idx))]
    w = np.r_[target, 1 - target]; keep = w > 0
    m = LogisticRegression(C=C, max_iter=5000).fit(Xd[keep], yd[keep], sample_weight=w[keep])
    return m.predict_proba(X[te])[:, 1]

def fit_mlp(idx, target, seed):
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 256), torch.nn.ReLU(),
                              torch.nn.Dropout(0.3), torch.nn.Linear(256, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    xt, yt = torch.tensor(X[idx]), torch.tensor(target, dtype=torch.float32)
    for _ in range(150):
        net.train(); opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(net(xt).squeeze(1), yt)
        loss.backward(); opt.step()
    net.eval()
    with torch.no_grad(): return torch.sigmoid(net(torch.tensor(X[te])).squeeze(1)).numpy()

tri = np.where(tr)[0]; results = []
def add(name, p): results.append((name,) + metrics(p))
add("logreg_hard", fit_logreg(tri, y[tri]))
add("logreg_soft", fit_logreg(tri, k[tri] / 7))
for e in (0.1, 0.2, 0.3): add(f"logreg_smooth_{e}", fit_logreg(tri, y[tri] * (1 - e) + e / 2))
cons = tri[(k[tri] == 0) | (k[tri] == 7)]
add("logreg_consensus", fit_logreg(cons, y[cons]))
subs = [np.random.default_rng(s).choice(tri, len(cons), replace=False) for s in range(5)]
add("logreg_random_subset", np.mean([fit_logreg(s, y[s]) for s in subs], axis=0))
add("mlp_hard", np.mean([fit_mlp(tri, y[tri], s) for s in range(3)], axis=0))
add("mlp_soft", np.mean([fit_mlp(tri, k[tri] / 7, s) for s in range(3)], axis=0))

print(f"\n{'method':24s} label_auc  dis_auc  [95% CI]        nll")
with open(f"results/controls_{tag}.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["method", "label_auroc", "disagree_auroc", "ci_lo", "ci_hi", "binomial_nll"])
    for r in results:
        print(f"{r[0]:24s} {r[1]:.3f}    {r[2]:.3f}    [{r[3]:.3f}-{r[4]:.3f}]  {r[5]:.3f}")
        w.writerow([r[0]] + [f"{v:.4f}" for v in r[1:]])
print(f"\nconsensus-only trained on {len(cons)} images; random subset matched in size (5 seeds)")
