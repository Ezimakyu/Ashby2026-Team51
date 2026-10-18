#!/usr/bin/env python3
"""Attention head on MUSK's pooled patch grid (frozen backbone, CPU training).

Each tile is 36 patch vectors (a 6x6 grid from musk_tokens.npz). Instead of
MUSK's fixed summary, a small gated-attention pooling (Ilse et al., 2018)
learns a weight per grid cell, takes the weighted average, and predicts
p(SSA). It is trained on the vote fraction k/7, like logreg_soft, so its
uncertainty (closeness of p to 0.5) is comparable with the other heads.

    tokens       the 36 grid vectors only
    tokens+cls   the 36 grid vectors plus MUSK's last-layer class token as a 37th
                 "token", so attention can fall back on the usual summary

Weight decay is picked by 5-fold CV on train (held-out binomial NLL), with early
stopping on a slice of each training fold. The final model averages 3 seeds.
Writes metrics.csv / test_predictions.csv in probe.py's format (so triage.py,
vote_heatmaps.py and agreement_bins.py work), plus per-image attention maps to
embeddings_full/ (gitignored: they are per-image).

    python attention_head.py --annotations ../data/annotations.csv --out ../results_musk_layers
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import mhist  # noqa: E402
import probe  # noqa: E402


class GatedAttention:
    """Projection -> gated attention pooling over tokens -> logit."""

    def __init__(self, dim_in, hidden=128, attn=64, dropout=0.25, weight_decay=1e-3, lr=1e-3,
                 max_epochs=200, patience=20, batch_size=64, seed=0):
        self.cfg = dict(hidden=hidden, attn=attn, dropout=dropout, weight_decay=weight_decay, lr=lr,
                        max_epochs=max_epochs, patience=patience, batch_size=batch_size, seed=seed)
        self.dim_in = dim_in

    def _build(self):
        import torch
        nn = torch.nn
        c = self.cfg

        class Net(nn.Module):
            def __init__(s):
                super().__init__()
                s.proj = nn.Sequential(nn.Linear(self.dim_in, c["hidden"]), nn.ReLU(), nn.Dropout(c["dropout"]))
                s.v = nn.Linear(c["hidden"], c["attn"])
                s.u = nn.Linear(c["hidden"], c["attn"])
                s.w = nn.Linear(c["attn"], 1)
                s.out = nn.Linear(c["hidden"], 1)

            def forward(s, x):                       # x: [B, T, D]
                h = s.proj(x)                        # [B, T, H]
                a = s.w(torch.tanh(s.v(h)) * torch.sigmoid(s.u(h))).squeeze(-1)  # [B, T]
                a = torch.softmax(a, dim=1)
                z = (a.unsqueeze(-1) * h).sum(1)     # [B, H]
                return s.out(z).squeeze(-1), a

        return Net()

    def fit(self, X, p):
        import torch
        c = self.cfg
        torch.manual_seed(c["seed"])
        rng = np.random.default_rng(c["seed"])
        idx = rng.permutation(len(X))
        n_val = max(1, int(0.15 * len(X)))
        va, tr = idx[:n_val], idx[n_val:]
        Xt, pt = torch.tensor(X, dtype=torch.float32), torch.tensor(p, dtype=torch.float32)
        self.net = self._build()
        opt = torch.optim.AdamW(self.net.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])
        loss_fn = torch.nn.BCEWithLogitsLoss()
        best, best_state, bad = np.inf, None, 0
        for _ in range(c["max_epochs"]):
            self.net.train()
            for b in np.array_split(rng.permutation(tr), max(1, len(tr) // c["batch_size"])):
                opt.zero_grad()
                logit, _ = self.net(Xt[b])
                loss_fn(logit, pt[b]).backward()
                opt.step()
            self.net.eval()
            with torch.no_grad():
                val = loss_fn(self.net(Xt[va])[0], pt[va]).item()
            if val < best - 1e-4:
                best, bad = val, 0
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= c["patience"]:
                    break
        self.net.load_state_dict(best_state)
        return self

    def predict(self, X, return_attention=False):
        import torch
        self.net.eval()
        with torch.no_grad():
            logit, a = self.net(torch.tensor(X, dtype=torch.float32))
        p = torch.sigmoid(logit).numpy()
        return (p, a.numpy()) if return_attention else p


def standardize(train, *others):
    mu = train.reshape(-1, train.shape[-1]).mean(0)
    sd = train.reshape(-1, train.shape[-1]).std(0) + 1e-6
    return [(x - mu) / sd for x in (train, *others)]


def run_variant(name, Xtr, ptr, Xte, test, seed, tag, n_seeds=3):
    grid = [1e-4, 1e-3, 1e-2]
    wd, cv = probe.pick_by_cv(lambda v: GatedAttention(Xtr.shape[-1], weight_decay=v, seed=seed),
                              grid, Xtr, ptr, seed)
    print(f"[{name}] weight decay {wd} (cv nll {cv})", flush=True)
    ps, atts = [], []
    for s in range(n_seeds):
        p, a = GatedAttention(Xtr.shape[-1], weight_decay=wd, seed=seed + s).fit(Xtr, ptr).predict(
            Xte, return_attention=True)
        ps.append(p)
        atts.append(a)
    p, a = np.mean(ps, 0), np.mean(atts, 0)
    u = probe.uncertainty_from_p(p)
    row = {"embeddings": tag, "method": name, **probe.point_metrics(test, p, u),
           **probe.bootstrap_ci(test, p, u, seed)}
    return row, p, u, a, {"weight_decay": wd, "cv_nll": cv}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--npz", default=str(HERE.parent / "embeddings_full" / "musk_tokens.npz"))
    parser.add_argument("--out", default=str(HERE.parent / "results_musk_layers"))
    parser.add_argument("--grid-layer", default="24",
                        help="one layer (e.g. 12), or several (e.g. 12,18,24) combined per --combine")
    parser.add_argument("--combine", choices=("stack", "tokens"), default="stack",
                        help="several layers: 'stack' joins each grid cell's vectors across layers "
                             "(36 tokens, wider), 'tokens' keeps every layer's cells as separate tokens "
                             "(36 x layers tokens, so attention picks location and layer)")
    parser.add_argument("--variants", default="attn_tokens,attn_tokens+cls")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    data = np.load(args.npz)
    layers = [int(v) for v in args.grid_layer.split(",")]
    missing = [L for L in layers if L not in list(data["grid_layers"])]
    if missing:
        raise SystemExit(f"grid layer(s) {missing} not in {args.npz} (has {list(data['grid_layers'])})")
    ann = mhist.load_annotations(args.annotations).set_index("image").loc[list(data["images"])].reset_index()
    tr, te = (ann["split"] == "train").to_numpy(), (ann["split"] == "test").to_numpy()
    test = ann[te].reset_index(drop=True)
    ptr = ann.loc[tr, "p_ssa"].to_numpy()
    # Standardize each layer on its own (layers have different scales), then combine.
    def scaled(x):  # fit on train rows, apply to all rows
        return standardize(x[tr], x)[1]

    per_layer = [scaled(data["grids"][:, list(data["grid_layers"]).index(L)].astype(np.float32))
                 for L in layers]                                          # each [N, 36, D]
    if len(layers) == 1:
        grids = per_layer[0]
    elif args.combine == "stack":
        grids = np.concatenate(per_layer, axis=2)                          # [N, 36, D * layers]
    else:
        grids = np.concatenate(per_layer, axis=1)                          # [N, 36 * layers, D]
    cls = data["layer_feats"][:, list(data["layers"]).index(24), 0].astype(np.float32)[:, None]  # [N, 1, D]
    cls = scaled(cls)

    layer_tag = "-".join(map(str, layers)) + ("" if len(layers) == 1 else f"_{args.combine}")
    out = Path(args.out) / f"attention_head_L{layer_tag}"
    out.mkdir(parents=True, exist_ok=True)
    rows, preds, chosen, maps = [], test[["image", "votes_ssa", "conflict", "y"]].copy(), {}, {}
    inputs = {"attn_tokens": lambda: grids, "attn_tokens+cls": lambda: np.concatenate([grids, cls], 1)}
    for name in args.variants.split(","):
        X = inputs[name]()
        row, p, u, a, hp = run_variant(name, X[tr], ptr, X[te], test, args.seed, f"musk_L{layer_tag}_grid")
        rows.append(row)
        preds[f"p_{name}"], preds[f"u_{name}"] = p, u
        chosen[name], maps[name] = hp, a
    table = pd.DataFrame(rows)
    table.to_csv(out / "metrics.csv", index=False)
    preds.to_csv(out / "test_predictions.csv", index=False)
    (out / "hyperparameters.json").write_text(pd.Series(chosen).to_json(indent=2))
    np.savez_compressed(HERE.parent / "embeddings_full" / f"attention_maps_L{layer_tag}.npz",
                        images=test["image"].to_numpy(), **{k.replace("+", "_"): v for k, v in maps.items()})
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(table[["method", "label_auroc", "label_acc", "binomial_nll", "disagree_spearman",
                     "disagree_auroc"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
