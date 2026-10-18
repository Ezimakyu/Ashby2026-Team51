#!/usr/bin/env python3
"""Turn musk_tokens.npz layer features into the usual embedding CSVs.

Writes one CSV per (layer, kind) to --out-dir, named musk_L<layer>_<cls|mean>.csv,
in the image_path,embedding format probe.py reads. Optionally also a
"grid mean" CSV (the 36 pooled patch vectors averaged) as a sanity reference.

With --check-against embeddings_full/musk_ms.csv it compares the last layer's
class token with the first 1024 dims of the multi-scale features (that half
is the same scale-1 class token), to confirm the extraction matches MUSK's
own output.

    python to_csv.py --npz ../embeddings_full/musk_tokens.npz --out-dir ../embeddings_full/layers
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import mhist  # noqa: E402

KINDS = ("cls", "mean")


def write_csv(path: Path, images, vectors):
    pd.DataFrame({
        "image_path": images,
        "embedding": [json.dumps(np.round(v.astype(np.float32), 5).tolist()) for v in vectors],
    }).to_csv(path, index=False)
    print(f"wrote {path} ({len(images)} rows, dim {vectors.shape[1]})")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--npz", default=str(HERE.parent / "embeddings_full" / "musk_tokens.npz"))
    parser.add_argument("--out-dir", default=str(HERE.parent / "embeddings_full" / "layers"))
    parser.add_argument("--check-against", help="musk_ms.csv from musk_full, for a sanity check")
    args = parser.parse_args(argv)

    data = np.load(args.npz)
    images, layers, feats = data["images"], data["layers"], data["layer_feats"]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for li, layer in enumerate(layers):
        for ki, kind in enumerate(KINDS):
            write_csv(out / f"musk_L{layer}_{kind}.csv", images, feats[:, li, ki])
    for gi, layer in enumerate(data["grid_layers"]):
        write_csv(out / f"musk_L{layer}_gridmean.csv", images, data["grids"][:, gi].astype(np.float32).mean(1))

    if args.check_against:
        if layers[-1] != 24:
            print("check skipped: layer 24 wasn't extracted")
            return 0
        ref = mhist.load_embeddings(args.check_against).set_index("image").loc[list(images)]
        ref_cls = np.stack(ref["embedding"])[:, :1024]
        mine = feats[:, list(layers).index(24), 0].astype(np.float32)
        cos = np.sum(ref_cls * mine, 1) / (np.linalg.norm(ref_cls, axis=1) * np.linalg.norm(mine, axis=1))
        print(f"check vs {args.check_against}: cosine similarity of last-layer class token, "
              f"median {np.median(cos):.4f}, min {cos.min():.4f} (expect ~1.0)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
