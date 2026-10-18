#!/usr/bin/env python3
"""Extract MUSK features *before* they are summarized into one vector per tile.

MUSK's normal output is a single class-token vector from the last layer. This
script keeps two kinds of richer information, from one forward pass per image
(scale 1, 384 px, no multi-scale or flips):

    layers   class token and mean of the patch tokens from intermediate layers
             (default 6, 12, 18, 24 of 24), 1024-d each. Tests whether middle
             layers carry more of the HP/SSA signal than the last one.
    grid     the 24x24 grid of patch tokens from chosen layers (default the
             last), average-pooled to 6x6 = 36 vectors of 1024-d per image. Input
             for an attention head that can learn which part of the tile matters.

Everything goes into one compressed .npz (float16). Run to_csv.py afterwards
to turn layer features into the usual embedding CSVs for probe.py.

On the first run it writes model_structure.txt (module tree, layer count,
observed token shapes) next to the output, so problems can be diagnosed.

Loading and the HF token reuse ../musk_full (same .env, same weight cache).

    python extract_musk_tokens.py --annotations ~/mhist/annotations.csv --images ~/images
"""

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "musk_full"))
import mhist  # noqa: E402
import embed_musk_full as emf  # noqa: E402

GRID_SIZE = 24   # 384 px / 16 px patches


def load_musk(device: str, token: str):
    """Return (model, preprocess, dtype). Same loading as musk_full."""
    import timm
    import torch
    from musk import modeling  # noqa: F401  (registers musk_large_patch16_384 with timm)
    from musk import utils as musk_utils
    from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
    from torchvision import transforms

    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = timm.create_model("musk_large_patch16_384")
    musk_utils.load_model_and_may_interpolate(emf.download_weights(token), model, "model|module", "")
    model = model.to(device=device, dtype=dtype).eval()
    preprocess = transforms.Compose([
        transforms.Resize(384, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.CenterCrop((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
    ])
    return model, preprocess, dtype


def make_encoder(model, device, dtype):
    """encode(batch) -> (all_hidden_states, final_out).

    Mirrors BEiT3.forward's image-only path (musk/torchscale/model/BEiT3.py) but
    asks the encoder for every layer's output. all_hidden_states[0] is the patch
    embedding, [k] is the output of layer k (before the final LayerNorm);
    final_out is the last layer after the final LayerNorm, which is what MUSK
    itself returns. Token 0 is the class token, then the patch grid row by row."""
    import torch

    @torch.no_grad()
    def encode(batch):
        x = model.beit3.vision_embed(batch.to(device=device, dtype=dtype), None)
        out = model.beit3.encoder(src_tokens=None, token_embeddings=x,
                                  multiway_split_position=-1, return_all_hiddens=True)
        return out["encoder_states"], out["encoder_out"]

    return encode


def summarize(states, final, layers, grid_layers, pool):
    """Turn one batch's hidden states into (layer_feats [B, L, 2, D], grids [B, G, pool*pool, D])."""
    import torch
    import torch.nn.functional as F

    def pick(k):
        return final if k == len(states) - 1 else states[k]

    feats = torch.stack([torch.stack([pick(k)[:, 0], pick(k)[:, 1:].mean(1)], dim=1) for k in layers], dim=1)
    grids = []
    for k in grid_layers:
        patches = pick(k)[:, 1:]
        b, n, d = patches.shape
        side = int(round(n ** 0.5))
        img = patches.transpose(1, 2).reshape(b, d, side, side).float()
        grids.append(F.adaptive_avg_pool2d(img, pool).flatten(2).transpose(1, 2))
    return feats.float().cpu().numpy(), torch.stack(grids, dim=1).cpu().numpy()


def write_structure(path: Path, model, states, final, layers, grid_layers):
    lines = [
        f"encoder layers: {model.beit3.encoder.num_layers}",
        f"hidden states returned: {len(states)} (index 0 = patch embedding, k = after layer k)",
        f"token tensor shape (batch, tokens, dim): {tuple(final.shape)}  "
        f"[expected tokens = 1 class + {GRID_SIZE}x{GRID_SIZE} = {1 + GRID_SIZE ** 2}]",
        f"layers saved: {layers}   grid layers saved: {grid_layers}",
        "",
        "top-level children of model:",
        *[f"  {name}: {type(m).__name__}" for name, m in model.named_children()],
        "",
        "full module tree:",
        str(model),
    ]
    path.write_text("\n".join(lines))
    print(f"wrote {path}")


def run(paths, names, model, preprocess, encode, layers, grid_layers, pool, batch_size, out: Path):
    import torch
    from PIL import Image

    n_layers = model.beit3.encoder.num_layers
    for k in list(layers) + list(grid_layers):
        if not 1 <= k <= n_layers:
            raise SystemExit(f"layer {k} is out of range; MUSK has layers 1..{n_layers}")
    out.parent.mkdir(parents=True, exist_ok=True)
    structure = out.parent / "model_structure.txt"
    all_feats, all_grids = [], []
    for start in range(0, len(paths), batch_size):
        chunk = paths[start:start + batch_size]
        batch = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in chunk])
        states, final = encode(batch)
        if start == 0:
            if final.shape[1] != 1 + GRID_SIZE ** 2:
                write_structure(structure, model, states, final, layers, grid_layers)
                raise SystemExit(f"unexpected token count {final.shape[1]}; see {structure}")
            if not structure.exists():
                write_structure(structure, model, states, final, layers, grid_layers)
        feats, grids = summarize(states, final, layers, grid_layers, pool)
        all_feats.append(feats.astype(np.float16))
        all_grids.append(grids.astype(np.float16))
        print(f"  {min(start + batch_size, len(paths))}/{len(paths)}", flush=True)
    np.savez_compressed(
        out, images=np.array(names), layers=np.array(layers), grid_layers=np.array(grid_layers),
        layer_feats=np.concatenate(all_feats),   # [N, len(layers), 2 (cls, patch_mean), 1024]
        grids=np.concatenate(all_grids),          # [N, len(grid_layers), pool*pool, 1024]
    )
    print(f"wrote {out} ({out.stat().st_size / 1e6:.0f} MB)")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--out", default=str(HERE.parent / "embeddings_full" / "musk_tokens.npz"))
    parser.add_argument("--layers", default="6,12,18,24", help="layers for class/mean features (1-24)")
    parser.add_argument("--grid-layers", default="24", help="layers whose patch grid is saved")
    parser.add_argument("--pool", type=int, default=6, help="pool the 24x24 patch grid to pool x pool")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None)
    parser.add_argument("--env", default=None, help="default: ./.env, else ../musk_full/.env")
    args = parser.parse_args(argv)

    env = Path(args.env) if args.env else next(
        (p for p in (HERE / ".env", HERE.parent / "musk_full" / ".env") if p.exists()), HERE / ".env")
    emf.load_env(env)
    token = emf.hf_token()
    layers = [int(v) for v in args.layers.split(",")]
    grid_layers = [int(v) for v in args.grid_layers.split(",")]

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ann = mhist.load_annotations(args.annotations)
    folder = Path(args.images)
    paths = [str(folder / name) for name in ann["image"]]
    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} annotated images not found under {folder}, e.g. {missing[0]}")

    model, preprocess, dtype = load_musk(device, token)
    encode = make_encoder(model, device, dtype)
    run(paths, list(ann["image"]), model, preprocess, encode, layers, grid_layers, args.pool,
        args.batch_size, Path(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
