#!/usr/bin/env python3
"""Embed MHIST with MUSK on your own GPU, writing feature variants the hosted
server doesn't return.

Variants (each its own CSV, same image_path,embedding format as embed_server.py,
so probe.py / pca_analysis.py read them directly):
    ms          2048-d: MUSK's multi-scale features without the retrieval head
                (ms_aug=True, with_head=False). MUSK's model card recommends these
                for linear probes / classifiers on frozen features.
    ms_tta      `ms` averaged over the 8 flips/rotations of each image. Tissue has no
                "up", so this averages out orientation noise. 8x the compute.
    retrieval   1024-d retrieval-head features (with_head=True, out_norm=True), what
                the hosted server returns. Use it to check this pipeline matches
                embeddings/musk.csv.

Needs: HF access to xiangjx/musk approved on your account, a token in ../.env
(see .env.example), and the one-time installs in README.md. Run on a GPU node
(slurm job), except --download-only, which only fetches weights and is fine on
the login node.

    python embed_musk_full.py --annotations ~/mhist/annotations.csv --images ~/images \
        --out-dir embeddings_full
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import mhist  # noqa: E402

VARIANTS = ("ms", "ms_tta", "retrieval")
MUSK_REPO_ID = "xiangjx/musk"


def load_env(path: Path):
    """Minimal .env reader (KEY=value lines), so no extra package is needed.
    Values already set in the environment win."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def hf_token() -> str:
    token = os.environ.get("HF_TOKEN", "")
    if not token or token.startswith("hf_xxx"):
        raise SystemExit("HF_TOKEN isn't set. Copy .env.example to .env in this folder and paste your token.")
    return token


def download_weights(token: str) -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=MUSK_REPO_ID, filename="model.safetensors", token=token)


def load_musk(device: str, token: str):
    """Return (preprocess, encode) where encode(batch, variants) -> {variant: array}.
    Loading mirrors the ncsa repo's models.py load_musk()."""
    import timm
    import torch
    from musk import modeling  # noqa: F401  (registers musk_large_patch16_384 with timm)
    from musk import utils as musk_utils
    from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
    from torchvision import transforms

    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = timm.create_model("musk_large_patch16_384")
    musk_utils.load_model_and_may_interpolate(download_weights(token), model, "model|module", "")
    model = model.to(device=device, dtype=dtype).eval()

    preprocess = transforms.Compose([
        transforms.Resize(384, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.CenterCrop((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
    ])

    def run(batch, **kw):
        return model(image=batch.to(device=device, dtype=dtype), return_global=True, **kw)[0].float()

    @torch.no_grad()
    def encode(batch, variants):
        out = {}
        if "ms" in variants:
            out["ms"] = run(batch, with_head=False, out_norm=False, ms_aug=True).cpu().numpy()
        if "ms_tta" in variants:
            views = [torch.rot90(b, k, dims=(2, 3)) for b in (batch, torch.flip(batch, dims=(3,))) for k in range(4)]
            out["ms_tta"] = torch.stack([run(v, with_head=False, out_norm=False, ms_aug=True) for v in views]
                                        ).mean(0).cpu().numpy()
        if "retrieval" in variants:
            out["retrieval"] = run(batch, with_head=True, out_norm=True, ms_aug=False).cpu().numpy()
        return out

    return preprocess, encode


def already_done(path: Path) -> set:
    return set(pd.read_csv(path, usecols=["image_path"])["image_path"]) if path.exists() else set()


def extract(paths, preprocess, encode, variants, out_dir: Path, batch_size: int, tag: str):
    """Embed `paths` for every variant, appending to <out_dir>/<tag>_<variant>.csv.
    Rerun to resume: each CSV only gets the images it's missing."""
    import torch
    from PIL import Image

    outs = {v: out_dir / f"{tag}_{v}.csv" for v in variants}
    done = {v: already_done(p) for v, p in outs.items()}
    todo = [p for p in paths if any(p not in done[v] for v in variants)]
    print(f"{len(paths)} images, {len(todo)} still missing from at least one variant", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(todo), batch_size):
        chunk = todo[start:start + batch_size]
        batch = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in chunk])
        need = [v for v in variants if any(p not in done[v] for p in chunk)]
        embs = encode(batch, need)
        for v in need:
            keep = [i for i, p in enumerate(chunk) if p not in done[v]]
            rows = pd.DataFrame({
                "image_path": [chunk[i] for i in keep],
                "embedding": [json.dumps(np.round(embs[v][i], 6).tolist()) for i in keep],
            })
            rows.to_csv(outs[v], mode="a", header=not outs[v].exists(), index=False)
        print(f"  {min(start + batch_size, len(todo))}/{len(todo)}", flush=True)
    for v, p in outs.items():
        n = len(already_done(p))
        dim = len(json.loads(pd.read_csv(p, nrows=1)["embedding"][0])) if n else 0
        print(f"{p}: {n} rows, dim {dim}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotations", help="MHIST annotations.csv (picks which images to embed)")
    parser.add_argument("--images", help="folder containing the MHIST .png files")
    parser.add_argument("--out-dir", default=str(HERE.parent / "embeddings_full"))
    parser.add_argument("--variants", default=",".join(VARIANTS), help=f"comma list from {VARIANTS}")
    parser.add_argument("--tag", default="musk")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None, help="default: cuda if available")
    parser.add_argument("--env", default=str(HERE / ".env"))
    parser.add_argument("--download-only", action="store_true",
                        help="fetch the weights into the HF cache and exit (login node is fine)")
    args = parser.parse_args(argv)

    load_env(Path(args.env))
    token = hf_token()
    if args.download_only:
        print(f"weights at {download_weights(token)}")
        return 0
    if not args.annotations or not args.images:
        raise SystemExit("--annotations and --images are required (unless --download-only).")
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = set(variants) - set(VARIANTS)
    if unknown:
        raise SystemExit(f"Unknown variant(s) {sorted(unknown)}; choose from {VARIANTS}")

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        print("warning: no GPU found; this will be very slow. Run it as a GPU job (see README).")

    ann = mhist.load_annotations(args.annotations)
    folder = Path(args.images)
    paths = [str(folder / name) for name in ann["image"]]
    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} annotated images not found under {folder}, e.g. {missing[0]}")

    preprocess, encode = load_musk(device, token)
    extract(paths, preprocess, encode, variants, Path(args.out_dir), args.batch_size, args.tag)
    return 0


if __name__ == "__main__":
    sys.exit(main())
