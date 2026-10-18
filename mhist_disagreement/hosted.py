"""Shared helpers for calling the organizers' hosted model servers
(/embed and /classify, see ncsa/ashby-hackathon-2026 STUDENT_GUIDE.md).

Endpoint and key come from --endpoint/--api-key or the PATHOLOGY_ENDPOINT /
PATHOLOGY_API_KEY env vars. Don't commit either.
"""

import base64
import io
import os
from pathlib import Path
from typing import List

import requests

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def add_server_args(parser):
    parser.add_argument("--endpoint", default=os.environ.get("PATHOLOGY_ENDPOINT"),
                        help="e.g. http://gpub080:8001 (or set PATHOLOGY_ENDPOINT)")
    parser.add_argument("--api-key", default=os.environ.get("PATHOLOGY_API_KEY"),
                        help="bearer key from #compute-issues (or set PATHOLOGY_API_KEY)")
    parser.add_argument("--images", required=True, help="folder of MHIST image files (searched recursively)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=300.0)


def check_server_args(args):
    if not args.endpoint or not args.api_key:
        raise SystemExit("Need --endpoint and --api-key (or PATHOLOGY_ENDPOINT / PATHOLOGY_API_KEY).")
    args.endpoint = args.endpoint.rstrip("/")
    health = requests.get(f"{args.endpoint}/health", timeout=30).json()
    print(f"server: model={health.get('model')} device={health.get('device')}")
    return health


def find_images(folder: str) -> List[str]:
    root = Path(folder)
    if not root.is_dir():
        raise SystemExit(f"Image folder not found: {folder}")
    paths = sorted(str(p) for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not paths:
        raise SystemExit(f"No image files under {folder}")
    return paths


def encode(path: str) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


def encode_pil(img) -> str:
    """Base64 PNG of an in-memory PIL image (e.g. a perturbed tile)."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def post(args, route: str, payload: dict) -> dict:
    resp = requests.post(
        f"{args.endpoint}/{route}",
        json=payload,
        headers={"Authorization": f"Bearer {args.api_key}"},
        timeout=args.timeout,
    )
    if resp.status_code == 401:
        raise SystemExit("HTTP 401: API key is wrong or stale; re-check #compute-issues.")
    resp.raise_for_status()
    return resp.json()


def batches(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]
