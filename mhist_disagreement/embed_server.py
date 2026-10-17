#!/usr/bin/env python3
"""Embed every MHIST image through a hosted model server's /embed endpoint.

Writes a CSV with `image_path` and `embedding` (JSON list), the same format
as the ncsa repo's run_embeddings.py, so either source works with probe.py.
Rerun with the same --out to resume. Network-only, fine on the login node.

    python embed_server.py --endpoint http://gpub080:8001 --api-key $KEY \
        --images /path/to/mhist/images --out embeddings/musk.csv
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

import hosted


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    hosted.add_server_args(parser)
    parser.add_argument("--out", required=True, help="output CSV; rerun to resume")
    args = parser.parse_args(argv)
    hosted.check_server_args(args)

    paths = hosted.find_images(args.images)
    out = Path(args.out)
    done = set(pd.read_csv(out)["image_path"]) if out.exists() else set()
    todo = [p for p in paths if p not in done]
    print(f"{len(paths)} images, {len(done)} already embedded, {len(todo)} to go")
    out.parent.mkdir(parents=True, exist_ok=True)

    for i, batch in enumerate(hosted.batches(todo, args.batch_size)):
        result = hosted.post(args, "embed", {"images": [hosted.encode(p) for p in batch]})
        rows = pd.DataFrame({
            "image_path": batch,
            "embedding": [json.dumps(e) for e in result["embeddings"]],
        })
        rows.to_csv(out, mode="a", header=not out.exists(), index=False)
        print(f"  {min((i + 1) * args.batch_size, len(todo))}/{len(todo)}", flush=True)

    total = len(pd.read_csv(out))
    print(f"done: {total} rows in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
