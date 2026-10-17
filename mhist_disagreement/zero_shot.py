#!/usr/bin/env python3
"""Zero-shot HP vs SSA through a hosted server's /classify endpoint.

For each prompt pair in prompts_hp_ssl.json, sends exactly two prompts (HP and
SSA). With two labels the server's score is the softmax of the two cosine
similarities (serve.py classify()), so p(SSA) = score when SSA wins, else
1 - score. That's monotonic in the similarity gap, so it's fine for AUROC and
rank correlations; the absolute values sit close to 0.5 and aren't calibrated.

Writes `image_path`, one `p_ssa_<i>` per prompt pair, and their mean `p_ssa`.
Pass the output to probe.py --zero-shot to put it in the same results table.

    python zero_shot.py --endpoint http://gpub080:8001 --api-key $KEY \
        --images /path/to/mhist/images --out embeddings/musk_zeroshot.csv
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import hosted

HERE = Path(__file__).resolve().parent


def p_ssa_from_results(results) -> np.ndarray:
    return np.array([
        r["score"] if r["predicted_label"] == "SSA" else 1.0 - r["score"] for r in results
    ])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    hosted.add_server_args(parser)
    parser.add_argument("--prompts", default=str(HERE / "prompts_hp_ssl.json"),
                        help="JSON list of {\"HP\": ..., \"SSA\": ...} pairs")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    hosted.check_server_args(args)

    pairs = json.loads(Path(args.prompts).read_text())
    for pair in pairs:
        if set(pair) != {"HP", "SSA"}:
            raise SystemExit(f"Each prompt pair needs exactly the keys HP and SSA; got {sorted(pair)}")
    paths = hosted.find_images(args.images)
    print(f"{len(paths)} images x {len(pairs)} prompt pairs")

    cols = {f"p_ssa_{i}": [] for i in range(len(pairs))}
    for n, batch in enumerate(hosted.batches(paths, args.batch_size)):
        images = [hosted.encode(p) for p in batch]
        for i, pair in enumerate(pairs):
            result = hosted.post(args, "classify", {"images": images, "prompts": pair})
            cols[f"p_ssa_{i}"].extend(p_ssa_from_results(result["results"]))
        print(f"  {min((n + 1) * args.batch_size, len(paths))}/{len(paths)}", flush=True)

    df = pd.DataFrame({"image_path": paths, **cols})
    df["p_ssa"] = df[list(cols)].mean(axis=1)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"done: {len(df)} rows in {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
