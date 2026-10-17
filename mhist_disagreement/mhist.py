"""Load MHIST annotations and join them to an embeddings CSV.

MHIST's annotations.csv has one row per image: file name, majority label
(HP or SSA), how many of the 7 pathologists picked SSA, and the official
train/test partition. Column names are detected by substring so small
differences between copies of the CSV don't matter; pass explicit names via
the `columns` argument (or the CLI flags in probe.py) if detection fails.

Derived columns:
    votes_ssa  0..7, number of pathologists who chose SSA
    p_ssa      votes_ssa / 7, the soft label
    majority   "HP" or "SSA"
    y          1 if majority is SSA, else 0
    conflict   min(votes_ssa, 7 - votes_ssa): 0 = unanimous, 3 = 3-vs-4 split
    split      "train" or "test" ("val"/"validate"/"validation" count as test)
"""

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

N_ANNOTATORS = 7
EVAL_SPLIT_NAMES = {"test", "val", "valid", "validate", "validation"}


def _find_column(columns, *needles, exclude=()) -> Optional[str]:
    for col in columns:
        low = col.lower()
        if all(n in low for n in needles) and not any(e in low for e in exclude):
            return col
    return None


def detect_columns(df: pd.DataFrame, overrides: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    overrides = {k: v for k, v in (overrides or {}).items() if v}
    cols = list(df.columns)
    found = {
        "image": _find_column(cols, "image") or _find_column(cols, "file") or _find_column(cols, "name"),
        "votes": _find_column(cols, "ssa", "annotator") or _find_column(cols, "ssa", "number")
        or _find_column(cols, "votes"),
        "majority": _find_column(cols, "majority") or _find_column(cols, "label", exclude=("number",)),
        "split": _find_column(cols, "partition") or _find_column(cols, "split"),
    }
    found.update(overrides)
    missing = [k for k in ("image", "votes", "split") if not found.get(k)]
    if missing:
        raise SystemExit(
            f"Couldn't find column(s) {missing} in the annotations CSV. Columns are: {cols}. "
            "Pass them explicitly (e.g. --votes-col 'Number of Annotators who Selected SSA (Out of 7)')."
        )
    for key, col in found.items():
        if col and col not in df.columns:
            raise SystemExit(f"Column {col!r} (for {key}) isn't in the annotations CSV. Columns are: {cols}")
    return found


def load_annotations(path: str, overrides: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    raw = pd.read_csv(path)
    cols = detect_columns(raw, overrides)

    out = pd.DataFrame()
    out["image"] = raw[cols["image"]].astype(str).map(lambda p: Path(p).name)
    out["votes_ssa"] = raw[cols["votes"]].astype(int)
    if not out["votes_ssa"].between(0, N_ANNOTATORS).all():
        raise SystemExit(f"Vote column {cols['votes']!r} has values outside 0..{N_ANNOTATORS}.")
    out["p_ssa"] = out["votes_ssa"] / N_ANNOTATORS
    if cols.get("majority"):
        out["majority"] = raw[cols["majority"]].astype(str).str.strip().str.upper()
    else:
        out["majority"] = np.where(out["votes_ssa"] > N_ANNOTATORS / 2, "SSA", "HP")
    # MHIST uses "SSA"; accept "SSL" too, the newer name for the same lesion.
    out["y"] = out["majority"].isin(["SSA", "SSL"]).astype(int)
    out["conflict"] = np.minimum(out["votes_ssa"], N_ANNOTATORS - out["votes_ssa"])
    split = raw[cols["split"]].astype(str).str.strip().str.lower()
    unknown = set(split) - EVAL_SPLIT_NAMES - {"train"}
    if unknown:
        raise SystemExit(f"Unknown split value(s) {sorted(unknown)} in {cols['split']!r}; expected train/test.")
    out["split"] = np.where(split == "train", "train", "test")
    return out


def load_embeddings(path: str) -> pd.DataFrame:
    """Read a CSV with `image_path` and `embedding` (JSON list) columns, the
    format written by embed_server.py and the ncsa repo's run_embeddings.py."""
    df = pd.read_csv(path)
    if "embedding" not in df.columns or "image_path" not in df.columns:
        raise SystemExit(f"{path} needs image_path and embedding columns; it has {list(df.columns)}.")
    df["image"] = df["image_path"].astype(str).map(lambda p: Path(p).name)
    df["embedding"] = df["embedding"].map(lambda s: np.asarray(json.loads(s), dtype=np.float32))
    return df[["image", "embedding"]].drop_duplicates("image", keep="last")


def join(annotations: pd.DataFrame, embeddings: pd.DataFrame) -> pd.DataFrame:
    df = annotations.merge(embeddings, on="image", how="inner")
    if df.empty:
        raise SystemExit("No image names in the embeddings CSV match the annotations CSV.")
    dropped = len(annotations) - len(df)
    if dropped:
        print(f"warning: {dropped} annotated images have no embedding and are skipped")
    return df.reset_index(drop=True)
