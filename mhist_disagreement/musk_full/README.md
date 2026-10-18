# MUSK full features (own GPU)

The hosted MUSK server returns 1024-d retrieval-head vectors, built for
matching images to text. MUSK's model card recommends a different setting for
training classifiers on frozen features: multi-scale features without the
retrieval head, which are 2048-d. This folder computes those on your own GPU
allocation, plus two companions:

| Output (`../embeddings_full/`) | Dim | What |
|---|---|---|
| `musk_ms.csv` | 2048 | multi-scale, no head (`ms_aug=True, with_head=False`) |
| `musk_ms_tta.csv` | 2048 | same, averaged over 8 flips/rotations per image |
| `musk_retrieval.csv` | 1024 | what the hosted server returns; sanity check against `embeddings/musk.csv` |

All three use the same CSV format as `embed_server.py`, so `probe.py`,
`pca_analysis.py`, `vote_heatmaps.py` and `triage.py` work on them unchanged.

**Time-sensitive:** the reserved A100s end Sun 6 pm. After that, GPU jobs go to
the regular queue, which this weekend means days. Submit before then.

## One-time setup (login node)

```bash
cd Ashby2026-Team51/mhist_disagreement/musk_full
cp .env.example .env          # then edit .env and paste your HF token
module load pytorch-conda/2.8
unset PYTHONNOUSERSITE
pip install --user git+https://github.com/lilab-stanford/MUSK.git fairscale sentencepiece
python embed_musk_full.py --download-only     # fetches ~3.7 GB of weights into the HF cache
```

Downloading on the login node is fine (network only) and means the GPU job
doesn't depend on compute-node internet access. Your HF account must already be
approved for `xiangjx/musk`; a read token is enough.

## Run (GPU job)

```bash
mkdir -p logs
sbatch embed_musk_full.sbatch ~/mhist/annotations.csv ~/images
tail -f logs/<jobid>.out
```

Use `--variants ms,retrieval` to skip the slower flip/rotation averaging.

## Evaluate (CPU, laptop or Delta)

From `mhist_disagreement/`:

```bash
python probe.py --annotations data/annotations.csv \
    --embeddings musk_ms=embeddings_full/musk_ms.csv \
    --embeddings musk_ms_tta=embeddings_full/musk_ms_tta.csv \
    --embeddings musk_retrieval=embeddings_full/musk_retrieval.csv \
    --out results_musk_full
```

`musk_retrieval` should score about the same as `results_musk` (same features,
computed locally). If it doesn't, the pipeline differs from the server and the
other two variants can't be trusted yet.
