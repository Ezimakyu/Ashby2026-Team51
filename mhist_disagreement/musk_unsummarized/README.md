# MUSK features before summarizing

Every head so far sees one vector per tile: MUSK's last-layer class token,
which summarizes 576 patch vectors. This folder keeps what that summary throws
away, so small heads can test whether it matters:

| Saved (in `../embeddings_full/musk_tokens.npz`) | Shape per tile | Question it answers |
|---|---|---|
| `layer_feats`: class token + mean of patch tokens from layers 6, 12, 18, 24 | 4 x 2 x 1024 | Do middle layers (more cell/gland detail) separate HP vs SSA better than the last? |
| `grids`: last layer's 24x24 patch grid, pooled to 6x6 | 36 x 1024 | Input for an attention head that learns which part of the tile matters |

One forward pass per image at 384 px (no multi-scale, no flips), stored as
float16: about 50 MB of layer features and 230 MB of grid per pooled layer.

Layer 24 is taken after MUSK's final LayerNorm, so its class token is exactly
what MUSK itself returns. Layers 6/12/18 are raw block outputs.

## Run (Delta, before Sun 6 pm)

Uses the same setup as `../musk_full` (installed packages, `.env` with your HF
token, cached weights). If you've already run that, nothing else is needed.

```bash
cd Ashby2026-Team51/mhist_disagreement/musk_unsummarized
mkdir -p logs
sbatch extract.sbatch ~/mhist/annotations.csv ~/images
```

Options: `--layers 3,6,9,12,15,18,21,24` for a finer sweep, `--grid-layers 18,24`
to also save a middle layer's grid, `--pool 8` for an 8x8 grid.

On the first batch it writes `../embeddings_full/model_structure.txt`: the layer
count, observed token shape and full module tree. If the job fails, send that
file and `logs/<jobid>.out`.

## Afterwards (laptop)

Copy `embeddings_full/musk_tokens.npz` over, then from this folder:

```bash
python to_csv.py --check-against ../embeddings_full/musk_ms.csv
```

The check should print a cosine similarity of ~1.0. That confirms the
extraction matches MUSK's own output. It then writes one CSV per layer and
kind to `../embeddings_full/layers/`, ready for `probe.py`.

## Attention head (laptop)

```bash
python attention_head.py --annotations ../data/annotations.csv \
    --npz ../embeddings_full/musk_tokens_grids.npz --grid-layer 12 --variants attn_tokens
```

Writes `../results_musk_layers/attention_head_L<layer>/` (metrics, test predictions,
hyperparameters) and per-image attention maps to `../embeddings_full/attention_maps_L<layer>.npz`.
About 5 minutes per layer on a laptop CPU. `python ../overview.py` then rebuilds the
all-methods comparison in `../results_compare/`.
