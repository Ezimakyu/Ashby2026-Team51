# Final model: stacked MUSK layer heads

This folder holds the results of the final model: a weighted combination of
small heads, each reading a different layer of a frozen MUSK. The code lives in
[`../musk_unsummarized/`](../musk_unsummarized/) (extraction, attention head,
stacking). Everything below is on the official MHIST test split (977 images,
350 contested) and was evaluated once.

## Strategy

**1. Read MUSK before it summarizes.** MUSK is a stack of 24 transformer
layers. Each layer outputs 577 vectors of 1024 numbers: one class token (MUSK's
built-in summary) plus one vector per 16×16-px patch of the 384-px tile (a 24×24
grid). MUSK normally returns only layer 24's class token. We saved, in one
frozen forward pass per image
([`extract_musk_tokens.py`](../musk_unsummarized/extract_musk_tokens.py)):
- layers 6, 12, 18 and 24: the class token and the average of the 576 patch
  vectors (1024-d each)
- layers 12, 18 and 24: the patch grid, average-pooled to 6×6 (36 vectors of 1024-d)

**2. One small head per layer, trained on vote fractions.** Every head
predicts p(SSA) and is trained on the share of the 7 pathologists who chose SSA
(e.g. 4/7 = 0.57), not the majority label. "How contested" is how close p is to 0.5.
- *Layer-vector heads:* soft-label logistic regression on each of the 8 vectors.
- *Attention heads* ([`attention_head.py`](../musk_unsummarized/attention_head.py)):
  on each 6×6 grid, a small gated-attention head (~140k weights) learns how
  much each grid cell should count, takes the weighted average and predicts p(SSA).
  It can focus on the part of the tile that matters (e.g. crypt bases) instead of
  MUSK's fixed summary.

**3. Stack them with one interpretable weight per head**
([`layer_stack.py`](../musk_unsummarized/layer_stack.py)). The final p(SSA) is a
weighted average of the heads' p(SSA), with weights ≥ 0 that sum to 1. No test
data is used to build it:
1. each head's regularization is chosen by 5-fold CV on train
2. each head predicts every training image from a model that never saw it
   (5-fold out-of-fold predictions)
3. the weights minimize the binomial NLL of those out-of-fold predictions against the vote fractions
4. heads are refit on all of train, and the test set is scored once

Two versions: `stack_logreg` (the 8 layer-vector heads only) and
`stack_logreg+attn` (the 8 plus the 3 attention heads, **the final model**).

MUSK stays frozen throughout. The only GPU step is the one extraction pass;
every head and the stack train in minutes on a laptop CPU.

## Learned weights

| Head | `stack_logreg` | `stack_logreg+attn` (final) |
|---|---|---|
| Layer 6: class token / patch average | 0 / 0 | 0 / 0 |
| Layer 12: class token / patch average | 0 / **0.21** | 0 / 0 |
| Layer 18: class token / patch average | 0 / **0.47** | 0 / 0.13 |
| Layer 24: class token / patch average | 0.13 / 0.18 | 0 / 0 |
| Layer 12 / 18 / 24 attention head | — | **0.51** / 0.10 / 0.26 |

- **Layer 6 is never used.** It's too early in MUSK to help.
- **Patch averages beat class tokens.** Only layer 24's class token gets any weight.
- **With attention heads available, they take 87% of the weight,** led by layer 12.

Bar charts: `stack_logreg/weights.png`, `stack_logreg+attn/weights.png`.

## Test results vs the deck's model

The deck's best model is the soft-label logistic regression on MUSK's 2048-d
multi-scale features (slide 10). Its numbers here come from our evaluation code,
which gives disagreement AUROC 0.735 where the slide shows 0.740. Ranges are 95% bootstrap intervals.

| Metric | Deck: MUSK 2048-d soft head | `stack_logreg` | `stack_logreg+attn` (final) |
|---|---|---|---|
| Label AUROC (HP vs SSA) | 0.900 [0.88, 0.92] | 0.922 [0.91, 0.94] | **0.942** [0.93, 0.96] |
| Accuracy (vs majority vote; always-HP = 63.2%) | 82.8% | 85.0% | **87.6%** |
| Balanced accuracy | 82.2% | 84.2% | **87.2%** |
| SSA sensitivity | 79.7% | 81.4% | **85.8%** |
| HP specificity | 84.6% | 87.0% | **88.7%** |
| NLL vs vote fractions (lower is better) | 0.519 | 0.502 | **0.482** |
| Disagreement AUROC (2–5 SSA votes vs unanimous) | 0.735 [0.70, 0.77] | 0.774 [0.74, 0.81] | **0.819** [0.79, 0.85] |
| Disagreement AUROC (3–4 split vs unanimous) | 0.744 | 0.779 | **0.833** |
| Disagreement Spearman (uncertainty vs conflict) | 0.340 | 0.393 | **0.465** |
| Exact / within-1 SSA vote count | 27.7% / 65.7% | 27.5% / 68.1% | **29.1% / 73.0%** |
| Clear-SSA images (6–7 votes) put in the right group | 29% | 30% | **42%** |

**Triage** (flag the most uncertain cases for a second read):

| Flag | Deck model: caught / accuracy on rest | `stack_logreg+attn`: caught / accuracy on rest | Random: caught |
|---|---|---|---|
| 10% | 15% / 85.0% | 18% / 90.7% | 10% |
| 20% | 29% / 88.2% | **37% / 93.0%** | 20% |
| 30% | 44% / 91.1% | **51% / 95.3%** | 30% |

> Flagging the 20% most uncertain cases catches 37% of the cases pathologists
> disputed (1.8× random), and accuracy on the remaining 80% rises from 87.6% to 93.0%.

The stacked model also beats a plain equal-weight average of the same heads
(disagreement AUROC 0.819 vs 0.784), so the learned weights matter.

## Caveats

- One institution, one official split. We haven't verified that the split keeps each patient on one side.
- The heads and the stack never saw the test set. The choice of which layers' grids to
  save (12, 18, 24) was made after a layer comparison that did look at test scores.
  It's a minor, but real, source of optimism.
- 224-px tiles (upscaled to 384 for MUSK) can miss crypt-base context a pathologist would see on the full slide.

## What's in this folder

| Path | Contents |
|---|---|
| `stack_logreg+attn/` | **final model**: `metrics.csv` (stack, equal average, each head), `weights.csv`/`.png`, `bases.json` (each head's chosen hyperparameter), `vote_heatmap.png` (true vs predicted 0–7 votes), `conflict_heatmap.png`, `vote_heatmap_summary.csv`, `test_predictions.csv` (gitignored) |
| `stack_logreg/` | the same for the 8 layer-vector heads only |
| `triage_stacked/` | triage curve, headline numbers, `triage_summary.md` (plain-English sentences), with the deck model for reference |
| `agreement_groups.png` / `.csv` | 3-group heatmap (0–1, 2–5, 6–7 SSA votes) for both stacks |
| `agreement_groups_deck_reference.png` / `.csv` | the same for the deck model |
| `overview.png` / `.csv` | every representation we tried, in one figure (copy of `../results_compare/overview.*`) |
| `individual_musklayers/` | the details: each layer head on its own (`musk_L*_cls`, `musk_L*_mean`, `musk_L24_gridmean`), each attention head (`attention_head_L12/18/24`), `layer_vectors_summary.csv`, and `exploratory_multilayer_attention/` (multi-layer heads compared on the test set before stacking; superseded by the stack) |

## Reproduce (laptop, after the one GPU extraction)

From `mhist_disagreement/musk_unsummarized/`, with `../embeddings_full/musk_tokens_grids.npz`
from `extract.sbatch ... --grid-layers 12,18,24` (see that folder's README):

```bash
python attention_head.py --annotations ../data/annotations.csv --npz ../embeddings_full/musk_tokens_grids.npz --grid-layer 12 --variants attn_tokens
python attention_head.py --annotations ../data/annotations.csv --npz ../embeddings_full/musk_tokens_grids.npz --grid-layer 18 --variants attn_tokens
python attention_head.py --annotations ../data/annotations.csv --npz ../embeddings_full/musk_tokens_grids.npz --grid-layer 24 --variants attn_tokens
python layer_stack.py --annotations ../data/annotations.csv --npz ../embeddings_full/musk_tokens_grids.npz \
    --with-attention --out ../results_musklayers_multi/stack_logreg+attn
```

The attention runs only provide each head's weight decay (chosen by train CV) to
the stack. Each takes about 5 minutes; the stack takes about 15.
