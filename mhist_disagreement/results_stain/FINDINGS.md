# Stain robustness: findings (MUSK + soft head)

**Question.** Does the HP/SSA call change under mild image perturbations, and do
images the pathologists disagreed on change more?

**Setup.** 299 MHIST test images, sampled to keep each agreement group's share
(`stain_robustness.py --n 300`). Each image gets 3 stain-shifted copies
(`perturb.stain_shift`, strength 0.2, seeds 0–2) and 1 blurred copy (Gaussian,
radius 1.5 px), embedded through the hosted MUSK server. The call comes from the
soft-label logistic regression trained on the train split (`heads.py`; test
label AUROC 0.879, same as `results_musk/summary.csv`). A flip = the call
(p(SSA) ≥ 0.5) differs from the unperturbed image's.
Per-group numbers with 95% intervals: `musk/summary.csv`; chart:
`musk/flip_by_agreement.png`.

## Results

| Pathologists agreeing | n | Any copy flipped | Stain flips | Blur flips | Mean \|Δp(SSA)\| | Mean \|Δlogit\| | Median \|p − 0.5\| |
|---|---|---|---|---|---|---|---|
| 7/7 | 99 | 19% | 1.3% | 17% | 0.052 | 0.347 | 0.30 |
| 6/7 | 93 | 26% | 3.6% | 24% | 0.061 | 0.353 | 0.24 |
| 5/7 | 57 | 33% | 7.6% | 28% | 0.077 | 0.400 | 0.14 |
| 4/7 | 50 | 40% | 5.3% | 30% | 0.081 | 0.396 | 0.15 |
| All | 299 | 27% | 3.9% | 23% | 0.065 | 0.367 | 0.22 |

Trend with conflict (0 = 7/7 … 3 = 4/7), Spearman across images:

| Measure | ρ | p |
|---|---|---|
| Number of flips | 0.17 | 0.003 |
| \|Δp(SSA)\| | 0.29 | 2 × 10⁻⁷ |
| \|Δlogit p(SSA)\| | 0.12 | 0.03 |

## What it means

1. **Contested images flip about twice as often** (40% of 4:3 splits vs 19% of
   unanimous images), and the trend is monotonic across the four groups.
2. **That's mostly because they sit near the decision boundary**, not because
   their embeddings are much less stable. Contested images start closer to
   p = 0.5 (median margin 0.15 vs 0.30), so the same shift crosses the threshold
   more often. |Δp| is inflated there too, because the sigmoid is steepest at
   0.5. The logit change, which removes both effects, rises only 14% from 7/7 to
   4/7 (ρ = 0.12): a real but small extra instability.
3. **The model is robust to stain shift and sensitive to blur.** Stain shifts
   flip 1–8% of calls; a 1.5 px blur flips 17–30%, including 17% of images all
   seven pathologists agreed on.

**One-line version.** Under mild perturbations the HP/SSA call flipped for 19%
of unanimous images but 40% of 4:3 splits, mostly because contested cases sit
on the decision boundary; the model is robust to stain shift (1–8% flips) but
sensitive to blur (17–30%). Cases pathologists disagree on are where small image
changes can tip the call: another reason to route them to a second review.

## Caveats

- One model (MUSK) and one head (logreg_soft). CONCH, QuiltNet and PLIP weren't run.
- 50–57 images in the 4/7 and 5/7 groups, so their intervals are wide (see `summary.csv`).
- One perturbation strength each. Blur at 1.5 px on a 224 px tile is not
  subtle; a dose-response sweep (strength 0.1–0.5, radius 0.5–2) would show
  where calls start to break.
- Synthetic perturbations, not real scanner or lab variation.
- MHIST's official split hasn't been checked for patient overlap (see the main README).
