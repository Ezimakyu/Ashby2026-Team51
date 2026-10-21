# Teaching Pathology AI Where Experts Disagree

Team 51, 2026 Ashby Prize in Computational Science Hackathon (NCSA)
Yvette Qin, Amanda Chi, George Wang, Bryce Liu, Stephen Zhu

## The question

Pathologists often disagree on whether a colon polyp is a hyperplastic polyp (HP) or a sessile serrated lesion (SSL), and the call sets whether the next colonoscopy is in 3 or 10 years. Most cases are read by one pathologist, who can't know whether colleagues would have called it differently.

Using [MHIST](https://bmirds.github.io/MHIST/), where 7 pathologists labeled each of 3,152 tiles, we asked whether pathology foundation models can identify the cases pathologists disagree on, and whether training on the vote split (for example 4 of 7) instead of the majority label teaches them to.

## Main results (MHIST test split, 977 tiles)

| | Disagreement AUROC | HP vs SSL AUROC |
|---|---|---|
| Zero-shot, 4 models (CONCH, MUSK, QuiltNet, PLIP) | 0.43 to 0.60 | 0.56 to 0.69 |
| Logistic head on MUSK 2048-d features, majority labels | 0.69 | 0.90 |
| Logistic head on MUSK 2048-d features, vote fraction | 0.74 | 0.90 |
| Final stacked model, majority labels | 0.73 (0.69 to 0.77) | 0.94 |
| Final stacked model, vote fraction | **0.82 (0.79 to 0.85)** | **0.94** |

- Vote-trained (soft) heads beat majority-trained (hard) heads on all four models, with both logistic and MLP heads.
- Label smoothing, training on unanimous cases only, and MLP heads with dropout did not reproduce the gain, so it comes from the vote information itself.
- Triage simulation: flagging the 20% most uncertain test cases with the final model catches 37% of disputed cases, 1.8 times random selection.

Disagreement AUROC asks whether model uncertainty separates contested tiles (2 to 5 SSL votes) from unanimous ones. Ranges are 95% bootstrap intervals. See [`mhist_disagreement/README.md`](mhist_disagreement/README.md) for every metric's definition.

## Repository layout

```
demo.py                          Gradio demo: second opinion, image test, language test
mhist_disagreement/
  README.md                      how to run the probes on Delta, metric definitions
  mhist.py, hosted.py            data loading, clients for the hosted model servers
  embed_server.py                image embeddings from the hosted /embed endpoints
  zero_shot.py, zero_shot_full.py, prompts_hp_ssl.json, prompt_sets*.json, prompt_ablation.py
                                 zero-shot evaluation and prompt comparison
  probe.py                       hard vs soft heads, ridge conflict head, metrics with CIs
  controls/                      label smoothing, unanimous-only and MLP controls,
                                 per-class metrics, calibration, image quality
  heads.py                       fits the heads used by the demo
  triage.py, agreement_bins.py   triage simulation and agreement-group analysis
  perturb.py, stain_robustness.py
                                 stain shift and blur robustness by agreement group
  pca_analysis.py, vote_heatmaps.py
                                 embedding PCA and vote heatmaps
  musk_full/                     2048-d multi-scale MUSK features on your own GPU
  musk_unsummarized/             per-layer MUSK tokens, attention heads, final stacked model
  results_{musk,conch,quiltnet,plip}/
                                 hard vs soft probes per model
  results_musklayers_multi/      final stacked model, soft and hard, with triage
  results_compare/, triage_results/
                                 cross-model triage and agreement groups
  results_stain/                 robustness results
  slurm/, tests/                 Delta job scripts and unit tests
```

## Running it

Everything ran on NCSA Delta with `module load pytorch-conda/2.8`. The zero-shot and embedding steps call model servers that were hosted only during the event. The MUSK feature extraction in `musk_full/` and `musk_unsummarized/` runs on your own GPU and needs Hugging Face access to MUSK. Exact commands are in [`mhist_disagreement/README.md`](mhist_disagreement/README.md), [`musk_full/README.md`](mhist_disagreement/musk_full/README.md) and [`musk_unsummarized/README.md`](mhist_disagreement/musk_unsummarized/README.md).

The demo runs without servers or data in mock mode (numbers are fake):

```
pip install gradio
python demo.py
```

## Data and license

MHIST is not included. Request it at https://bmirds.github.io/MHIST/ and agree to its research use terms. Per those terms, this repo contains no MHIST images, per-image predictions or embeddings (they are gitignored), and the demo must never be launched with a public share link.

## Limitations

All results come from one institution, 7 pathologists and 224-pixel tiles, and the MHIST paper does not state whether its split is by slide. The model predicts where pathologists are likely to disagree, not which diagnosis is correct, and the triage result is a retrospective simulation. In the final model, which layers to use was decided after looking at test scores, which may slightly inflate its results. This is a research prototype and is not for clinical use.

## References

- Wei J, et al. A Petri Dish for Histopathology Image Analysis (MHIST). arXiv:2101.12355, 2021.
- Lu MY, et al. A visual-language foundation model for computational pathology (CONCH). Nature Medicine, 2024.
- Xiang J, et al. A vision-language foundation model for precision oncology (MUSK). Nature, 2025.
- Huang Z, et al. A visual-language foundation model for pathology image analysis using medical Twitter (PLIP). Nature Medicine, 2023.
- Ikezogwo WO, et al. Quilt-1M: One million image-text pairs for histopathology (QuiltNet). NeurIPS, 2023.
