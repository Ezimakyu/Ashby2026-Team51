# MHIST disagreement probe (prototype)

Quick test of Option 2: can frozen pathology-model embeddings (MUSK now,
CONCH/QuiltNet/PLIP for comparison) tell hyperplastic polyps (HP) from sessile serrated
lesions (SSA/SSL), and can they tell which patches the 7 MHIST pathologists
disagreed on?

The output is a table of numbers per method. It doesn't decide whether
they're good enough; that's the team's call.

## What gets compared

| Method | Trained on | Gives |
|---|---|---|
| `baseline_constant` | nothing (train-set SSA rate) | reference row: what "no information" scores |
| `zero_shot` | nothing, just HP/SSA text prompts | p(SSA) |
| `logreg_hard`, `svm_hard` | majority label only | p(SSA) |
| `logreg_soft`, `knn_soft`, `mlp_soft` | vote fraction k/7 | p(SSA) |
| `ridge_conflict` | conflict = min(k, 7−k) directly | conflict score only |

For the p(SSA) methods, "how contested" = how close p is to 0.5.

## Metrics (test split only, 95% bootstrap intervals in `summary.csv`)

| Column | Meaning | No-skill value |
|---|---|---|
| `label_auroc` | HP vs SSA ranking vs the majority label | 0.5 |
| `label_acc` / `label_balanced_acc` | accuracy at p = 0.5 | `baseline_constant`'s / 0.5 |
| `vote_spearman` | p(SSA) vs fraction of pathologists voting SSA | 0 |
| `vote_mae` | mean error in votes (0–7 scale) | `baseline_constant`'s |
| `binomial_nll` | fit to the 7 votes; lower is better (zero-shot isn't calibrated, ignore it there) | `baseline_constant`'s |
| `disagree_spearman` | predicted contestedness vs actual conflict (0–3) | 0 |
| `disagree_auroc` | contested (2–5 SSA votes) vs unanimous (0 or 7) | 0.5 |
| `disagree_auroc_3v4` | 3–4 splits vs unanimous | 0.5 |

`metrics.json` also has the test-set vote histogram and the chosen
hyperparameters. Caveat for the write-up: we haven't checked whether MHIST's
official split keeps each patient on one side.

## Running it on Delta

One-time, on the login node:

```bash
git clone https://github.com/Ezimakyu/Ashby2026-Team51.git
cd Ashby2026-Team51/mhist_disagreement
module load pytorch-conda/2.8
export PATHOLOGY_ENDPOINT=http://<host>:<port>    # MUSK server, from #compute-issues
export PATHOLOGY_API_KEY=<key>
```

1. **Embed** (about a minute, uses the hosted server, no GPU of yours):
   ```bash
   python embed_server.py --images <MHIST images dir> --out embeddings/musk.csv
   ```
2. **Zero-shot baseline** (optional, a few minutes):
   ```bash
   python zero_shot.py --images <MHIST images dir> --out embeddings/musk_zeroshot.csv
   ```
3. **Train and evaluate** (CPU job, a few minutes):
   ```bash
   sbatch slurm/probe_cpu.sbatch <path>/annotations.csv musk=embeddings/musk.csv \
       --zero-shot musk=embeddings/musk_zeroshot.csv
   ```
   The table is printed at the end of `logs/<jobid>.out`. Files go to `results/musk/`.
4. **Share:** `git add results && git commit && git push`. Per-image predictions,
   embeddings and data are gitignored (MHIST terms); metrics and plots are pushed.

Other models: rerun steps 1–2 against another server with a new tag
(`embeddings/quiltnet.csv`), then pass several `tag=path` arguments to step 3
to get one combined `results/summary.csv`.

If the hosted server is down (or after Sun 6 pm), `slurm/embed_gpu.sbatch`
embeds on one GPU with the ncsa repo's `run_embeddings.py`. It writes the same
CSV format, but uses your own GPU-hours.

If the annotations columns aren't detected, `probe.py` prints the actual
column names; pass them with `--votes-col`, `--split-col`, `--image-col`, `--majority-col`.

## Files

- `mhist.py`: loads annotations.csv, derives votes/p_ssa/conflict, joins to embeddings
- `hosted.py`: shared HTTP helpers for the hosted servers
- `embed_server.py`, `zero_shot.py`: call `/embed` and `/classify`
- `probe.py`: all models, metrics, plots
- `prompts_hp_ssl.json`: zero-shot prompt pairs (edit freely)
- `tests/`: synthetic-data checks (`python -m pytest tests/`)
