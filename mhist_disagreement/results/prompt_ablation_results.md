# Prompt ablation (Yvette, Sat 10/17)

Zero-shot /classify on 100 unanimous MHIST images (50 with 0/7 SSA votes, 50 with 7/7), random.seed(0).
Metric: AUC (server scores are compressed near 0.5, so threshold accuracy is not meaningful).

| Prompt set | CONCH |
|---|---|
| 1_name_morph | 0.818 |
| 3_morph_only | 0.712 |
| 5_report_style | 0.706 |
| 6_observed (3-prompt avg) | 0.696 |
| 4_new_name_SSL | 0.689 |
| 2_name_only | 0.666 |

Takeaway: adding morphology to the diagnosis name raises AUC from 0.67 to 0.82.
Caveat: n = 100 easy cases; small gaps may be noise.
TODO: QuiltNet, PLIP, MUSK columns.
