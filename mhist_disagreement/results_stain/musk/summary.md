# Stain robustness (MUSK + soft head)

299 test images; perturbations: 3 stain shifts (strength 0.2, seeds 0, 1, 2) and 1 blur (radius 1.5 px).
A flip = the HP/SSA call (p(SSA) >= 0.5) differs from the unperturbed image's.

| agreement | n_images | flip_any | flip_rate | flip_stain | flip_blur | mean_dp | mean_dlogit | median_margin |
|---|---|---|---|---|---|---|---|---|
| 7/7 | 99 | 0.192 | 0.053 | 0.013 | 0.172 | 0.052 | 0.347 | 0.302 |
| 6/7 | 93 | 0.258 | 0.086 | 0.036 | 0.237 | 0.061 | 0.353 | 0.237 |
| 5/7 | 57 | 0.333 | 0.127 | 0.076 | 0.281 | 0.077 | 0.400 | 0.143 |
| 4/7 | 50 | 0.400 | 0.115 | 0.053 | 0.300 | 0.081 | 0.396 | 0.147 |
| all | 299 | 0.274 | 0.088 | 0.039 | 0.234 | 0.065 | 0.367 | 0.216 |

Trend, conflict (0 = 7/7 … 3 = 4/7) vs number of flips per image: Spearman 0.172 (p = 0.0029).
Trend, conflict vs mean |change in p(SSA)|: Spearman 0.294 (p = 2.2e-07).
Trend, conflict vs mean |change in logit p(SSA)|: Spearman 0.123 (p = 0.034).

Caveat: contested images have p(SSA) nearer 0.5 (see median_margin), so the same
shift is more likely to cross the threshold, and mean_dp is inflated there too because
the sigmoid is steepest at 0.5. mean_dlogit removes both effects: how much it rises
with conflict is how much more the contested images' embeddings actually move.
95% intervals are in summary.csv (Wilson for flip_any, bootstrap over images otherwise).
