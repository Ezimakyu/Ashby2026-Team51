# Triage summary

How to read `triage_curve.png`:
- **Top row:** x = share of test images sent for a second read, most uncertain first. y = share of all contested images (2-5 of 7 SSA votes) that ended up flagged. The dashed diagonal is random flagging. The further a curve sits above it, the better that uncertainty score finds the cases pathologists disputed.
- **Bottom row:** accuracy of the HP/SSA call (logreg_soft) on the images that were *not* flagged, vs the majority label. The dashed line is accuracy with no flagging. A rising curve means flagging removes the cases the model gets wrong.
- "Contested caught" counts disputed cases, not model errors. A score can be good at one and not the other: the conflict head finds disputed cases well but barely raises accuracy on the rest.

## musk

Uncertainty from the conflict head (ridge_conflict):
- Flagging the 10% most uncertain cases catches 19% of contested cases (1.9x random flagging, which catches 10%); accuracy on the rest is 81% (vs 80% with no flagging).
- Flagging the 20% most uncertain cases catches 34% of contested cases (1.7x random flagging, which catches 20%); accuracy on the rest is 82% (vs 80% with no flagging).
- Flagging the 30% most uncertain cases catches 46% of contested cases (1.5x random flagging, which catches 30%); accuracy on the rest is 84% (vs 80% with no flagging).

Uncertainty from the soft head (logreg_soft, closeness of P(SSA) to 0.5):
- Flagging the 10% most uncertain cases catches 15% of contested cases (1.5x random flagging, which catches 10%); accuracy on the rest is 82% (vs 80% with no flagging).
- Flagging the 20% most uncertain cases catches 27% of contested cases (1.4x random flagging, which catches 20%); accuracy on the rest is 86% (vs 80% with no flagging).
- Flagging the 30% most uncertain cases catches 43% of contested cases (1.4x random flagging, which catches 30%); accuracy on the rest is 89% (vs 80% with no flagging).

Uncertainty from the hard head (logreg_hard, closeness of P(SSA) to 0.5):
- Flagging the 10% most uncertain cases catches 13% of contested cases (1.3x random flagging, which catches 10%); accuracy on the rest is 83% (vs 80% with no flagging).
- Flagging the 20% most uncertain cases catches 28% of contested cases (1.4x random flagging, which catches 20%); accuracy on the rest is 85% (vs 80% with no flagging).
- Flagging the 30% most uncertain cases catches 42% of contested cases (1.4x random flagging, which catches 30%); accuracy on the rest is 90% (vs 80% with no flagging).

## conch

Uncertainty from the conflict head (ridge_conflict):
- Flagging the 10% most uncertain cases catches 17% of contested cases (1.7x random flagging, which catches 10%); accuracy on the rest is 81% (vs 80% with no flagging).
- Flagging the 20% most uncertain cases catches 29% of contested cases (1.4x random flagging, which catches 20%); accuracy on the rest is 82% (vs 80% with no flagging).
- Flagging the 30% most uncertain cases catches 43% of contested cases (1.4x random flagging, which catches 30%); accuracy on the rest is 83% (vs 80% with no flagging).

Uncertainty from the soft head (logreg_soft, closeness of P(SSA) to 0.5):
- Flagging the 10% most uncertain cases catches 16% of contested cases (1.6x random flagging, which catches 10%); accuracy on the rest is 83% (vs 80% with no flagging).
- Flagging the 20% most uncertain cases catches 32% of contested cases (1.6x random flagging, which catches 20%); accuracy on the rest is 87% (vs 80% with no flagging).
- Flagging the 30% most uncertain cases catches 47% of contested cases (1.6x random flagging, which catches 30%); accuracy on the rest is 89% (vs 80% with no flagging).

Uncertainty from the hard head (logreg_hard, closeness of P(SSA) to 0.5):
- Flagging the 10% most uncertain cases catches 16% of contested cases (1.6x random flagging, which catches 10%); accuracy on the rest is 83% (vs 80% with no flagging).
- Flagging the 20% most uncertain cases catches 31% of contested cases (1.5x random flagging, which catches 20%); accuracy on the rest is 87% (vs 80% with no flagging).
- Flagging the 30% most uncertain cases catches 45% of contested cases (1.5x random flagging, which catches 30%); accuracy on the rest is 89% (vs 80% with no flagging).
