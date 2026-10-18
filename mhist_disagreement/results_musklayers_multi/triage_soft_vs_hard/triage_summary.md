# Triage summary

How to read `triage_curve.png`:
- **Top row:** x = share of test images sent for a second read, most uncertain first. y = share of all contested images (2-5 of 7 SSA votes) that ended up flagged. The dashed diagonal is random flagging. The further a curve sits above it, the better that uncertainty score finds the cases pathologists disputed.
- **Bottom row:** accuracy of the HP/SSA call (logreg_soft (or the @head given per model)) on the images that were *not* flagged, vs the majority label. The dashed line is accuracy with no flagging. A rising curve means flagging removes the cases the model gets wrong.
- "Contested caught" counts disputed cases, not model errors. A score can be good at one and not the other: the conflict head finds disputed cases well but barely raises accuracy on the rest.

## final_soft

Uncertainty from stack:
- Flagging the 10% most uncertain cases catches 18% of contested cases (1.8x random flagging, which catches 10%); accuracy on the rest is 91% (vs 88% with no flagging).
- Flagging the 20% most uncertain cases catches 37% of contested cases (1.8x random flagging, which catches 20%); accuracy on the rest is 93% (vs 88% with no flagging).
- Flagging the 30% most uncertain cases catches 51% of contested cases (1.7x random flagging, which catches 30%); accuracy on the rest is 95% (vs 88% with no flagging).

## final_hard

Uncertainty from stack:
- Flagging the 10% most uncertain cases catches 17% of contested cases (1.7x random flagging, which catches 10%); accuracy on the rest is 91% (vs 88% with no flagging).
- Flagging the 20% most uncertain cases catches 32% of contested cases (1.6x random flagging, which catches 20%); accuracy on the rest is 94% (vs 88% with no flagging).
- Flagging the 30% most uncertain cases catches 43% of contested cases (1.4x random flagging, which catches 30%); accuracy on the rest is 95% (vs 88% with no flagging).
