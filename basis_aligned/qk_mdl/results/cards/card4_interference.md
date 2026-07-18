# Circuit card 4: the two-head interference, dissected

**Why does killing BOTH heads hurt less than killing H7 alone?** (card 1's
anomaly, taken to corpus scale on the repeat harness; baseline CE 5.485 on second halves of 16 random A+A sequences)

| arm | ΔCE (repeat 2nd half) |
|---|---|
| live | +0.0000 |
| H7 dead | +6.6814 |
| H7 + H5 dead | +7.1483 |
| H7 dead + H5 CLEANED | +7.0724 |
| H7 + H0 dead (control) | +6.2473 |
| H7 + H3 dead (control) | +6.6097 |
| H5 dead alone (ref) | +0.1302 |

## Verdict

**The card-1 anomaly does not generalize.** At corpus scale, co-ablating H5 with H7
is ordinarily additive-harmful (+7.15 vs +6.68), and cleaning H5 instead of removing
it is no better (+7.07). A mild version of the interference appears with a DIFFERENT
head here (H7+H0 = +6.25 < H7 alone). Conclusion: two-head ablation interactions are
real but context-idiosyncratic — card 1 measured one prompt one target; no stable
compensation mechanism exists to name. Methodological takeaway (now twice over, cf.
card 2): single-prompt card findings are hypotheses; the card→scale-test pipeline is
the unit of evidence.