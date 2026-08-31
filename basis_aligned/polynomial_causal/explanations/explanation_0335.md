# Explanation — 2026-08-31 03:35 UTC: the twins give way — anatomy, a working merge, and the corrected knockout numbers

Convention (§2135): all numbers are CE added above the real model — lower is better. Continues explanation_0100.

## What changed in the last two hours (§2184–§2186, user-driven)

The user proposed merging the replicated attention pairs on their outputs. Three versions were separated and
tested:
- A fixed output-to-output map: already dead (document-gauge, §2181).
- **Drop one twin + a scalar gain on the survivor: WORKS.** α = 1.45 on attn2 recovers 55% of deleted attn3
  (+0.237 → +0.107). A scalar carries no coordinates, so the gauge cannot touch it — and the survivor already
  computes the whole function, just ~1.45× too quietly. First held construction since the constructive pause.
- A distilled single block-2/3 unit: now the licensed follow-up (see anatomy below).

**Twin anatomy (§2185/§2186):** 73% of the old knockout damage was just the missing MEAN vector, not signal;
the survivor shows no directed compensation (its change is downstream sensitivity, cosine ≈ 0 toward the lost
signal); the twins write DIFFERENT vectors (per-position cos ~0.22, position-independent) — so "twinness" is
functional equivalence in document-gauged coordinates, not duplicated writes; and attn2's per-position value
flows 84% through block 3 (attn3+m3), only 26% via the residual stream. The right mental picture is a
two-stage local circuit whose stages are individually sufficient — which is why one module plus one number can
stand in for the pair.

## Running now
Rung 94: the mean-ablation knockout suite for all three duos (does the backup super-additivity survive the DC
correction, or was it partly mean-term compounding?). Rung 95: the scalar merge at 14/16 and b4/b5 — if it
holds, each duo compresses to one module + one scalar, three numbers replacing three modules across the
architecture.
