# Three-hourly mathematical review (Claude lane) — 2026-09-01 22:10 UTC

Grounding: ledger §2544–§2556 + rungs 439/439b; user redirect (21:52) to structural sparse/continuous
decompositions and the sealed learned-simplicity protocol; quantization closed as non-interpretability.

## Executed analysis: spectral-truncation certificates for the 36 normalizer quadratics (CPU, exact)

Object: the per-head PSD forms A_j = W_j^T W_j/128 (W_j the [128,1152] layer-0 Q/K head maps) whose token
quadratics are the causally necessary denominators (433b: constants cost +.32122 nat; CE added above native —
lower is better). Theorem used: for RMS-normalized x (||x||² = D = 1152), the rank-r truncation error obeys
|x^T(A−A_r)x| ≤ λ_(r+1)·1152 (spectral bound), composing through the 1/√ Lipschitz factor to a certified
score-error bound. Computed all 36 exact spectra from the checkpoint (float64 SVD of W_j; rank ≤ 128 exactly).

RESULT — truncation is certifiably hopeless, and provably so before any GPU spend:
- energy ranks: r90 median 100 (min 76, max 106) of 128; r99 median 124 (111–126) — near-flat spectra;
- worst-case r=64 truncation bound λ_65·1152 is **8.18× the entire trace**
  (min 5.24×, max 8.64×); even r=96
  leaves 5.65× trace worst-case;
- top eigenvalue carries only 4.4% median energy (one outlier 29.4%).

CONSEQUENCE (theorem-backed pricing): no per-head low-rank surrogate of the normalizers can carry a nontrivial
worst-case certificate; the certified representations are exactly the native maps (5,308,416 scalars) or the
enumerated token table (1,809,252 scalars, vocabulary-closed). This completes the normalizer story begun by
433b/435 with a *certificate-level* closure: generic in alignment (435), generic in spectrum (here). Artifact:
scratchpad normalizer_spectra.json (36 rows; also computed c_v/c_proj slices, same flatness — 54 total).

## Ranked mathematical moves
1. (executed above) Spectral truncation certificates — closed negatively with proof-grade numbers.
2. Minimal-realization certificate for 424's 6/6 score quotient: Kalman/Hankel view — the exact edge-behavior
   matrix's singular gap at rank 6 would certify the block ranks minimal (or license rank-5). Cheap CPU screen
   on stored factor machinery; measurable consequence: a lower bound, not a fit. Proposed to direction authority.
3. Response-metric-anchored archetypes: 439 measured identifiability(+63%)-vs-computation(13×) tension; theory
   says the hull was built in raw token geometry while the computation lives in whitened/response geometry —
   an anchor in 424's factor metric is the mathematically indicated retry. Codex's lane.

Pruned: common congruence/INDSCAL (closed 435 at control level); quantization factorials (user: not
interpretability); global sparse atom semantics (430: restart-unstable); further covariance-metric engineering
(five falsifications closed it).
