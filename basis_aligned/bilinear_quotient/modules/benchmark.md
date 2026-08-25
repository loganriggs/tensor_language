# The understanding benchmark — per-module scores, valid stand-ins, measurement rules

Scale: 0 = mean-ablate, 1 = full model. The VALID measure is PER-MODULE-IN-ISOLATION —
per-module understanding does NOT compose (whole-model greedy substitution 12%,
compounding-dominated; content passthrough 0.39; §1070-1071). Whole-model simultaneous held-out:
**0.32 ± 0.06** (draw-sensitive; §1013-1014).

## Current per-module scores + stand-ins
| Module | Score | Valid stand-in | Ref |
|---|---|---|---|
| attn L0 (all heads) | ~1.00 | token+position pattern + per-token value table | sink arc |
| head 5.7 | ~0.985 | ONE fixed vector (global mean) | §1089/§1091 |
| other front routers L0H3/L1H1/L2H5 | 0.7-0.95 | residual-window routing | §1054/§1091 |
| mlp0 | ~0.90 | class-code writer / token map | §905/§1045 |
| mlp1 | ~0.93 | static per-token table (held-out) | §1088 |
| mlp3 | **0.95 (r256) / 0.83 (r64)** certified | own-basis projection (held-out) | §1130 |
| mlp4 | 0.82 (r256) / 0.48 (r64) certified | own-basis projection; token table HURTS held-out. WIRING (S1427, exact ledger; S1422 RETRACTED): marginal diet = attn4 .053 + mlp3 .032, all else <.01 — local chain attn_L + mlp_{L-1} -> mlp_L; token/topic-blind (S1423); ridge ladder (S1424/28): lin2 [attn4,mlp3] .617 / lin3 [front MLPs] .612 / lin5 [all] .679 held-out; ~32% quadratic residual open | §1130, S1422-23 |
| mlp5-14 | ~0.10 as variables | NO TESTED FORM WORKS YET (token table / low-rank / linear read) — NOT a claim of no structure; "content" is a residual label. Open program (user directive 2026-08-25): input decomposition vs understood components + semantic (BoW-topic) conditioned tables + sectional decomposition. Starts at mlp4: mlp4_reads | §1000/38/42, S1421+ |
| middle attn (collective) | 0.58 partial | static distance-kernel (values dynamic); kernel+content-sim next | §1099 |
| readout mlp16/17 | **0.81/0.84 certified** (mlp15 0.40, tiny stakes) | fitted linear read; CEILING: top-256 own-neurons also 0.81 — functional tail is neither sparse nor linear | §1131-1132 |
| block-17 calibration | ~1.00 | rank-1 w_freq | §650-651 |
| the merge | understood | additive-linear (−W·c, cos 0.77) | §1082/§1086 |

## Anchors (updated 2026-08-25, user spec)
Floor anchor moving from MEAN-ablation to OPTIMAL-ablation (Li & Janson 2409.09951):
learned constant vector per component, trained against full-model CE, mean-init.
fidelity(repl) = (d_opt - d_repl)/d_opt — optimal constant scores 0, component scores 1.
Per-position constants = a higher complexity budget, logged separately. Constants:
opt_ablation_consts.pt (optimal_ablation.py; first four: mlp4, mlp1, mlp16, head 13.8).
Prior "recovery of ymean-gap / mean-stake" numbers remain valid but are MEAN-anchored;
re-anchor when comparing across papers.

## Measurement rules (violations produced retractions — check every design against these)
1. HELD-OUT everything: in-sample per-token means leak on singletons (~20% of positions)
   (§901 0.81→0.30; §1088 deep tok-recovery 0.59→0.01-0.14).
2. Matched nulls: keep-only needs shuffled-label matched-rank nulls (§836); random-subspace is
   far too weak (§821).
3. Perturbation-size controls: norm-match any removal before claiming direction-specificity
   (§1086); low K for specificity (destructive regime at K=256; §1067-1068).
4. Partial ≠ share: partial removals of baselines/constants cost MORE than full removal
   (§1087/§1091/§1093) — never read fractions off partials.
5. Per-part sums ≠ collective: measure both granularities (5.7× gap, §1093; middle band 4×,
   §813; front-6 §952).
6. Stand-in must match the removal point/scale (§1066).
7. Banding vs output-ablation flip on redundant parts (§1008-1009); firing ≠ function (§726).
8. Draw-sensitivity: report understanding as a band (±0.06), not a point (§1013-1014).

## The target
90% per module bottom-up (user directive). Solved: front, readout, sink, L0-attention,
calibration. Bounded by real model properties: deep-middle content (high-rank). Genuinely open:
mlp4's variable, the middle-attention collective pool.

## The module simplicity ladder (§1322-1327) — stake / table ceiling / elbow per MLP

A second, cheaper axis than the understanding score: for each MLP, mean-ablation STAKE
(nats), token-table CEILING (recovery of the stake by a per-token mean table, held-out),
and the ELBOW index k16/ceiling (how much of the table 16 k-means categories buy).
Instrument: mlp*_clusters.py / mlp_ladder_depth.py. **The instrument is uninformative below
~0.15 nats of stake** (negative ceilings = table estimation noise; §1326). One k-means
seed, one eval draw — treat third decimals as noise.

| Module | Stake | Ceiling | k16/ceiling | Reading |
|---|---|---|---|---|
| mlp0 | 0.80 | 0.863 | 0.43 | token-resolved, log-linear (§1324) |
| mlp1 | 7.00 | 0.945 | 0.43 | token-resolved, log-linear; THE big front module (§1322-23) |
| mlp2 | 0.76 | 0.716 | 0.14 | token-resolved (§1326) |
| mlp3 | 0.63 | 0.593 | 0.40 | token-resolved (§1326) |
| mlp4-15 | 0.03-0.10 | n/a | n/a | UNEVALUABLE at this instrument's floor (§1326) |
| mlp16 | 0.15 | 0.494 | **1.10** | categorical: K=16 BEATS the 50k table (§1326) |
| mlp17 | 0.38 | 0.497 | 0.84 | half-contextual + categorical elbow at K=16 (§1325) |

Context arms (§1327-1330): a (token16 x context16) key beats the 50k table at the top —
and ONLY the within-token null (resample labels from P(ctx|token)) separates context from
token re-encoding; the random-label null does NOT (§1327). Purity is a powerless
diagnostic (retired); use NMI(ctx;tok)/H(ctx). **The §1328 ceiling-bound gate
(increment <= 1 - ceiling) was RETRACTED in §1330**: the mean table is L2-optimal, not
CE-optimal, so 1 - ceiling lower-bounds token-only inadequacy but does NOT upper-bound
context value. Data-scaling arbitrates instead: artifacts shrink with fit data, signal
grows (mlp1's increment grew 0.092->0.142 as sparse-token mass halved). Standing picture:
mlp17/16 contextual half real and fast (not doc-state at K=16 grain, §1328); mlp1 = a
context-indexed per-token MENU (~1.0 nats of within-token choice at 4-bit grain, §1330).
Instrument draw-spread ~+-0.05 per point — trends, not points, are evidence.
**FINAL STATE after §1332 (read this, not the §1328/§1330 intermediate reversals):** the
standing instrument for "does context add anything" is the RESIDUAL construction — fit
the ctx delta on (out - full-token-table[token]) and score on top of the table; every
coarse-grain label-vs-null comparison is confounded on token-dominated modules by the
CONSISTENT-HASH effect (a deterministic label routes rare tokens to the same cell at fit
and eval; the null scatters them — an advantage that GROWS with data, so §1330's
growth criterion cannot catch it). Verdicts: mlp1 ~95% token table, un-tableable residue
0.045 of stake, ctx16 reaches +0.0066 of it ("menu" withdrawn); mlp17 +0.1328 ON TOP of
its full table (null -0.001, 42% of residual variance) — real, fast context at the top.

## ONE-MODEL BILL (S1403)
28/162 heads live (22 commons ungated + 6 specialists), all other heads v1-routed with
meaned values, MLPs live: question .644 / comparative .918 / closer .846 / capitalized
.826 / elsewhere .812 recovery of ymean-gap. Co-residence synergistic (+.03-.04 for
closer/capitalized over solo kits). Question is the only gated family (gate = functional
state protection, S1402). Slim (S1405): 20 heads at ~.01-.03 haircut; 13.8 droppable
(construction-redundant). FINAL BILL (S1408; FRESH-CERTIFIED S1421): 28/162 heads + 11/18 MLPs live
({4,6,7,8,9,11,13} mid + 0-3 + 16-17) + gmean tables for mlp5,10,12,14,15 = comparative
.998 fresh / closer .859 / capitalized .799 / question .695 / else .751 — reproduced on
disjoint rows within .006 per line. Three families beat all-mids-live (width-inv #10).
20-head variant at ~.02 haircut (S1405). Newline served at .863 with no newline head (S1417).
