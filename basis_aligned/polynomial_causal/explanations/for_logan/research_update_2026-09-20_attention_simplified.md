# 20 September — Simplifying all of attention: every head valued by mean ablation, and what a program recovers relative to that

**Path now.** Every one of bilin18's 162 heads has a price (its CE cost under mean ablation) and a program. On the *pattern* side (the QK products that decide where a head reads) the whole model is now a fitted program — a positional kernel plus a few content directions per head, exact rank by construction — that costs +0.072 CE where deleting every head's variation costs 3.996 (recovery 0.982), preserves what each head is worth (rank correlation 0.84 with the native values, median scale 1.08), transfers to a never-used text window (0.078 there), and uses 3.7× fewer numbers than the native QK maps (25.9M vs 95.6M). On the *write* side (the values and c_proj that decide what a head says) heads use most of their 128 write directions: rank 32 for all heads costs +0.275; rank 64 costs +0.074 (recovery 0.981, values preserved 0.88 / 1.27) at 3.6× fewer numbers. Attention in this model is simple in where it reads and wide in what it writes. **Both sides together, exact rank, no native heads: +0.165 CE (recovery 0.957) at 39.2M numbers against 95.6M native (2.4× fewer); the two sides compose additively (0.072 + 0.077), further joint fitting does not help, and under the whole program the values of the 24 most valuable heads are preserved to median 1.26× with rank correlation 0.70 (v720).**

**Method, as you asked.** Value = CE added when the head's 128-d output z_h is replaced at every position by its fit-row mean (v701). Recovery of a program = 1 − cost(program) / value. Programs replace the off-diagonal pattern only; values, c_proj, the λ-mixed token branch and the diagonal stay native unless a write program is also installed. Fits use the 480+192 skip80/skip11000 rows split 576/96 with validation stopping; **every CE number below is on the fresh 192×512 skip7000 rows** (native 3.1324 replayed by every rung). Rungs v701–v715, all `ops/run_attention_*.py`, results in `circuits/followups/`, scored on the board.

## Claim table

| # | claim | tag | numbers | status |
|---|---|---|---|---|
| 1 | heads are cheap singly and dear jointly: median mean-ablation value 0.002, top 0.062 (0.3), 5.7 only 0.012; the 162 sum to 0.68 but all together cost 3.996 | edit | v701 | held |
| 2 | the sink head 5.7 is not the costliest head (ledger census 0.916 was a deletion, not a mean ablation): its mean *is* its function — a near-constant write that downstream blocks use as a bias | edit | 5.7 mean-ablated 0.012; projected off its mean +1.48 (v708) | held (v701/v708) — corrects my earlier "not a sink" |
| 3 | per-head ladder: kernel κ(d) < kernel + rank-4 / 16 / 64 content < native, chosen by recovery relative to value | edit | X=0.9: 129 kernels, 5 rank-16, 28 native → +0.507 (rec 0.873) | measured (v702) |
| 4 | fitting the kernels alone recovers 0.908; the content heads (2.5, 3.8, …) need rank ≥ 32 | fit | 0.507 → 0.369 | held (v703) |
| 5 | hybrids everywhere: kernel + rank-r content for all heads, no native heads | fit | 4/16/64 mix +0.124 (12.4M values, 7.7× fewer); 16/64 mix +0.075 (20.2M, rec 0.981) | held (v705/v706) |
| 6 | a program that reproduces the loss need not preserve what heads are worth: v706 leans 2–4× on 8.3, 5.5, 1.4, 3.5, 1.1 | edit | Spearman 0.65, median ratio 1.11; 8.3 4.0× | falsified bar (v706) |
| 7 | the leaning is not those heads' own programs (restoring them native leaves 8.3 at 4.4×) and not attributable to any single head's approximation (max residual 1.5% of the joint cost) | edit | v707 | held |
| 8 | the leaning is local: each over-weighted head is inflated by its own layer band's programs; joint cost is near-additive across bands, not across heads | edit | band 6-8 alone 3.2×; leave-6-8-native 1.36×; band sums 0.078 / 0.089 vs 0.084 | held (v709) |
| 9 | inside band 6-8 it is head 7.3's rank-16 program — a head worth 0.002 — that routes error through 8.3 and 5.5; rank 64 for the band fixes it closed-form | edit | 7.3 native: 3.21 → 2.13; band at 64: cost 0.084 → 0.061, 8.3 1.30× | held (v711) |
| 10 | **the fitted pattern program with band 6-8 at rank 64 passes all three gates** | fit | free-map fit +0.060 reported / 0.073 endpoint (v713); **factored, exact rank: +0.072, rec 0.982, Spearman 0.84, median ratio 1.08, max 1.70 (1.4); 25.9M values vs 95.6M (v718)** | held |
| 10a | the free-map fits of v705–v716 leaned on 0.1% off-rank energy worth 0.07 CE; the factored refit reaches the same cost without it — the numbers column is earned only from v718 on | fit | v713 as saved 0.073 → 0.142 when exactly truncated; factored 0.072 | held (v718) — instrument correction |
| 10b | the program transfers across text: 0.073 / 0.078 / 0.072 on skip7000 / a never-used 512-row window / skip1200; the per-head *values* do not (Spearman 0.64 across windows: 1.4 .012→.041, 3.5 .010→.001) but the program reproduces them on whichever window it is asked (0.89) | edit | v717 | held |
| 10c | more fitted values overfit the fit stream: bands 0-8 at 64 (34.5M) gives 0.047 on skip7000 but 0.088 on the fresh window, worse than v713's 0.078 | fit | v715/v717 | held — v713's configuration is the one to report |
| 11 | the write side in c_proj's singular basis is not low-rank at all; **centered** on the head's mean and in the head's own write-covariance basis, rank 32 recovers 0.85 singly (5.7: 0.89) | edit | v708 → v710 | held (v710) |
| 12 | closed-form write projections stack (0.124 singly → 0.666 jointly); fitted, rank 32 for all heads costs +0.275 and doubles every head's value; rank 64 costs +0.074 (rec 0.981, Spearman 0.88, median ratio 1.27; 13.3M vs 47.8M) with lr 1e-4 and snapshotting — at 3e-4 it overfits from step 75 | fit | v712/v714/v716 | held |
| 12a | pattern + write compose additively once both are measured at their validation minima (0.170 at step 0 vs 0.073 + 0.074); the earlier super-additivity was an endpoint-vs-snapshot instrument defect of mine (all fits before v716 ran their follow-ups on endpoint parameters) | fit | v716 | held — instrument correction |

## What this means

- **Mean-ablation value is the right reference, and it has a blind spot.** It prices a head's *variation*; a program that replaces the variation with a *wrong* variation can cost more than deleting it (7.3, claim 9; 5.7 under projection, claim 2). So "how much was recovered relative to the value" is meaningful per head only in the joint program, and a rank budget should be set by in-program leaning (claims 8–9), not by native value alone.
- **Manipulability is a checkable gate and it failed once for a reason we could localize.** The fix cost 5.7M numbers in one band; after it, what every head is worth is preserved to rank 0.82 / scale 1.05 (claim 10). The remaining 1.4 / 3.5 over-weighting is the same effect in bands 0-2 / 3-5 (v715 queued).
- **Read side simple, write side wide.** The pattern of a head is a kernel plus a few content directions (16 suffice for 111 heads, 64 for 51). Its write uses most of its 128 dimensions, and the heads compensate for each other's truncations (values double under the rank-32 write program). The honest simplification of attention is therefore asymmetric, and the numbers say where the budget goes: 26M of the 30M QK-side numbers can go; on the OV side we are still finding the knee.
- **Lessons re-confirmed:** single edits nominate, joint edits certify (claims 1, 7, 12); closed-form projections stack and fitted programs compose — on the pattern side (claim 5) but *not* across pattern and write (claim 12); validation stopping is necessary (every fit's held-out turns up after step 100).

## Price ladder (pattern side; fresh CE added; native QK = 95.6M numbers)

| program | values | cost | recovery | values preserved? |
|---|---|---|---|---|
| all 162 kernels, closed-form | 83k | +1.45 | 0.64 | — |
| all 162 kernels, fitted | 83k | +0.96 | 0.76 | — |
| kernels + 28 native content heads | 28 × 590k + 69k | +0.37 | 0.908 | — |
| 28 native + rank-16 hybrids | 27.6M | +0.065 | 0.984 | not checked |
| no native, 4/16/64 mix | 12.4M | +0.124 | 0.969 | not checked |
| no native, 16/64 mix (v706) | 20.2M | +0.075 | 0.981 | no (Spearman 0.65) |
| no native, 16/64 + band 6-8 at 64, free maps (v713) | (25.9M nominal; not exact) | +0.060 reported / 0.073 endpoint | 0.985 | yes (0.82 / 1.05) |
| **same configuration, factored exact-rank maps (v718)** | **25.9M exact** | **+0.072** | **0.982** | **yes (0.84 / 1.08)** |
| write side, rank 64 all heads (v716) | 13.3M vs 47.8M | +0.074 | 0.981 | yes (0.88 / 1.27) |
| **both sides, exact rank (v720)** | **39.2M vs 95.6M** | **+0.165** | **0.957** | **partly (0.70 / 1.26)** |

## Addendum (19:30 UTC) — what the program is made of, and what it is not (v721–v727)

| # | claim | tag | numbers | status |
|---|---|---|---|---|
| 13 | **every positional kernel is a combination of four shapes**: projecting all 162 kernels onto the top-4 SVD basis of the 162×512 kernel matrix costs +0.001 over the program; k = 1 costs +0.40, k = 2 +0.046, k = 3 +0.046 | edit | v724/v725 | held — the positional side is 2.7k numbers |
| 13a | the shapes: a decaying window (~1/d over d ≤ 16); "not the previous token, the medium range" (−.53 at d = 1, plateau 4–16, slow tail); a weak mid-range subtraction; "skip the previous, read 3–8, subtract the far context". The previous-token taps of blocks 0–2 are window minus plateau | fold | v725 (values by offset in the result JSON) | measured |
| 14 | the fitted content directions are CE-equivalents, not copies of the native pattern: on exact single-token tables (layers 0–1, 1024-token grid, d = 1) they correlate 0.64–0.95 with the native tables and carry systematically *less* query×key interaction (0.0: .42 vs .59; 0.1: .24 vs .58) — the fit kept the separable gating and dropped the bigram term wherever the loss allowed | fold | v721 | held; no token readings claimed (v660/v661 lesson) |
| 14a | head 1.4 is a duplicate-token head: interaction share 0.79 on the exact tables with identical-token top pairs (fold), and the one-number edit κ(d) + 0.05·[tok_i = tok_j] recovers 0.70 of its value — exactly what its closed-form rank-16 program (73k numbers) recovers; |β| ≥ 0.25 or β < 0 hurt | fold + edit | v721/v728/v729 | held — first content reading of the chapter confirmed by an edit |
| 15 | the write side is wide in *energy*, not only in importance: median 90%-energy rank of a head's centered write is 58 of 128 (0.3: 83, 6.3: 85, 1.1: 15), and energy predicts the CE rank a head needs (Spearman 0.76) — unlike the token metric on the MLP side, the head's own write covariance is an honest guide | fold | v722 | held — the OV side has no hidden low rank |
| 16 | there is no shared read subspace: the 11.6k content directions of all heads spread over the whole residual stream (top-256 of 1152 carry 40% of their energy; projecting reads onto them costs +1.00) | edit | v726 | held |
| 17 | the weights-only writer→reader coupling graph (reader's content directions inside the writer's write subspace) is flat: median 1.06× chance, 3 of 12,393 pairs ≥ 4×, no decay with layer distance; 5.5's strongest writers are the layer-4 heads 4.3/4.5/4.7 (2–3×), 8.3's is 6.3 — not 7.3 | fold | v727 | held — the circuit skeleton is not in pairwise weights |

**So the honest description of a head in this model is per head:** *read these r directions of the residual stream (r = 16 for 111 heads, 64 for 51), weigh keys by one of four positional shapes (a 4-vector of coefficients) plus a rank-r bilinear content term that is mostly separable gating, write 64 of your 128 directions.* There is no shared coordinate system across heads on either side, and the pairwise weight overlaps do not expose the circuit — that still needs activations (the census and path-patching of the earlier lanes).
| 19 | the sink 5.7 is κ(d) plus one number on the position-0 column: c = 0.50–0.55 recovers 0.53 of its value (kernel-only −6.2, closed-form rank-16 −5.4; c = 0.35 or 0.70 give 0.15) — an absolute-position read with a sharply tuned weight, half of the head; the rest is its off-sink content | edit | v732/v733 | held at half strength |
| 18 | one-number content programs are rare: of the 38 valuable heads only 1.4 (same-token, recovery 0.70) and 5.5 (induction indicator [tok_i = tok_{j−1}], 0.40 at β = 0.1; its rank-16 term gets 0.54 — the rest of the induction read is graded similarity) gain from an indicator; the sink 5.7 needs an absolute-position read that no offset kernel gives | edit | v730/v731 | held |
| 20 | the three one-number readings hold inside the joint program: each costs about its standalone cost (1.4 +0.006, 5.5 +0.007, 5.7 +0.005 over the 0.072 program), they add (+0.021 vs 0.018), and the program with all three costs +0.093 (recovery 0.977); inside it the partial readings are worth less than the heads (5.5 0.46×, 5.7 0.72×) | edit (joint) | v734 | held — certified |
| 21 | the twelve block-0/1 readings of the embedding-forward report (eleven gated filters κ(d)A(t_i)B(t_j), 1.8 = −1/i) hold inside the joint program: +0.037 over the 0.072 program (0.036 standalone); with the three one-number heads, **fifteen readable heads + 147 kernel-and-rank heads cost +0.142 (recovery 0.964)** | edit (joint) | v735 | held |
| 22 | readings do not stack in depth without a joint fit: the five separable layer-2 gates (v628) cost +0.042 on the base program (sum of singles 0.032) but +0.091 on top of the fifteen — their tables were fitted with native layer-0/1 inputs; twenty readable heads +0.233 (recovery 0.942) | edit (joint) | v736 | falsified as stacked; the readable frontier stands at fifteen heads (+0.142) |
| 23 | readings stack in depth once the deeper tables are refitted with the shallower readings installed: the five layer-2 gates refit in the context of the fifteen cost +0.026 (from +0.091), with tables that barely move (corr 1.000); **twenty readable heads + 142 kernel-and-rank heads = +0.168 (recovery 0.958)** | fit (joint, snapshot) | v737 | held |

**Vocabulary of the atlas, as it stands:** kernel + separable gate for most heads; kernel + one indicator for 1.4 and (partly) 5.5; kernel + rank-64 content for the ~28 content heads; an absolute-position read for 5.7; 64 write directions for everyone.

## Open

- Done: v720 — both sides exact-rank: +0.165 at step 0, +0.172 at the validation-chosen step (39.2M numbers). The whole program leans 1.5–2× on 1.3 / 1.4 / 1.1 / 3.5 / 8.3.
- Canary `ops/run_attention_canary_v719.py` replays the registered numbers (native, joint value, v713, v716 exact-rank, the fresh window): 5/5 on first run.
- Not yet done: *reading* the fitted content directions (what the 16–64 directions of the content heads are — the induction key of 5.5 is known; the rest are not), and whether the write directions align with the MLP readers downstream. The kernels of blocks 0–2 were read in the embedding-forward report; the deeper kernels (layer 5 content-critical, layers 6-8 mostly positional) are described only by shape.
