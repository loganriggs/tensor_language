# 20 September — Simplifying all of attention: every head valued by mean ablation, and what a program recovers relative to that

**Path now.** Every one of bilin18's 162 heads has a price (its CE cost under mean ablation) and a program. On the *pattern* side (the QK products that decide where a head reads) the whole model is now a fitted program — a positional kernel plus a few content directions per head — that costs +0.060 CE where deleting every head's variation costs 3.996 (recovery 0.985), preserves what each head is worth (rank correlation 0.82 with the native values, median scale 1.05), and uses 3.7× fewer numbers than the native QK maps. On the *write* side (the values and c_proj that decide what a head says) the picture is the opposite: heads use most of their 128 write directions, and truncating them jointly is expensive (rank 32: +0.275; rank 64 pending, v714). Attention in this model is simple in where it reads and wide in what it writes.

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
| 10 | **the fitted pattern program with band 6-8 at rank 64 passes all three gates** | fit | +0.060, rec 0.985, Spearman 0.82, median ratio 1.05, max 1.81 (1.4); 25.9M values vs 95.6M | held (v713) |
| 11 | the write side in c_proj's singular basis is not low-rank at all; **centered** on the head's mean and in the head's own write-covariance basis, rank 32 recovers 0.85 singly (5.7: 0.89) | edit | v708 → v710 | held (v710) |
| 12 | closed-form write projections stack (0.124 singly → 0.666 jointly); fitted, rank 32 for all heads costs +0.275 and doubles every head's value; pattern + write compose super-additively (+0.486 vs 0.385) | fit | v712 | held; rank 64 and a 32/64 mix running (v714) |

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
| **no native, 16/64 + band 6-8 at 64 (v713)** | **25.9M** | **+0.060** | **0.985** | **yes (0.82 / 1.05)** |

## Open

- v714 (running): write maps at rank 64 (13.3M vs 47.8M) and a 32/64 mix; composition with the pattern program. v715 (queued): bands 0-8 at 64.
- Not yet done: *reading* the fitted content directions (what the 16–64 directions of the content heads are — the induction key of 5.5 is known; the rest are not), and whether the write directions align with the MLP readers downstream. The kernels of blocks 0–2 were read in the embedding-forward report; the deeper kernels (layer 5 content-critical, layers 6-8 mostly positional) are described only by shape.
