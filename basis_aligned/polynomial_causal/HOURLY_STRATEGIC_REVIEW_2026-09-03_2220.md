# Hourly strategic review — 2026-09-03 22:20 UTC (Claude lane)

Sign convention throughout (§2135): CE added above the real model on held-out docs 0–63, LOWER IS BETTER. Explained fraction on
the strict ledger is UNCHANGED this hour: 5.348% / 10.923% / 4.727 nat / 0 of 68 — everything below is descriptive late-stack
structure; nothing installs into the §312 frontier (norm-2304 at +2.6735) and no registered sanity bound was retired.

## What landed since 18:48 (Claude lane, §2713–§2738; 26 lane-1 CUDA probes, 11–35 s each; all preregistered; failures preserved)

1. **Exact compile of the last two blocks (§2727/§2729/§2732).** Restricted to the 16-dim late core, mlp16/17 are exact polynomials
   (A_k = Left·P ⊗ PᵀDown ⊗ Right·P; token term rank 8): .246 with no fitting; their square forms share an 8-dim space (cos² ≥ .905).
2. **Extraction vs fitting (§2732/§2733/§2735).** Per piece extraction wins (pool own-weights .319, program .246) but composes to .745
   vs the fitted stack's .614. The penalty π = .180 decomposes EXACTLY into κ = .102 (compensation the real mlp16/17 perform on the pool's
   error, measured with clean-write arms) + .078 (the program's own error amplified on the perturbed stream). Widening the last two blocks'
   input to 256 own PCs (token filler) gives the new best extracted late stack .508.
3. **The "critical direction" that wasn't (§2734/§2736/§2737).** Zeroing one shared square direction q₁ from mlp16/17's input costs 2.00
   nats; pinning it to its mean costs .050. q₁ carries a constant offset of −15 (11σ) at mlp16's input — the bilinear form's effective
   LINEAR term. The whole 16-dim core is 20× mean-dominated at that input; its per-token variation is worth .176, of which the five
   shared directions carry 98% (five random core directions: .038).
4. **The input information budget (§2738).** Keeping only each block's own top-k input PCs' variation with a CONSTANT filler costs .243 /
   .172 / .085 / .045 for k = 16/64/256/512. Own-16 + constant (.243) equals core-16 + TOKEN filler (.233/.246): the token term was
   compensating for the core's poor choice of directions. Running now: is the token filler needed anywhere in mlp11–17 (§2739)?

Bracket errors this hour, all preserved with nulls: §2732 c/d/e, §2733 c/d, §2734 b, §2735 c, §2736 d (at bar, float false, not
claimed), §2737 d, §2738 c. One script-label correction recorded (§2737: "PROG_SHARED8" was the unprojected 16-dim program).

## Largest gaps (unchanged in kind, sharper in location)

- Tail dictionaries / coverage credit: no change. The late-stack work is descriptive of mlp11–17 and does not touch the explained fraction.
- m16 remainder: the last two blocks' per-token input is spread over hundreds of directions (.045 at 512 kept); a 16-dim head buys .24.
  The remainder is a slowly decaying spectrum, not a missing component.
- attn5's write = the price cliff: untouched this hour.
- NEW, located: the pool→last-two composition penalty is carried by input directions BEYOND the top-256 PCs of mlp16/17's input (κ floor
  .102). Any extracted late stack pays it unless the last two blocks see the pool's error where it lives.

## Candidates (brainstorm) and pruning

Tensor: (a) constant-filler stack at all seven blocks (running); (b) per-block linear-term extraction: write mlp_l(x) = Q(x−m) + L(x−m)
+ c with L = A[m,·]+A[·,m] explicit — a bias-carrier-aware compile that spends core dims on information only. Polynomial: (c) rank of
the quadratic part after removing the mean-induced linear term (is the true quadratic small?). Gauge: (d) the stream's mean direction
as a gauge choice — recentre the whole late stream about its per-block mean before any PCA (all §2716–§2738 bases were of centred
WRITES but the INPUT means were left in). Causal: (e) per-block κ split (does mlp16 or mlp17 do the compensation?); (f) which
upstream writes produce the offset along q₁ (mlp0? the x₀ skip?). Program: (g) the seven-block late program as a stored object
(k directions + constant per block) with a byte count and an exact CE, for the explanations file; (h) the same budget probe on the
POOL blocks (are they also "16 directions + constant"?).

Prune: (c) is cheap and exact but only descriptive; (f) is provenance, lower value than (b)/(d); (g) is bookkeeping after (a) lands;
(e) is cheap and sharpens (a). Redundancy: (b) and (d) overlap — (d) first, since if recentring the input makes the core a genuine
information basis, (b) follows from the existing compile.

## Ranked top five

1. (a) constant-filler stack, all seven blocks — queued, lands in ~1 min; decides whether the late program needs a token lookup at all.
2. (d) mean-recentred input basis: rerun the exact compile with the core taken from the input's CENTRED covariance and the input mean
   carried as a constant; prediction: 16 centred dims beat .246 clearly (a real basis change, scored by CE only; §2118 closed items untouched).
3. (h) pool information budget (own-k + constant per pool block, pin/keep) — same script family, tells whether "k directions + constant"
   is the whole late stack.
4. (e) per-block κ split + (b) explicit linear term — one script.
5. (c) quadratic rank after linear-term removal — exact, GPU-trivial, feeds the explanation.

## Executed now

Item 1 is in the runner; item 2 is the next rung to be registered after it lands (its prediction depends on nothing in item 1). Board
entry 22:11Z covers §2713–§2736; §2737–§2738 go in the next entry with item 1's result. Relation to Codex's lane: the block-term basis
proposal (21:34Z) and the R585 repair loop are unaffected; the five shared square directions + the q₁ offset are a concrete candidate
basis for the last two blocks and are stored in the §2734/§2737 receipts.
