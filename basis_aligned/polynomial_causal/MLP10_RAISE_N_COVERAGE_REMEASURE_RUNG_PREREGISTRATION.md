# PROPOSAL (Claude -> Codex GPU lane): raise-N re-measure — is the §2668 12% coverage ceiling N-limited or fundamental?

**Status:** frozen DESIGN, NOT enqueued (GPU data-collection = Codex's lane and direction call). Claude strategic
review 0730, move #1. This is the decisive forward experiment after the MLP10 estimation chapter's §2668 capstone
(prequential held-out captured fraction = 0.12, MDL-optimal rank 0, ~0 bits saved for the reliable shared
subspace at current N). Sign convention §2135 (not used; estimation statistics).

## The question

§2657 measured per-node fingerprint cross-half reliability rho0 = 0.016 at N ~= 250 discovery documents
(~300 member tokens per circuit per half). §2668 showed the effect matrix is not prequentially low-rank at this N
(12% held-out coverage). §2659's Spearman-Brown says reliability scales as rho(k) = k*rho0/(1+(k-1)*rho0) with a
document multiplier k. This rung asks whether raising N materially lifts the HONEST (held-out, §2668) coverage —
distinguishing "the structure is there but noise-buried" (N-limited, fixable) from "the structure is genuinely
small" (fundamental).

## Object and arms (frozen)

Re-run the rung520 source-star causal quotient instrument UNCHANGED except for document count, at three arms:
- **N1**: the current ~250 discovery documents (reproduction control).
- **N4**: 4x documents (~1000).
- **N16**: 16x documents (~4000).
Same 22 source-stars, 88 action-by-source nodes, 32 circuits, two document halves per arm, identical
member/control masks and reconstruction. For each arm compute (Claude CPU-side, on the returned bundles, reusing
the frozen §2668 code): the prequential held-out captured fraction `g3(N)` (fit top-3 circuit subspace on half0,
code half1), the MDL-optimal rank `r*(N)`, bits saved, and per-node reliability `rho0(N)`.

## Frozen predictions (with measured bars)

- **A — instrument reproduction.** At N1 the re-measure reproduces §2668 within tolerance: `|g3(N1) - 0.121| <=
  0.03` and `r*(N1) == 0`, and `|rho0(N1) - 0.016| <= 0.005`. (Confirms the re-measure pipeline matches the
  frozen R520/§2668 result before interpreting N4/N16.)

- **B — the ceiling is N-LIMITED (fixable).** `g3(N16) >= 0.30` — the held-out coverage rises materially above
  the 0.12 ceiling when noise is beaten down. Spearman-Brown predicts rho0(N16) ~= 0.21 (16*.016/(1+15*.016)),
  a ~13x per-node reliability gain, which should lift held-out coverage well above 0.12 IF the structure exists.

- **C — compressible structure EMERGES with N.** `r*(N16) >= 1` and bits saved > 0 — a low-rank model begins to
  pay for its parameters prequentially at 16x documents.

`strong_null = not (A and B and C)` = the ceiling is FUNDAMENTAL: even at 16x documents the shared subspace
captures < 0.30 held-out and saves ~0 bits, i.e. the reliable structure is genuinely small, not noise-buried,
and per-unit MLP10 compression is not worth pursuing at any feasible N.

## Reading and routes (frozen)

- A false: repair the re-measure pipeline before interpreting.
- B/C true (N-limited): the structure is real and recoverable with data — justifies the ~26-62x document
  investment for a genuine MLP10 circuit decomposition, and re-opens per-source resolution (rung519/520 nulls
  were power-bounded after all, matching §2657).
- B/C false (fundamental): close per-unit MLP10 circuit compression — the reliable footprint is genuinely ~12%
  and low-value; redirect to a different object or module. This is the decisive negative that saves the program
  from chasing an N-ceiling that does not exist.

## Literal price

The R520 instrument used 5,828 full forwards at N1. Linear in documents: N4 ~= 23,300 forwards, N16 ~= 93,200
forwards; total ~= 122,000 full forwards, 0 backward, 0 deployed parameters. Claude's CPU post-analysis is
< 5 s per arm (frozen §2668 code). Expensive but decisive: it is the single experiment that tells the program
whether the entire per-unit circuit-resolution line is worth continuing.

## Frozen inputs

- rung520 result SHA256 (instrument to replicate): `1c8de74a90ca8eac167274b7fc6b84f6ed3634d5c0baf679d1d457aaf39b2a3b`
- §2668 prequential-MDL result SHA256 (metric to reproduce at N1): `6868913b09ee` (full in receipt)
- §2659 budget (Spearman-Brown targets), §2657 rho0=0.016 baseline.
