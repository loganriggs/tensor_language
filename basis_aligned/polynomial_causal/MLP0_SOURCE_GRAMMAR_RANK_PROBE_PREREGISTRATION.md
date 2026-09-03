# Preregistration — is MLP0's diffuse source grammar LOW-RANK? (CPU probe; parallel lane)

Date: 2026-09-03 02:32 UTC
Owner: Claude (parallel probe lane)
Status: frozen before any rank outcome; zero model forwards, CPU only, on
rung517's published group CE-effect profiles. Explains §2652's redundancy.
Codex's 518 untouched.

## Question

Rung517 (§2652) found MLP0's five source-relation effects are diffuse (spread
shares) yet strongly REDUNDANT (large singleton benefit, tiny removal
necessity). Redundancy implies the relations carry OVERLAPPING information for
MLP0's context computation. This asks whether the diffuse grammar is actually
LOW-RANK: do the five relations' causal CE-effect profiles over the 192 scored
positions span few dimensions (so any relation substitutes because MLP0 reads
one low-dimensional context summary), or are they five genuinely independent
directions? An effective-rank statement, not a compression fit.

## Computation (exact, deterministic)

Rung517's receipt (sha c8405a36cab0e8b50d91e3f525bf5a5106a95d2c42447ce9b83ab29378fd8307) stores, per corpus C in {PROSE,STRUCTURED}
and role R in {FIT,SELECT}, endpoint_position_ce_profiles: a 5×192 matrix
(five source-relation groups × 192 absolute positions 64:256) of the singleton
CE-effect profile. For each (C,R): center columns, SVD; report the effective
rank exp(-sum p_i log p_i) with p_i = S_i^2/sum(S^2), the top-1 and top-2
energy fractions, and the top-2 group-space LEFT singular vectors (5-dim). The
5-dim LEFT vectors are the structural invariant (the "which relations load the
shared context direction"); the 192-dim right vectors are position-indexed and
are NOT used for stability (internalized §2649 lesson: compare the low-dim
structural axis, never a sample-indexed loading).

## Frozen predictions

### pred_a — exact reproduction
Receipt sha matches; every profile matrix is 5×192 and finite; the pooled rank
recomputes deterministically.

### pred_b — the diffuse grammar is low-rank (both corpora, SELECT)
For BOTH PROSE and STRUCTURED SELECT: effective rank of the 5×192 centered
profile matrix <= 2.5 (the five relations' causal effects span <=~2 effective
dimensions — the redundancy IS low-rank, not five independent relations).

### pred_c — the low-rank structure is stable (group-space left vector)
The top group-space left singular vector (5-dim) has cosine >= .90 between FIT
and SELECT within EACH corpus, AND cosine >= .90 between PROSE-SELECT and
STRUCTURED-SELECT (the structural loading is stable across split and corpus).
NOTE: this compares the 5-dim GROUP loading, sanity-checked against a trivial
null — a random 5-vector has expected |cos| ~ 1/sqrt(5) ~= .45, so .90 is a
non-trivial bar.

## Strong null
Fires if pred_a fails, or pred_b fails on either corpus (effective rank > 2.5
-> the diffuse grammar is genuinely multi-dimensional and the redundancy is
functional overlap, not low-rank), or pred_c fails (the loading is split- or
corpus-specific). Reported beside §2652 either way; no bar changes.

## Price
Zero model forwards; CPU < 10 s; one receipt JSON. Nothing deployed.
