# Hourly circuit review — 2026-09-07 00:14 UTC

## Goal and circuit-quality bar

The goal remains a smaller transparent tensor program that is jointly predictive on fresh/OOD
text, composable under joint installation, selectively manipulable, and literally cheaper than the
native model path. A circuit is not counted as mature merely because an activation direction or
single patch works: it needs a read/compute/write specification, native head/module boundaries,
held-out prediction, sufficiency/extraction, selective edits, composition, and stable identification.

## What changed since 23:14

The is/was MLP8-complement route advanced from separate local branches to a nearly complete causal
response program. The corrected direct-skip compiler closes the literal residual, c_v, attention
response, and logits to float precision, but the direct L11/L15 reader branch explains only
5.47%/3.91% of parent behavior. Exact MLP9 factoring instead yields stable Left, Right, and
interaction terms: Left+Right transfers 98.97%/95.69% behavior and 99.91%/101.43% Q8 on fresh v11,
while the interaction remains 6.23%/15.32% behavior and cannot be dropped uniformly.

Composing the semantic seed (L9H1/H4, exact MLP9, L11H1/H3, L15H5) initially retained only
49.85%/54.90% behavior and 42.98%/48.00% Q8 because the experiment base-clamped later propagation.
A complete group factorial then localized the missing branch: restoring downstream responses raises
coverage to 92.51%/95.90% behavior and 93.03%/94.80% Q8, versus only 57.36%/59.04% behavior from the
attention9 remainder. Group interaction is .192%, so greedy composition is licensed.

Panel-balanced zero-fit selection compresses that downstream branch to complete MLP10-14 plus the
attention15 remainder. On sealed rows it recovers 92.01%/94.04% of downstream-target behavior and
93.16%/96.96% of Q8. Splitting the last group finds H1 alone is the parsimonious correction: its
sealed behavior error is .664%/.280% and Q8 error 1.211%/.542%. The five-head fidelity variant is
numerically closer but adds four heads only to polish a tiny residual. Across the sequential sealed
tests, the parsimonious assembled route is therefore roughly in the mid-80s to low-90s percent range
of unrestricted behavior/Q8; a single end-to-end fresh confirmation must measure this directly
before release.

## DAS and weight-tensor status

The DAS conclusion is now qualified rather than binary. Scalar constrained DAS can beat difference
in means on its scalar training objective while losing full-vocabulary transfer. Noise/KL alone has
not dominated the unregularized method across both panels and banks. Aligned full-effect optimization
at lambda .3 wins one sealed vector panel but loses scalar usefulness. The genuine optimized win is
the multicue pooled aligned rank-one method: full-vocabulary joint objective is .7588 versus .8308
for pooled DIM on A1 and .7664 versus .8304 on A2. Yet complement fractions .345/.354 show it remains
task-conditioned. The next DAS advance should therefore use multiple environments and independent
downstream readers; another single-task regularization sweep would mostly retest memorization.

The user's weight-translation proposal is productive. Exact Q8 contractions rank attention15 H1
first under Q, K, Q2, V, O, and OV and second under K2; H1's OV score is 65.32 versus at most 11.62
for other remainder heads. The static Q8-to-Q8 MLP tensor score correlates .70 with causal singleton
importance. But static attention-group norms are weak (best OV correlation .314), and exact
activation-conditioned MLP Q8-write magnitude correlates only .10. The activation contractions
themselves replay cached writes within 2.15e-5 relative error, so this is a scientific result rather
than an implementation failure: weight tensors identify incidence and within-module heads, while
cross-layer causal importance additionally requires downstream sensitivity. Q8-input terms explain
33–67% of MLP response norm; orthogonal task context contributes 69–94%, with only 1–3% residual
bilinear interaction.

## Throughput, failures, and confounds

This hour produced eleven durable experimental receipts plus one preserved implementation failure.
The failure was an out-of-range attention-head inventory (twelve assumed versus the checkpoint's
nine); it wrote no evidence, was repaired to H0-H8 excluding H5, and reran successfully. All GPU work
used the managed serial queue. Search was discrete and zero-fit; discovery and confirmation rows were
disjoint for both greedy stages. Combined arms were executed dynamically in causal order rather than
estimated by adding singleton effects. Full-pool replay and native self-clamps are exact.

`CIRCUIT_FOCUS: PASS` — every unit changed a computational boundary, exact tensor identity, held-out
prediction, or composition claim.

`CEREMONY_BUDGET: PASS` — the dominant cost was causal execution and compiler implementation; the
zero-forward weight atlas reused frozen causal labels and fixed metric families.

`NOVELTY_LESSON_GATE: PASS` — the missing-response failure changed the topology, the distributed
result triggered greedy selection, and the weak activation-rank result now triggers downstream
sensitivity rather than another norm search.

## Direction choice

The immediate next experiment is a finite downstream-sensitivity atlas for MLP10-14. Each local Q8
write is already exactly compiled from Left/Right/Down weights; removing that cached response while
allowing the later network to recompute measures the finite reader gain from the module to resid18
and behavior. This creates the requested weight-readable edge table: exact local write, fraction read
from Q8 versus context, and downstream causal gain. After that, freeze the parsimonious program and
run one end-to-end fresh/OOD confirmation with selective collateral controls. In parallel priority,
the next DAS experiment should be multi-environment/multi-reader aligned optimization with DIM,
plain, noise/KL, and anchored baselines under matched rank and sealed construction blocks.

Ranked moves:

1. **MLP10-14 finite downstream-sensitivity atlas** — required to turn weight incidence into causal
   edge strength; kill the scalar ranking interpretation if local removal effects remain context-only.
2. **End-to-end fresh program confirmation** — directly measure unrestricted behavior/Q8 coverage
   and collateral after freezing H1 as the late correction.
3. **Multi-environment, multi-reader constrained DAS** — test whether regularization discovers a
   reusable causal operator rather than a task-specific separator.
