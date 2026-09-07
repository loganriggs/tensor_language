# Hourly strategic review — 2026-09-07 06:14 UTC

## Circuit target and current status

The controlling goal remains a smaller literal tensor program that says what is read, computed,
written, and consumed; predicts held-out and OOD effects; supports selective swaps/removals/edits;
and remains stable across prompts, construction families, gauges, and independent fits. A low
training loss, a fitted DAS axis, or an exact local algebraic identity is not a completion
certificate by itself.

The temporal/is-was path now has a stable eight-site response program: four attention heads
(`L8H1`, `L9H1`, `L9H4`, `L11H3`) and four MLP outputs (`MLP1`, `MLP3`, `MLP4`, `MLP6`), each carrying
a rank-8 response projector. The two independently fit projector families have minimum mean squared
principal cosine `.8907`; their aligned weight maps have minimum cosine `.9483`; and their local
`c_proj`/`Down` contraction identities close below `6.3e-14` RSE.

Prospective OOD testing on temporal-v10 and is/was-v11 prompts that were absent from projector fitting
shows worst six-cell residual about `.0162` and signed behavior recovery `.876-.907`. Ten-percent
sitewise activation noise barely changes this (`.01632` worst residual). The corrected control report
puts margin movement at `.058-.060` target scale and median full-vocabulary KL near `.0011-.0014`
nat. Its only registered miss is two top-1 flips among 36 controls (`.0556` versus the `.05` bar).
This rejects example memorization and ordinary noise fragility; it localizes the remaining issue to
selective decision-boundary movement.

## Constrained DAS and regularization verdict

The user's regularization hypothesis is partly right and needs a precise qualifier. Existing H3
red-teams show that full-vocabulary KL is useful: it reduced held-out full-vocabulary error from
roughly `.4485` to `.2445`, close to DIM's `.2385`, while tangent noise alone left it near `.4486`.
Row-held-out noisy/KL optimization nevertheless failed on complete construction families. The hard-
feasible follow-up selected the pooled step-zero axis and returned `family_memorization`. Thus the
old scalar complement target was under-specified, KL repairs a real loophole, and noise alone is not
an adequate cure; complete-family feasibility is the essential anti-memorization unit.

That rule was applied to the eight-site response program. A rank-8 search restricted to the union of
the two cross-fit subspaces used full-vocabulary KL and could not trade target fidelity for control
inertness. A step-10 consensus update passed sealed-family targets (signed recovery `.900/.926`,
worst residual `.0106/.0078`) and sealed controls (margin `.0479`, median KL `.00182`, zero top-1
flips). However, its selection objective improved only `.521766 -> .521373`, or about `.075%`, far
below the registered 10% materiality bar. The honest terminal is `kl_refinement_inconclusive`: the
corrected optimizer can find a slightly better point, but nearly all transferable quality was
already in the cross-fit response geometry. The simpler frozen projector remains the scientific
baseline.

## Weight tensors and operation-level program

Both frozen projectors were then executed literally through model weights rather than generic
projection hooks. Attention writes use

`(base_head - live_head) W_O^T + (delta_head Q)(W_O Q)^T`,

and MLP writes use

`base_mlp - live_mlp + (delta_hidden (W_D^T Q)) Q^T`.

All five registered gates pass. Local attention and MLP contractions close below `2.84e-14` and
`2.71e-14` RSE; end-to-end literal versus generic execution closes below `6.97e-13` RSE. On the
v11/v10 OOD bank, literal target recovery is at least `.878`, worst residual is `.0475`, and control
movement is `.0451`. This promotes the circuit from a readable subspace to a literal
weight-executable response program.

The next exact factor atlas decomposed the coordinates themselves. Across both tasks, both fits, and
all four attention heads, all 16 cells are dominated by base-pattern transport of changed values;
all 16 derive at least 80% of coordinate norm from the cue or post-cue suffix. Cross-fit factor
profiles have minimum cosine `.99977`. Fourteen of 16 MLP cells are adequately described by the
left/right linear response; the two exceptions are both `MLP1` on is/was, where the bilinear
interaction contributes `.284-.290` of complete coordinate norm. Exact factor closure is
`2.08e-14` RSE. The remaining nonlinear term is therefore localized, not diffuse.

## Circuit count and quality trajectory

The broad unit-circuit program screened 21 additional behaviors in the first Tier-3 batch and used
a fixed row-2 continuation on the misses. Thirteen behaviors now satisfy the complete row-2/3/4/5
screen across those two receipts; partial circuits remain explicitly partial. The family audit says
the possessive tasks largely share one circuit, the number family is only partly separable, and the
correlative family is mostly entangled. Tier-4/5 work has begun turning those masks into routes:
direct unembedding paths are literal for ten direct sets; seven near-offset paths are content-driven;
early heads write the near content for number and medial possessive tasks; and long possessive
constructions relay the cue through the first between-cue position. These are route components, not
yet thirteen independent completed transparent programs.

Quality is improving faster than raw count: the temporal/is-was circuit now has split stability,
OOD prediction, noise robustness, selective controls, literal weight execution, and exact operation
factorization. The remaining gap is recursive generation of the selected coefficients and causal
replay of the factor-pruned program. The broad thirteen have good causal row coverage but only a
subset has comparable route and tensor detail.

## Next highest-information work

1. Causally replay the factor-pruned temporal/is-was program: attention suffix value transport plus
   MLP left/right terms, retaining the MLP1/is-was bilinear interaction as an explicit exception.
   Kill the simplification if the pruned factors fail OOD target or control bars.
2. Recursively localize which earlier heads/MLPs generate the transported value coordinates for
   `L8H1/L9H1/L9H4/L11H3`, using the exact `W_O Q` maps as readers rather than final logits.
3. Keep the frozen cross-fit projector as the primary program; treat the KL step-10 axis as a
   diagnostic until a material sealed advantage appears on a new complete family.
4. Continue the high-throughput Tier-5 route recursion only where it changes read/write boundaries;
   do not inflate circuit count by naming every partial factorization a completed circuit.

`CIRCUIT_FOCUS: PASS` — every main-line experiment changed OOD validity, selective manipulation,
weight execution, or operation-level identification.

`CEREMONY_BUDGET: PASS` — two bookkeeping failures were preserved and corrected without changing
scientific thresholds; the decisive factor and literal-weight screens cost 10 and 21 forwards.

`NOVELTY_LESSON_GATE: PASS` — the next experiment is causal factor replay, not another rank,
penalty, or cosmetic optimizer sweep.

