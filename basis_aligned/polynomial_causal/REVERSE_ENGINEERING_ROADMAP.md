# Roadmap to an executable reverse engineering of bilin18

## Operational end state

“Fully reverse engineered” does not mean assigning an English phrase to every
parameter. The target is a smaller, typed tensor program `P` with an explicit map to
the original model `F` such that:

1. `P` reproduces natural and OOD behavior at a declared distortion budget;
2. its response to held-out interventions predicts `F`'s response;
3. independently described fragments compose without large interaction drift;
4. target fragments can be removed, transplanted, or edited with predicted collateral;
5. its complexity is reported conditionally on a frozen grammar and decoder, with
   gauge-equivalent programs priced identically within tolerance;
6. every claimed exact simplification has a certificate, and heuristic compiler
   outputs are labeled as upper bounds.

Provisional whole-program gates are global delta-CE at most `0.02` on held-out
FineWeb, at most `0.05` on a second corpus, no powered behavior class losing more
than 10% of its clean advantage, normalized held-out intervention-response error at
most 25%, and no more than 1% price drift under registered gauges and rewrites. These
are adoption targets, not claims about the present model.

## Progress accounting

Maintain a circuit balance sheet rather than counting discoveries. Every candidate
fragment records:

- typed input and output interfaces, including precision and norm semantics;
- standalone and amortized description bits, product/FLOP cost, and interface size;
- natural/OOD replacement distortion;
- intervention families used for fitting versus held out for testing;
- selective-removal effect and collateral;
- overlap/shared dependencies with other fragments;
- composition error when installed with the current best partial program;
- the residual effect not explained by the fragment.

Replacement recovery is scored relative to an explicit null at the same site. The
primary dashboard is cumulative whole-program recovery and the remaining CE/KL and
causal-response residual, not the sum of overlapping ablation effects.

## Ranked workstreams

### P0 — Build the whole-model coverage and composition ledger

Inventory the already certified heads, MLP slices, tables, and shared services. Put
each through one common replacement harness and assemble the strongest non-duplicated
composite. Measure where its residual CE lives by layer, token class, output slice,
and intervention family.

**Why first:** this converts thousands of local facts into a map of what is actually
missing. Without it, another named circuit may duplicate an existing service or
explain no additional model behavior.

**First synthesis (2026-08-27):** `whole_model_balance_sheet.py` now assembles the
frozen Theseus anchor, current ship registry, causal coverage ledger, named-variable
draws, and composition diagnostics without flattening their denominators. The
current fidelity ship replaces 36/36 top-level modules but is still about +0.93 CE
above the 2.9455 clean anchor. Separately, analytic-interface substitution recovers
0.9982 of its own mean-floor denominator, named token+topic+previous variables score
0.321, named causal paths cover 0.1092 of global ablation headroom, and the older
all-stand-in composition stress test recovers 0.124. These facts are complementary,
not candidates for one averaged “percent understood.” The next P0 increment is
residual localization and a manifest-complete composite record, not a new registry.

### P1 — Trace typed causal interfaces backward from outputs

For each behaviorally important output slice, identify the minimal residual channel,
its writers, its computational readers, and its final readout. Represent the result
as a small directed tensor program with shared nodes. Use both class-seeded slices
and behavior-agnostic output bases, but evaluate discovery on disjoint classes.

**Why second:** output-to-input tracing supplies stable boundary variables for
replacement and intervention transport. Internal coordinates without a boundary are
gauge-dependent and do not compose.

### P2 — Compile exact polynomial fragments between norm boundaries

Exploit the literal bilinear MLP and attention structure. Canonicalize linear maps,
real scalar quadratics by inertia, and registered CP/tensor gauges. Next, develop
joint vector-valued quadratic factorizations and shared projection dictionaries
across output slices. Treat RMSNorm and softcap as explicit analytic primitives, or
freeze/approximate them on a certified domain.

**Why third:** algebra can give exact reductions that activation-only compression
misses. The question form already demonstrates this: spectral interface rank 2 but
exact multiplicative complexity 1. The harder payoff is shared vector-valued forms,
where one factor library may serve many outputs and circuits.

**Scope correction (2026-08-27):** the full-vector output flattening has stable
numerical rank 1152/1152 for audited MLPs 0,1,2,11,17, against 4608 native products.
So an unchanged full MLP permits at most a 4x exact product reduction under this
bound, whereas the selected question scalar permits 4608-to-1 at its interface.
The compiler target must be a jointly discovered causally sufficient output/content
API, not the whole 1152-output tensor by default.

### P3 — Synthesize replacements jointly, not one module at a time

Fit tensor-program fragments against natural outputs plus selected intervention
families, price shared factors once, and install them into the current composite.
Optimize the residual after every addition. Include explicit transport maps only when
intervening on latent variables rather than common module boundaries.

**Why fourth:** local low reconstruction error routinely fails under composition.
The desired object is a globally useful program, not a folder of individually good
approximations.

### P4 — Attack the distributed middle only after its residual is localized

The early/late and narrow output-channel mechanisms are relatively tractable; the
contextual middle MLPs remain the likely irreducible wall. Once P0 localizes their
unique residual, test data-supported lifted subspaces, shared bilinear dictionaries,
conditional/routed fragments, and low-degree interventional response models. Require
held-out compositional gain over rank, parameter-count, and compressed-byte baselines.

**Why fifth:** this is probably the hardest and most expensive work. It should be
conditioned on exactly what the existing composite fails to reproduce, rather than
another broad search over 64M parameters.

### P5 — Validate extraction, editing, and generalization

For every mature fragment, predeclare transplant/removal edits and predict both target
damage and collateral. Test fresh rows, a second corpus, disjoint token classes,
unseen intervention families, pairwise composition, and all-module composition.

**Why sixth:** these are the practical benefits of reverse engineering and the final
defense against a merely descriptive compression.

## Immediate queue after the current registered batch

1. Score and preserve the behavior-agnostic output-slice audit now running.
2. Extend the P0 balance sheet with manifest-complete current-composite residuals by
   layer group, token class, output slice, and held-out intervention family.
3. Preregister a matched-one-product question test: the exact indefinite paired form
   versus the best single square, measuring scalar reconstruction, class/global CE,
   finite-precision stability, and composition with the known question circuit.
4. Generalize from one scalar output direction to a small joint output basis and test
   whether projections/factors can be shared at lower bits with equal causal fidelity.
5. Rebuild the best whole-model partial program and let its residual choose the next
   target layer/interface.

## Pruning rules

Stop or demote a direction when:

- it does not improve held-out prediction beyond simple rank/bits/FLOP baselines;
- its evaluation distribution is badly OOD for the claim being made;
- it explains a local reconstruction but adds no composite replacement recovery;
- it re-discovers a service already priced in the shared library;
- its score changes materially under gauge-equivalent rewrites;
- it needs the same intervention outcomes later claimed as predictions;
- it lacks a certificate but is being described as a minimum;
- its expected information gain per GPU-hour is lower than measuring the current
  composite's largest residual.

The prefix/continuation Hankel route is currently demoted by these rules: splice CE
is roughly 3.5 nats/token above natural, rank-95 is 23–24 of 48, and low-rank
completion improves only 4.5–10.1% against a registered 30% bar.

## Hourly strategic review

The local session receives an hourly prompt from
`hourly_strategic_review.sh`. Each tick inspects new evidence, brainstorms candidate
actions, prunes them by the rules above, ranks the top five, and executes the highest
priority safe unblocked action. The cron is session-local and must be recreated after
a container/session recycle; this file and the script are the durable policy.
