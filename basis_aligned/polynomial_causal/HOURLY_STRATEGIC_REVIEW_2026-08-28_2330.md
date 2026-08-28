# Hourly strategic review — 2026-08-28 23:30 UTC

## The short version

The project has reached a clean boundary.  A program that assigns each token one
fixed output row can now exactly attain the model's own best **context-free**
prediction at full rank.  It still sits about 2.74 cross-entropy nats above the live
model because deleting contextual attention deletes information that no fixed
per-token table can recreate.  This means more work inside the same fixed-row class
cannot explain the missing contextual computation.

The highest-priority next experiment therefore crosses two real intervention axes:

1. which subset of MLP0, MLP1, and MLP2 is replaced; and
2. how much of the later contextual network is replaced.

It asks whether the resulting 64 whole-model losses obey a low-rank interaction
law.  Such a law would be useful because a small set of measured programs would
predict untouched compositions, not merely reconstruct one module's activations.
The statistical design was already frozen.  This review completed its source-closed
two-role backend, transaction, and scorer; 33 focused CPU tests pass.  An independent
launch audit is running.  No model outcome from this new grid has been opened.

## What fraction is actually explained?

These ledgers measure different things and must not be added together.

| Ledger | Current value | Meaning |
|---|---:|---|
| Structural module coverage | 36 / 36 | Every attention and MLP site has an executable structural surrogate or characterization. This does **not** mean the whole model is behaviorally reproduced. |
| Certified storage removed | 5.3481% | Parameters removed under the project's strict storage certificate. |
| Older named-behavior accounting | 32.1% ± 6.4% | Fraction attributed to named behavioral mechanisms under the older, broader accounting. |
| Strict named causal CE recovery | 10.923% | Recovery under the strict causal replacement currency. The unexplained strict residual is 4.72714 nats. |
| Final extraction/removal/OOD actions | 0 / 68 | The action-level semantic interface is physically captured but not yet scientifically reduced and certified. |

There is also a narrower result about the 36-site context-free compiler.  At full
rank it exactly reaches the model-defined per-token ceiling on covered positions.
That exhausts its **function class**, not the live model.  The remaining roughly
2.74 nats to the live model are contextual-class error.

## Important new evidence

### Rank is not a sufficient simplicity coordinate

At 5,419 covered token types, the full-rank context-free program costs 230.1 million
stored real numbers and reaches all-position CE 6.01167.  At 16,110 covered types,
a rank-256 program costs only 164.5 million reals and reaches CE 5.98851.  The latter
is both smaller and better.

So the simplicity curve is at least two-dimensional:

\[
\text{program cost and loss} = f(\text{vocabulary coverage},\text{row rank}).
\]

The axes interact: tripling coverage helps full rank by 0.106 nats but hurts rank 4
by 0.061 nats.  A one-axis "rank versus reconstruction" plot can therefore select a
dominated program.  This is direct evidence that a valid simplicity measure must
price the whole executable representation—dictionary coverage, rank, fallback, and
corrections—then compare programs by predictive loss on one common population.

### Independent local choices do not compose

- A rank-at-most-two factorization of the previous compiled-mask response grid
  looked spectrally compact but failed to predict untouched interactions.
- Choosing the locally better table independently at each site made the whole
  program worse than either uniform choice.
- The early MLPs are causally coupled: exact joint repair is much larger than the
  sum of singleton repairs, and MLP2 changes sign after MLP0+1 are repaired.

These facts jointly prune "simplify every component independently and concatenate
the answers."  The relevant object is an interface law for compositions.

## What the active tensor-cross experiment computes

Let \(C_{ij}\) be whole-model CE after applying early-MLP choice \(i\) and contextual
suffix choice \(j\).  Its non-additive interaction is

\[
\Delta_{ij}=C_{ij}-C_{i0}-C_{0j}+C_{00}.
\]

If \(\Delta\) has rank \(r\), selected rows and columns determine the rest:

\[
\widehat{\Delta}=\Delta_{:,J}\,\Delta_{I,J}^{-1}\,\Delta_{I,:}.
\]

This formula is invariant to a change of latent basis: it predicts observable
whole-model losses without assigning a possibly arbitrary meaning to individual
latent coordinates.  The fixed pivot rows and columns were selected using older CE
data before this grid existed.  Rank 3 uses 48 discovery cells and predicts seven
unseen validation cells.  Rank 4 then uses those seven and predicts nine final
heldout cells.  Both disjoint document roles must pass independently.

This is a tensor-network advantage: the experiment varies legal program prefixes
and suffixes around a physical cut, so a successful low-rank law can compose across
the cut.  It is stronger than a low local activation MSE and weaker than a complete
semantic interpretation.

## Largest remaining gaps

1. **Contextual interface law.** We do not yet know whether the early MLP program
   and later contextual program communicate through a small predictive state.
2. **Whole-model composition.** Good MLP0/1/2 approximations have not yet been shown
   to compose without downstream compensation changing the answer.
3. **Stable simplicity frontier.** Coverage and rank interact; fallback and
   correction costs must enter the same executable cost model.
4. **Semantic/action reducer.** The 68 intervention actions lack the reviewed
   response reducer needed for extraction, selective removal, and OOD claims.
5. **OOD breadth.** The current two roles are document-disjoint, but do not by
   themselves establish transport across substantially different distributions.

## Ranked next actions

The ordering uses expected information gain, causal relevance, ability to compose,
falsifiability, GPU cost, and duplication with completed work.

1. **Finish audit and execute the prospective early-MLP/context cross.** This is the
   cheapest direct test of a missing whole-model interface. It has untouched cells,
   two disjoint roles, exact failure gates, and moderate GPU cost. The CPU-side
   implementation is complete; launch remains NO-GO until independent audit and a
   pushed source closure.
2. **Close the 68-action semantic reducer.** This supplies vector-valued causal
   responses—copying, frequency, and other action outcomes—so a CE-only factorization
   cannot masquerade as extraction or selective editability.
3. **Measure and price the two-dimensional coverage-by-rank frontier.** The current
   GPU lane is already testing the middle coverage point, so no duplicate job should
   be started. The deliverable is a Pareto set under common-population CE plus total
   stored and executable cost, not another rank-only curve.
4. **Fit one joint downstream-weighted MLP0/1/2 representation.** If the cross passes,
   use its latent interface as the weighting/canonicalization target for a shared
   dictionary or tensor factorization. If it fails, use the measured residual to
   choose a hierarchical/Mobius interaction model. Do not optimize local MSE alone.
5. **Test an adjacent cut, then a minimal action realization.** Two adjacent passing
   cross laws would justify Hankel/system-identification machinery to find the
   smallest predictive state. One passing cut alone is insufficient.

## Ideas pruned in this review

- More refinements confined to fixed per-token rows: that class is exhausted at
  full rank and cannot recreate context.
- Rank as the sole definition of simplicity: the coverage-by-rank dominance result
  falsifies it operationally.
- Independent per-site table selection: it already fails whole-program composition.
- Rank-at-most-two response factorization: it fails untouched interaction tests.
- Generic low-rank-plus-a-few-outliers repair of that residual: its error is spread
  across many cells rather than concentrated in a few exceptions.
- Semantic naming or SAE sparsity as a success criterion by itself: neither licenses
  prediction, OOD transport, extraction, or selective removal without causal tests.

## Safe action executed during this review

Implemented the complete frozen launch boundary:

- one immutable shared program built once and bound to both evaluation roles;
- a truly live `(0,0)` origin and a genuine MLP0 factor;
- all 36 native modules counted before any requested output substitution;
- exact ordered execution of 64 cells on `skip7000`, then 64 on `skip11000`;
- authority published before outcomes, no partial outcome publication, backend
  close and input/source revalidation, tensor payload, manifest, and receipt last;
- capability-separated rank-3 and rank-4 scoring;
- independent two-role CE conjunction and no invented top-1 pass threshold;
- failure tests for second-role crashes, namespace collision, wrong-role rows,
  descriptor mismatch, lock ownership, and payload mutation.

Focused result: **33 CPU tests pass**.  The GPU was occupied by the independent
coverage-by-rank job, so no competing model process was launched.  The early-MLP
cross remains explicitly **NO-GO pending the independent audit and a clean pushed
source commit**.

## UPDATE — launch closure after independent audit (23:43 UTC)

The pending conditions above are now closed.  The final independent audit found no
remaining bypass after four rounds of adversarial review.  The suite is now **47/47
passing**, including independent ALS and quantile references, both authority race
windows, replay of all 128 physical call ledgers, corruption tests for cell/role/stage
hash chains, and a negative test proving that a synthetic measurement cannot receive
a canonical score receipt.

The implementation bytes are committed and pushed at `116fdfd2`.  The runner
recomputes and binds the then-current pushed `HEAD` across the exact 22-file closure
immediately before launch, so the authority cannot cite a stale hash after later
documentation-only commits. Both canonical output namespaces remain pristine.  The launch is scientifically and
operationally **GO**, but has not started because the GPU is occupied by the
already-running coverage-by-rank frontier-knee job.  No early-MLP/context-cross
outcome has been opened.
