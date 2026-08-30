# Hourly strategic review — 2026-08-30 00:25 UTC

## Bottom line

The strict native-model explanation has not increased in this interval. The main
scientific experiment is still the MLP2 error-Rayleigh test: it asks whether an
MLP2 approximation is simple in the directions that the rest of the native model
can actually observe, and whether that definition predicts the known MLP0–MLP2
composition error on documents that were not used to choose the metric.

The first DESIGN attempt failed before producing a numerical ledger because its
control hash sent a BF16 tensor directly through NumPy. That exact failure is
preserved. The BF16 repair passed its mathematical and byte-level tests, but an
independent audit found a second, non-scientific lifecycle gap: v2 bound the failed
v1 DESIGN authority and failure, but did not bind the absence of every possible v1
HELDOUT/predictor terminal and lock. Launching in that state would permit a late v1
artifact to appear without invalidating v2.

That gap is now repaired at pushed commit `8eb77232`. Thirteen absent v1 paths are
part of every protected snapshot; the authority, failure, and all absences are read
twice to close the cross-file race window; and every late path has a direct test.
The focused recovery suite passes `47/47`. A fresh exact-commit independent audit is
running. No v2 model response or scientific outcome has been opened.

In parallel, the shared GPU is running the shipped-program experiment
`are_the_two_knees_one_boundary`. It tests whether the table-rank boundary near MLP10
and the stream-map boundary near MLP8 are the same boundary. This is useful executable
compression work, but it is not allowed to move the strict native explanation ledger.

## What fraction of the model is actually explained?

The strict balance sheet remains:

- structural intervention access: `36/36` component sites;
- certified removable storage: `29,196,288 / 545,904,054 = 5.348245316%`;
- named causal cross-entropy: `0.57968 / 5.30682 = 10.923302467%`;
- unexplained causal cross-entropy: `4.72714` nat, or `89.0767%`;
- terminal extraction/removal/OOD actions: `0/68`.

The current all-table program uses about `202.6M` stored values after heterogeneous
rank allocation, but its CE is still around `5.9` rather than the native model's
roughly `3.14`. It is therefore a useful compression artifact, not a nearly complete
reverse engineering of bilin18.

## Terms and the current computation

For an MLP2 input state $z$, let $f_2(z)$ be the native MLP2 write and $P(z)$ one of
three rank-512 approximation writes. Its error is

$$
E(z)=P(z)-f_2(z).
$$

Local mean-squared error measures only $\lVert E\rVert^2$. The Rayleigh experiment
instead injects small signed fractions of the error into the exact native model,

$$
f_2(z)+\alpha E(z),
\qquad
\alpha\in\{-1/8,-1/16,+1/16,+1/8\},
$$

and measures what downstream computation does with that direction. Its features are:

1. the directional change in true-token CE;
2. the categorical-Fisher energy of the final-logit change, which approximates the
   local change in the model's output distribution while ignoring uniform logit
   shifts;
3. response energy at attention 5;
4. response energy at attention 6.

The finite target on document $d$ is the mixed MLP0–MLP2 effect

$$
i_{d,P}=
[CE_{d,C}(1)-CE_{d,C}(0)]
-[CE_{d,N}(1)-CE_{d,N}(0)],
$$

where $N$ uses native MLP0, $C$ uses compressed MLP0-C512, $0$ uses native MLP2,
and $1$ applies the complete MLP2 approximation. This subtraction removes each
MLP2 program's ordinary standalone damage and asks specifically how much additional
damage appears after MLP0 is compressed.

The three predictor families are local error only; final CE/Fisher consequences;
and final consequences plus attention-5/6 responses. They are selected using only
32 DESIGN source documents and then frozen. Performance is evaluated once on 32
disjoint HELDOUT source documents. Deranged and covariance-matched random error banks
test whether a predictor merely exploits error magnitude rather than the correct
document-specific direction.

This is intended to validate a proposed definition of simplicity: two approximations
are equivalent in a task-relevant quotient when their error difference is nearly
invisible to a specified downstream consumer family. The definition is useful only
if it predicts finite composition, not merely local reconstruction.

## Largest remaining gaps

1. **No validated native interface metric.** We still do not know whether the local
   downstream-consequence quantities predict the finite mixed effect.
2. **Large residual CE.** `4.72714` nat of the strict causal ledger remains unnamed.
3. **Composition failure.** MLP0-C512 and MLP2 rank-512 replacements can be acceptable
   alone but have a reproducible `0.0074–0.0086` nat positive interaction together.
4. **Sparse consumer coverage.** We do not yet have enough functionally distinct,
   verified late consumers to identify a small early-layer causal state.
5. **No certified edit.** None of the 68 proposed extraction/removal/OOD actions has
   passed the full terminal gates.
6. **OOD transport is largely untested.** FineWeb document splits test fresh natural
   text, not semantic interventions or distribution shifts.
7. **Raw coefficient rank is not the answer.** All 18 native MLP polarization slices
   have numerical rank 1,152 and smooth rank-768 tails; the shipped layer-10 knee is
   reachable-state/consumer-conditioned rather than a raw tensor-rank transition.

## Candidate actions considered and pruned

- Another two-background local-MSE fit is redundant: it reduced the composition
  interaction by only about 13.2%, and its equal-compute advantage was unresolved.
- Raw SVD/HOSVD as a semantic explanation is low return: the coefficient spectrum has
  no layer-10 transition.
- Sparse document gating is unsupported: the interaction is diffuse over roughly
  108–118 effective documents.
- More shipped-program rank sweeps are useful for storage allocation but do not restore
  the missing live attention/residual interfaces, so they rank below native tests.
- A quotient/Hankel realization is premature until the consumer bank is sufficiently
  intervention-complete; otherwise its state equivalence is defined by too few outputs.
- A plan, audit, or unrun runner is not counted as a scientific result.

## Ranked next five

1. **Complete Rayleigh DESIGN → frozen predictor → HELDOUT.** Highest information gain,
   directly causal, prospectively falsifiable, moderate GPU cost, and not redundant.
2. **Conditional equal-price consequence-weighted MLP2 program.** Train this only if
   the held-out metric works; then test CE, composition, selective edits, and OOD. A
   failed metric kills this branch before an expensive fit.
3. **Direct mixed-functional model.** If tangent features work near zero but fail at
   $\alpha=1$, model the MLP0×MLP2 finite interaction directly, including RMSNorm and
   suffix curvature rather than pretending the whole model is locally quadratic.
4. **C512 × MLP1 × MLP2 factorial.** This locates whether MLP1 transports, amplifies,
   or compensates the early state mismatch and is the best independent early-layer
   entry point.
5. **Expand verified late consumers, then form a causal quotient.** Capitalization,
   numeric formatting, syntax, and entity-related consumers would provide independent
   readouts of early writes. A minimal-state/Hankel construction becomes meaningful
   only after withheld-consumer prediction is possible.

## Action executed in this review

- Independent audit verdict on v2 at `bfec160a`: **NO-GO**, outcome-blind; `117/117`
  transitive tests passed, but v1 absence lineage was incomplete.
- Implemented the exact requested closure at `8eb77232`:
  - explicit 13-path v1 absence set;
  - false-state embedded in the protected parent snapshot;
  - aggregate authority/failure/absence replay;
  - parameterized rejection test for every late v1 path.
- Focused recovery suite: `47/47` passed in `6.20 s`.
- Committed and pushed only the two recovery source files; the shared queue/log/cache
  files were left untouched.
- Fresh independent exact-source audit requested. The precise launch blocker is that
  audit GO, not data, RSPD, or GPU availability. The GPU is separately occupied by
  the already-running knee-localization job.

The expired eight-hour entry-point plan is not being silently reopened. Its preserved
status remains six measured negative cells, three scientifically pruned cells,
E4.1–E4.3 open, and a receipt-backed negative Family F result.
