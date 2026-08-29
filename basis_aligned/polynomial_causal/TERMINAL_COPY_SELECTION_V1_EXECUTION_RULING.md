# Terminal-copy selection v1 execution ruling

Status: **prospective, outcome-blind ruling; not execution authority**.

This document resolves conflicts between
`TERMINAL_COPY_INDUCTION_V1_PREREGISTRATION.md` and the later
`TERMINAL_COPY_INDUCTION_V1_SCREENING_AMENDMENT.md`. For the first attention-only
copy screen, the screening amendment governs every conflict. The original remains
historical context and governs only statements the amendment does not change.

Concretely, this run has exactly eight candidates: the six named single heads, the
registered L5/L7/L8 four-head group, and the L13/L14 late pair. It does not include
the original full-six, random, shuffle, no-op, or late-MLP arms. All MLPs remain
native. The nearest-prior-query label and five-coordinate matching stratum in the
amendment replace the older any-witness label and four-coordinate stratum. Selection
uses the shared 24-coordinate 10,000-draw lower confidence band; it does not select by
price and does not require the original absolute matched-negative 0.01-nat gate.
Synthetic crossover is descriptive and cannot select a candidate.

For candidate (a) and cell (c), the estimator is the pooled within-input loss
difference

$$
\widehat\tau_{a,c}=
\frac{\sum_d\sum_{p\in c_d}
  [\ell_{a,dp}-\ell_{\mathrm{native},dp}]}
 {\sum_d |c_d|}.
$$

The three promotive coordinates are

$$
\tau_{a,+},\qquad
S_a=\tau_{a,+}-\tau_{a,-},\qquad
C_a=0.01-\tau_{a,\mathrm{off}}.
$$

A candidate passes only when all three simultaneous lower bounds are positive. Among
passers, choose the greatest specificity lower bound and then lexicographic name.
This establishes differential specificity, not absolute matched-negative safety;
absolute \(\tau_-\), off-target damage, CE, accuracy, and KL must all be reported.

The schema-only selection-container access recorded in
`TERMINAL_COPY_SELECTION_INPUT_EXPOSURE_ERRATUM.md` did not reveal token or mask values
or any model outcome. The selection role remains eligible only if the authority binds
that erratum and truthfully disclaims pristine container secrecy.

The completed v3 fit receipt is a prerequisite but explicitly does not authorize
candidate selection. A separate independently audited selection authority must
semantic-replay the exact v3 authority/bank/result/manifest/receipt, license that bank
for this selection role, and bind the exact row receipt, selection payload, fit
frequencies, adapter result/receipt, checkpoint, source closure, call census,
protected paths, outputs, and lock before any further selection payload or model load.

The natural phase is exactly 48 four-document batches. Every batch uses one shared
native forward and all eight live sequential candidates. The production lifecycle
must retain closures and require identical native sufficient statistics across all
candidates. It must reconstruct masks independently from receipt-bound rows and fit
frequencies, never deserialize final/OOD roles before a passer, and publish a mutually
exclusive passer-or-negative receipt last after semantic bootstrap replay.

A pass identifies the total effect of the registered mean intervention. It does not
prove that a head represents copy, that a group is an interaction-resolved path, or
that the intervention is an extracted standalone program. A negative falsifies this
eight-candidate mean-ablation bank, not copy circuitry generally.
