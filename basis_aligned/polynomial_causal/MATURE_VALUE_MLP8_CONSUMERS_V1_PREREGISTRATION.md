# MLP8 branches with every downstream consumer live

Use unchanged32original/fronted worlds and full five-factor cubes. Split actual
MLP8 mixed output into new=Down(Lo*Rh+Lh*Ro) and inherited=Down(L0*Roh+Loh*R0).
Full uses Qoh of actual MLP8 output. At the MLP8 output subtract each branch;
recompute all later native operations, including residual carry, QK, all value
components, later attention/MLPs and decoding. No manual lambda transport is needed:
the native model applies it after the module output. No later output is clamped.

Reference: MATURE_VALUE_MLP8_ORIGIN_V1 partial value-path effects. This experiment
asks if those restricted effects agree with full-model source effects. Their difference
is not an identified other-consumer effect without additional joint interventions.
All previous source/branch failures retained; no new OOD rows or semantic labels.

4native/local captures+6actualsource arms perworld=320forwards5120seq,0fits,600s cap.
A instrument: existing factor/RMS/nativeC checks; partition closure<=1e-10 and MLP
weight sum vs actual mixed output<=1e-4 relative; native logits replay bound parent
maxabs<=1e-3 ANDrelative<=1e-5. Incoming module output bitwise equal to captured native;
shared tensor-subtraction identity, active edit and error-cleanup controls. Finite,
exactcounts, all hooks restored. All545902902nativeweights retained, saving0.

B value-path dominance: EACH full/new/inherited isolated-path effect differs from
its full-model counterpart <=.10 relative to that full-model effect, BOTH Q correct
margin and centered Q three-readers, EVERY world. Zero denominators fail.
C new-branch fidelity: new full-model effect matches total full-model MLP8 effect
<=.10 on bothreadouts, and total effect norm>=.10 of native mixed effect in both.
D inherited fidelity: same for inherited. E composition: complete full-model effect
versus sum of new+inherited full-model effects <=.10 on full correct-margin and
centered three-reader tables. Every world must pass; retain all failures and signed
cancellation. No smaller-error winner. B failure rejects value-path dominance and
requires accounting for other consumers/interactions; it does not establish that
MLP8 lacks a value role or identify a specific alternative reader.
