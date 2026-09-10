# Contextual value: local attention reading versus later reading

Use all 16 worlds / 512 prefixes from frozen THIRD_NOUN_VALUE_TRANSFER_V1_ROWS.
This is a new intervention on opened cases, not a fresh OOD test. No fitting,
head/position/rank selection, gain tuning, or removal of failed worlds.

Reuse live_value_factorial_executor_v1 for baseline and the frozen local-value
removal. Capture the full layer 9 attention output, at every position, for both.
Let D=A_native-A_edited. Its input correction is pure oh, but contextual attention
routing can make D contain other factors. Split Dm=Poh D and Ds=D-Dm.
Dm contains all terms involving both object number o and human category h;
Ds is its orthogonal complement on the complete sentence-factor table.

Run three additional arms on the otherwise native model: subtract D, Dm, or Ds
from the layer 9 attention output; preserve the tuple's shared first value and
recompute the entire native suffix. Full-D replay is a positive control against
the original local-value removal. No final-position-only patch: all 10 positions
must be replaced. Baseline, local-value removal, full-D, Dm, Ds total 160 forwards
and 2560 sequence instances, zero fits and offline decoder batches; 600s cap.

A instrument: bound files, finite logits, inherited executor controls, hook
cleanup; exact counts. All-world native/local-edited three-logit grids replay the
parent with BOTH maxabs<=1e-3 and relative Frobenius<=1e-5. Full-D arm matches
local-edited outputs under the same two bounds. Incoming attention output in
each patch equals captured baseline bitwise; shared first-value payload unchanged.
Dm+Ds closure and Q Dm=Dm, Q Ds=0 relative errors <=1e-10 in FP64 before native
casting. Save actual installed FP32 deltas; mixed purity error relative<=1e-4,
full delta early-position norm/full norm<=1e-6. Zero scientific effect fails
scientific gates rather than mechanical validity. Existing parent effects are live.

All following gates require EVERY world independently. Let E_F=z0-z_full,
E_M=z0-z_mixed, E_S=z0-z_spill; use actual softcapped answer logits.

- B mixed-branch fidelity: ||Q(E_F-E_M)||/||Q E_F||<=.10 for the correct answer
  margin (themselves-himself or themselves-herself). A zero denominator fails.
- C mixed-branch factor selectivity: ||(I-Q) E_M||/||Q E_M||<=.25 for that margin.
  Also report signed projection on the natural interaction; no small-effect
  branch is promoted from C alone. Zero denominator fails.
- D mixed-branch gender control: ||Q deltaG||/||Q deltaN||<=.25, with
  N=z_t-(z_m+z_f)/2, G=z_m-z_f and nonzero denominator.
- E composition: ||E_F-E_M-E_S||/||E_F||<=.10, separately for the correct margin
  and the centered three-reader output vector, using full tables (not just Q).
  Zero denominator fails. This tests composition of these two specified edits,
  not unrelated-task composition or an independently extracted program.

Opposing predictions: if Dm is faithful, selective and composable, the locally
generated nonmixed write Ds is a removable nuisance on this bank. If Dm itself
fails factor selectivity, downstream processing can generate off-target factors
from a pure mixed attention write. If fidelity or composition fails, both pieces
or their interaction remain necessary; do not infer a clean repair from one gate.
Report Ds effects, Dm effects, local raw write energy, and all failures. No change
to the completed parent verdict. All 545902902 native parameters remain, savings0;
all branch states still require native counterfactual inputs.
