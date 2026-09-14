# Full-value even interaction: native selective-removal screen

Freeze the original key projection and `EVEN_KEY_VALUE_BANK_V1_PROGRAM.pt`.
No fitting, quantization or changed native weights. The new component includes
all 128 current/first value channels and the full head9.8 output matrix. Native
prefixes, RMS, current and first-value inputs, and downstream suffix are retained.

Use all 72 regional and 32 unrelated natural rows from the existing
`SCALAR_JOINT_KEY_CONFIRMATION_V1_ROWS.json`. These are reused development rows.
Eight regional arms: native; remove even; remove odd; remove full head9.8;
replace native head by even+odd; remove prior selected scalar component;
donor even from aligned partner; self-donor even. Natural rows use the first six.
Donor amplitudes are not refitted. Swap the full computed even write at aligned
positions; earlier city-pair semantics and row validation remain in force.

- A: every live even+odd/native-head write relative error <=1e-5;
  recomposition/native and previous native score replay <=1e-5 relative;
  self-donor <=1e-4 relative (absolute <=1e-6 at tiny norm). Regional native
  mean contrast >=0.2 and >=10/12 positive pairs, each family; natural native
  CE <=5, newline/comma margin >=0.2 and >=12/16 positive rows, each half.
- B: even removal covers >=50% native cue contrast with >=10/12 positive pairs;
  even donor transfer >=50% and >=20/24 positive rows; unrelated/control-to-target
  mean absolute effect ratio <=0.5 for both. All three regional families required.
- C: even removal changes unrelated newline CE by mean absolute <=0.02 and maximum
  absolute <=0.1 in each half. Lower CE remains an improvement, but absolute
  preservation does not waive negative changes.
- D: prior scalar-removal arm matches historical head9-only removal scores within
  1e-5 relative. Full-head sensitivity and other arms are reported descriptively.

The full-head write control captures native pre-output-projection values, including
the signed current/first mixture. Sum replay is checked before suffix execution.
Remove-even plus remove-odd need not have additive downstream effects; report
their interaction rather than assuming additivity. Opposing prediction: extending
the old scalar component introduces broadly acting value directions and fails
selectivity or unrelated preservation, despite exact routing reuse.

Price: 768 complete body forwards, 104 prefixes, <=247 tokens, 300 seconds,
managed GPU lane1 only. One CPU device-backend replay precedes the hash-bound
dry run and enqueue. All external model parameters and the 4.26 MB conditional
program remain charged; no whole-model compression or new OOD claim.
