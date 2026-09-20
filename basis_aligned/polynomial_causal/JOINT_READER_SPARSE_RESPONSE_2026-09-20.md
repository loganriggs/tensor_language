# Joint readers and sparse shared interactions

The target-only eight-coordinate frame predicted number effects but failed two
preservation-reader predictions. v665 changes the calibration observable set to
four equally weighted logit-margin contrasts: is/are, can/will, may/might and
should/could. Calibration remains the same256 rows; width8, response snapshots,
normalization and native algebra remain unchanged. The rectangular reader-response
snapshot contraction selects a common space. No evaluation rows enter that basis.

v666 passes the opened syntax panel: worst-cell number-effect error5.97%, modal
prediction errors1.38–1.60%. v667 freezes new nouns and object-relative/adverb
structures before execution. It passes all eight cells: number-effect error
0.64–4.57%, nativeaccuracy100%, modalprediction1.71–3.36%, nativecollateral1.59–3.53%,
and target effect15.23 times median of eight equal-L2 random site edits.
These support a conditional suffix model for the declared four readers. Native
baseline contexts, first-layer values and initial post11 response remain required;
this does not establish token-input closure, arbitrary-reader fidelity or broad
cross-task reuse. Historical reuse of vocabulary is not excluded.

## What sparsity means in this experiment

Each of six MLP response cores has36 unordered coordinate pairs, shared across
its eight outputs. Deleting a pair deletes the entire eight-output coefficient
vector and its numerator product. This is interaction sparsity at fixed feature
width, not sparse leaf projections or fewer learned input features. Mixed
background terms and exact normalization stay explicit. Attention stays dynamic.

Let W contain the input encoder columns and V be the output decoder. For the
stored unordered-pair coefficient vector c_pq, the native-space tensor atom has
Frobenius norm

    ||V c_pq|| * sqrt((||W_p||^2 ||W_q||^2 + (W_p^T W_q)^2)/2).

This includes geometry on both sides. It is invariant to invertible diagonal
latent rescaling/sign changes, as independently tested against explicitly formed
tensors. It is not invariant to arbitrary basis rotations and is not a causal
importance score. The three matched random supports test whether this weight-only
ranking helps behavioral prediction.

## v668 opened-panel screen

| Pair support per stage | Worst-cell number error | Largest modal prediction error | Gates |
| --- | --- | --- | --- |
| Dense36, v667 | 4.57% | 3.36% | pass |
| Atom-norm18 | 7.66% | 4.15% | pass |
| Atom-norm9 | 28.63% | 4.09% | fail |
| Zero quadratic numerator | 22.52% | 4.01% | fail |
| Random18, seed668 | 17.80% | 3.27% | fail |
| Random18, seed669 | 9.28% | 4.04% | pass |
| Random18, seed670 | 26.10% | 3.95% | fail |

The half-support arm beats the median random error, but one random support also
passes. This does not uniquely identify the chosen interactions. The quarter-
support and zero baselines show a nontrivial quadratic numerator is needed for
this frozen representation; they do not prove a lower bound for every basis or
arithmetic circuit. Planted sparse recovery, independent atom norms, diagonal
gauges and dense-versus-sparse runtime execution pass CPU tests.

The half-support MLP consumer has1718 floating values plus216 integer indices,
versus2582 floating values for dense. It uses108 MLP-numerator pair products per
token instead of216. Six dense Gram normalization forms remain: these numerator
counts are NOT the full quadratic-computation count. Native context generation,
577536 producer values and compiled attention storage remain chargeable. Whole-
model savings and a fully simple circuit are not claimed.

v669 preregisters irregular-noun transfer for the unchanged sparse supports,
width and criteria. No row filtering, support refit or threshold adjustment.

## v669 prospective irregular nouns

Top18 predicts native effects within5.09% and modal changes within3.42% of target
norm. Dense worst-cell error is5.53%. However native accuracy is83.3% in one cell,
below the90% capability gate; sparse-screen prediction booleans do not include
that gate. Thus this is fidelity on a panel with a native capability limitation,
not an unqualified semantic OOD success. No rows are removed. Native collateral
is4.14–6.88% and target/random is11.48x. Zero quadratic and all three random18 arms
also pass on this panel; top9 fails10.94%. The need for quadratic terms depends on
context, and a single passing sparse support does not uniquely identify a circuit.

v670 exports the complete conditional attention/MLP consumer with prepared contexts
for eight fixed rows. This is extraction at an explicit response boundary; native
context and initial-response generators remain chargeable external dependencies.

## Portable consumer and gauge audit

The v670 artifact is15,805,403 bytes. An independent CPU-only process replays eight fixed examples without a checkpoint or CUDA; target and control logits match within1.78e-15. Runtime contains1718 floats and216 integer support indices; the saved prepared-case tensors contain1,943,258 values, including contexts and references. These are substantial explicit dependencies, not free inputs. Signed diagonal rescaling of all seven latent boundaries, including attention scores/values, RMS geometry, sparse cores and readouts, preserves outputs within1.78e-15. This verifies coordinate handling, not semantic uniqueness.

[Artifact](../bilinear_quotient/circuits/followups/subject_conditional_dag_v670.pt), [CPU replay](CONDITIONAL_DAG_CPU_REPLAY_2026-09-20.json), [gauge audit](CONDITIONAL_DAG_GAUGE_AUDIT_2026-09-20.json).

Native capability exception retained verbatim: [{"text": "The mice that the tooth saw", "native_margin": -1.0863184928894043, "expected_token": "are"}]. No retrospective exclusion.

Next scientific question is reuse across independently specified source writes, rather than just readout sharing. Split the five pre-L11 source ports into embedding/early and middle/MLP groups, construct their post-L11 response writes, and test the shared suffix on each and their explicitly defined post-L11 superposition. Keep the difference between pre-L11 joint edits and post-L11 superposition explicit: L11 is nonlinear. No result for this test is yet claimed.
