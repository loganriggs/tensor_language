# Does a shared frequency reference resolve the split-fit failure?

2026-09-10, frozen before native execution. Parent: calibration_stability_context_v1,
result commit aacb26913. The original bilinear handoff/pilot remains authority.

The parent changed both FIT examples and frequency labels: each24-row half
computed its own log(1+next-token-count) target. This successor changes exactly
one scientific factor: BOTH covariance fits now use the frozen original48-row
frequency histogram. They still use their own24-row MLP outputs and center
their own label values. The full48-row fit is unchanged. Frequency counts are
shared information, so this is controlled reference sensitivity, not fully
independent identification or new document/corpus generalization.

Reuse the parent's complete executor, original producer, rows, six arms,
numerical tolerances, path controls, observed32forwards126sequence count,
and A/B/C gates. Output/binding/axes names change to this experiment. No new
dimension, gain, fit-selection or evaluation change. The mean/donor arms
repeat identical work as controls, not additional independent discoveries.

Opposing predictions: if changing the target-frequency reference explains
the failure sufficiently, both new axes have centered full-vocabulary removal
effect error <=.20 versus the original on both corpora, mutual |cosine|>=.90,
and retain rare CE damage >=.10 / common CE improvement >=.02. This is pred_b
with the parent's unchanged gates. If pred_b fails, reference mismatch alone
is insufficient; close this repair and stop scalar-axis fitting refinements.
No choosing the better half or loosening the threshold.

An added pred_d replay requires native, original removal, mean replacement
and donor replacement common/rare/all mean CEs to equal the parent within
1e-12 on both reused cohorts. This isolates the changed fits from execution
drift. A or D failure makes the instrument invalid, not a scientific null.
C repeats the parent context-pairing gate and must be labelled replication.

Compare own-reference and shared-reference directions, effect errors and
class CE changes after the outcome; report a failure as written. No semantic
donor-task transfer or all-four-property claim is licensed by this diagnostic.
All545902902 nativeparameters remain. 32bodyforwards126seq length256, six
readout arms, <=900seconds managedruntime; no further cached output allocation.
