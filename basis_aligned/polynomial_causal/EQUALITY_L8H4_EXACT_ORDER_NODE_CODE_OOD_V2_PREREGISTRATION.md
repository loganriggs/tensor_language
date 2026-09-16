# Equality L8H4 exact-order node code-OOD V2

## Instrument repair and question

V1's projected-payload executor is a valid null.  It preserved the A8-only OOD
behavior (`1.17664` versus oracle `1.17756`) and passed the bilinear identity,
but it missed the deployed equality term by `.00237` relative L2 because it
projected values before summing.  That reorders BF16 operations and caused
`.0162--.0170` relative downstream logit error.  Do not change the V1 result or
thresholds.

Test a separately named V2 executor in the model's exact order:

`output_projection(bmm(score * equality_support, raw_head_payload))`.

Use the native L8H4 output-projection slice as an explicit parameter port.  The
package adds zero learned parameters.  Repeat V1's package removal, frozen
A8-only installation, and live bilinear composition tests on all 192 code-OOD
documents.

## Predictions

- **A — exact executor replay:** term, removal-logit/MLP9, and
  installation-logit/MLP9 relative L2 errors are each at most `2e-6`.
- **B — behavioral identity:** package and oracle recovery differ by at most
  `.001` aggregate and `.005` in every registered cell and half.
- **C — removal:** package subtraction reproduces the live positive removal
  effect in aggregate and both halves.
- **D — OOD installation:** package recovery is at least `.85`, every cell and
  half exceeds `.70`, and noncopy mean damage is at most `.01` nat.
- **E — extraction:** the manifest and execution receipt both report zero new
  learned parameters.
- **F — compositionality:** the FP32 live-port two-score/two-payload expansion
  has relative L2 error at most `2e-6`.

Passing establishes an exact, removable, OOD-installable, compositionally
reusable edge executor.  It does not repair the earlier natural-to-code scalar
calibration null or extract the score/payload producers.

## Price

One checkpoint load; 192 frozen code-OOD documents; five forwards per
four-document batch, exactly 240 forwards; no fits, gradients, new text, or
parameter updates.
