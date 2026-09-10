# Fixed grammatical output ports on new cues, verbs and a closer control

10 September2026. Original handoff/pilot and unembedding-structure direction.
Freeze old e and all36 output sites. No fitting, gain, direction, layer or rank
change. The scalar-only edit predictor stays rejected; this tests the already
specified distributed native intervention on new rows.

Six16-row panels: A1 new verbs and might/were with final already; A2 new verbs,
singular subject and would/was with final perhaps; P might/could same bare-form
answer; G same varied contexts with he/they and runs/run agreement;
C old either/not control; R original A1 positive replay. New16 verbs are disjoint
from eight weight-construction and sixteen earlier test verbs. All lexical
choices and checks are frozen in gerund_fresh_transfer_v1.py; no outcome filter.
This is new-data and new-construction transfer relative to our experiment,
not proof of absence from pretraining. G is a deliberately closer control.

Pre-execution draft correction: varying the agreement answer with all16 verbs
failed the single-token check at 'washes'. No model was run. Keep every target
verb and use fixed single-token runs/run for G; this is one agreement readout
over16 contexts, not16 independent agreement readouts. No results chose this
correction, and the final rows are frozen before execution.

Use the shared ScalarWriteNetwork executor extracted from the previous frozen
runner: original g forward, same live output projection, same first-value cache,
same trained residual-mixing coefficients and final RMS/softcap. Per panel six
body calls: native base/donor, all-site base no-op, donor projection swap, zero
projection on base and donor. Total36forwards576seq, <=900managedseconds, no
backward pass. All native weights retained, no independent extraction/saving.

Predictions, scored as written:

* A instrument:36/576 and36hook visits per body; finite, unit e; reconstruction
  and all-port identities at prior bars; no-op maxlogit<=1e-3/rel<=1e-5; R
  per-row donor recovery and both zero CE changes replay old all-group results
  within1e-3. Tiny exact accumulation controls pass.
* B native capability: all96 pairs have correct answer/foil margins at both
  native endpoints and matching x0 within pair; A1/A2/G/C denominators positive.
  Failure is reported without dropping or replacing rows.
* C fresh target transfer: A/B and mean raw donor recovery>=.80 on BOTH A1/A2.
* D preservation: A/B; donor-swap mean absolute correct-token CE change<=.10
  on P/G/C AND absolute mean raw recovery<=.10 on G/C.
* E selective removal: A/B; mean correct-CE damage across both endpoints>=.10
  on BOTH targets AND mean absolute CE change<=.10 on BOTH G/C. P removal is
  reported but P still uses the grammatical distinction being removed.

Only C/D/E together support the broader interface claim. G failure narrows the
previous C-only selectivity result; do not redefine G as part of the target to
erase failure. C failure prohibits cue-specific refit or best-frame promotion.
Preserve old positive receipts under their original domains. No claim that
one scalar independently produces the changed complementary state.

CPU continuation: paired lexical-group intervals for new transfer, control and
zero-removal effects, including signed versus absolute preservation. No new
general reporting framework; append the result to the existing explanation.
