# Does context matching survive explicit scalar units?

2026-09-10. Follow the failed shared-frequency stability repair, preserving that
failure and closing scalar-axis fitting refinements. Use the ORIGINAL frozen w,
Q and beta for this intervention test. No new coordinate or gain is selected.

The prior raw donor test could reflect scalar/background scale mismatch. Let
g=h-qw be the recipient background and R=sqrt(mean(g²))>0. Define tau=q/R.
The exact readout is

    30*tanh(U(g/R+tau*w)/(30*sqrt(mean((g/R+tau*w)^2)+eps32/R^2))).

The native epsilon must change units too. Native background computation remains
opaque and fully charged. A zero/nonfinite R invalidates this specified interface;
do not silently clip it or treat an excluded case as a passing evaluation.

Keep the original48FIT/42FW/16Pile rows and cyclic same-position donor rule.
Compute mean tau from all48FIT rows only. Two new arms:

* mean_tau: q_new=R_recipient*mean_FIT(tau).
* donor_tau: q_new=R_recipient*q_donor/R_donor.

All donor variables are computed from the original native donor. No evaluation
next-token labels enter either producer. Repeat native/original removal/raw mean/
raw donor arms from the first stability executor as exact execution controls.
Its two own-reference split-axis removal arms are unchanged archival replays;
do not use their outcomes to fit or select a direction.

Predictions:

* A instrument:34 observed bodyforwards134sequences; all outputs/R finite and
  allR>0; fullFIT original-axis replay <=1e-10rel, original scalar <=1e-5rel;
  native formula and online raw/native/compiled and new adjusted-donor bridges
  <=1e-3abs and <=1e-5rel. Add one adjusted-donor online forward on the first4
  rows of eachcohort to the parent's32forwards126seq.
* B residual context pairing: mean_tau AND donor_tau each add >=.005nat mean
  next-token CE on BOTH corpora. Passing supports useful context dependence
  beyond this particular background-scale factor; it does not name semantics.
* C scale accounts for most raw donor damage: on BOTH corpora, raw donor CE
  damage >0 and adjusted-donor CE damage <=.10 times raw donor damage.
  B/C are not mutually exclusive: a small residual may still be useful.
* D unchanged replay: native/original removal/raw mean/raw donor common/rare/all
  mean CEs equal the original stability result within1e-12. A orDfailure is
  invalid instrumentation; all scientific misses remain misses.

If B fails, stop citing raw donor damage as evidence against a mostly constant
dimensionless coefficient; retain its validity as a raw-coordinate intervention.
If B passes, the specific scale-only alternative is insufficient. Either way,
the original fitted-coordinate stability failure remains open, with no rescue.
This is conditional manipulation evidence, not semantic interchange or extraction
from tokens. All545902902 nativeparameters retained.34forwards134seq length256,
eight finalreadout arms over58rows, <=900seconds managedruntime. CPU exact-path,
same-tau/different-scale, epsilon-rescaling and zero-domain controls passed.
