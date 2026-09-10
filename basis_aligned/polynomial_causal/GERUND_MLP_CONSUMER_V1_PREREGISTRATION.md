# Fixed grammatical scalar reads and complementary MLP responses

10 September 2026. Follows fresh-transfer af770179c and original handoff/pilot.
The module dossiers and v184/v185 already contain normalized bilinear response
expansions. This experiment tests a new causal use of that expansion, not a new
algebraic discovery. No rank/direction/site/threshold rescue of failed selectivity.

At each MLP's final semantic position, let u be its actual post-RMS input,
s=e^T u and z=u-se. Save s0 from the natural base endpoint. To block changes in
this scalar read, evaluate the MLP at u'=u+(s0-s)e. Its other normalized input
coordinates remain live. Do NOT renormalize u': this is an intervention after
RMS, not a claim about a naturally normalized raw residual.

For delta=s0-s and m(u)=D[(Lu)*(Ru)]+b, the exact real-arithmetic difference is

    m(u')-m(u) = D[delta*((Le)*(Ru)+(Lu)*(Re)) + delta^2*(Le)*(Re)].

Add this folded difference to the native output before the existing all36-port
e swap/zero. The latter prescribes the output scalar, so only complementary
outputs can differ between blocked and live arms. Attention routing, other
positions, MLP normalized input complements and all native weights stay live.
This does not block RMS-induced changes to the complement or attention consumers.

For any output reader v, its scalar-sensitive context reader is

    k_v = L^T[(D^T v)*(Re)] + R^T[(D^T v)*(Le)] = 2 Q(v)e.

Report the per-layer k_e versus k_g cosines, where g is the fixed normalized
U_runs-U_run agreement contrast from the existing G rows. These are weight
diagnostics, not a selection or a separate causal claim. Nonzero or similar
maps alone cannot identify shared semantic circuits.

Use only already-open fresh A1/A2/G/C, 16 rows each. Seven bodies each:
native base, native donor, all-port donor swap, all-port base zero, blocked
donor swap, blocked base zero, and blocked no-op with original base output
scalars. Total28 forwards448 sequences, <=900 managed seconds, no gradients.
Save natural base post-RMS s0 before any consumer block. No donor-input fit.

Predictions:

* A instrument: exact28/448 counts; all36 output hooks per body; all18 consumer
  hooks per body; all64 native pairs capable; old swap and base-zero CE replay
  maxabs<=1e-3; no-op logit maxabs<=1e-3 and rel<=1e-5; existing residual/readout
  and all-port scalar checks pass. Direct versus folded MLP evaluation has
  relative error<=1e-5. FP64 toy delta/reader identities maxabs<=1e-10.
* B live target effect: A and original target swap recovery>=.80 both frames.
* C essential scalar-read consumers: A/B and blocking reduces target raw
  recovery by>=.20 on BOTH frames. Failure rejects that magnitude of necessity
  for this all-MLP post-RMS edge intervention, not all scalar consumers.
* D conditional scalar closure: A/B and blocked donor-swap centered full-vocab
  effect differs by<=.10 relative norm from the frozen-background final scalar
  predictor on BOTH targets. This tests whether blocking these specific reads
  closes the previously missing response; no promotion if only C holds.
* E agreement-damage mediation: A and G live base-zero CE damage>=.10, and
  blocking reduces its mean signed CE damage by>=50%. Also report absolute CE
  and per-row effects. This is mediation of a failed control, not reclassification
  of G as a target or a new selective-removal pass.

Frozen predictor hhat=hbase+e*e^T(hdonor-hbase), then actual RMS/U/softcap.
Compare both live and blocked swap to that same predictor. Report all controls
and negative results. Literal price retains every native parameter and charges
extra folded projections; no independent producers or storage saving.
CPU continuation: paired intervals for recovery loss, closure, and agreement
damage reduction. Append findings to the existing gerund explanation/dossiers.
