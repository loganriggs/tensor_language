# Does the calibration producer reread the current token through direct injection?

2026-09-10, original bilinear handoff/pilot authority. Frozen original q,w,Q,beta;
no fitting-direction refinement. Prior v185 concerns answer-axis response gates,
not this corpus-calibration quadratic. Its direct-token-small finding is known.

At the raw input a of MLP17, write T=B*x0, C=a-T. x0 is the native normalized
current-token embedding. Compute B exactly over real weights by starting at1
and applying B<-lambda0*B+lambda1 through blocks0..17. T denotes this direct
residual injection route; C contains ALL other native computation, including
indirect effects of the current token and accumulation-rounding differences.
This is not a decomposition into token-only information versus token-free context.

For rho²=mean(a²)+eps32,
q=beta+(T^TQT+2T^TQC+C^TQC)/rho². Define these three normalized terms tt,tc,cc.
Native-FP32 normalization and FP64 source splitting must agree within the
instrument tolerance. rho remains contextual even for the token numerator.
No whole-model polynomial expansion or rank approximation.

On the same42FW/16Pile evaluation rows, six output-projection interventions:
native q; remove q entirely; no_interaction=q-tc; token=beta+tt;
context=beta+cc; interaction=beta+tc. Retain original background g=h-qw,
replace its scalar, and recompute complete final RMS/softcap/vocabulary readout.
These are compiled-term edits mapped to MLP17 output projection edits, not
claims that deleting a native token injection yields the same downstream state.

Predictions:

* A instrument:18 observed bodyforwards70sequence instances, finite outputs;
  fullfold q vs native projection and source term sum vs fullfold each<=1e-5rel
  bothcorpora; reconstructed a normalized by native F.rms_norm matches captured
  u exactly; independent facade/native and first4rows eachcohort online tc
  removal vs formula <=1e-3abs and<=1e-5rel. Native lambda coefficient is read
  from checkpoint; positive CPU coefficient and quadratic controls are live.
* B interaction necessity: removing tc adds>=.005nat mean next-token CE on
  BOTH corpora. Report frequent/rare effects without changing this gate.
* C token-numerator sufficiency: q_token relative scalar error<=.10;
  centered full-vocabulary replacement error relative to original complete
  q-removal effect<=.10; absolute mean CE change<=.01nat, BOTH corpora.
* D context-numerator sufficiency: same three bars for beta+cc.
* E interaction-only sufficiency: same three bars for beta+tc.

Preserve every miss. A passing sufficiency arm is a conditional simplification
of this producer interface, not an independent token-to-output circuit. If none
passes, no mixture/gain/rank rescue. C's nativecontextualrho remains charged.
Held-out texts are reused; FineWeb document grouping unknown, Pile is corpus
shift only. All545902902 nativeparams retained.18bodyforwards70seq length256,
six finalreadout arms, <=900seconds managedruntime. No FIT forward is needed.
