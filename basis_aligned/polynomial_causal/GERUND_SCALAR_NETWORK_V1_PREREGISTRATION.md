# Does the grammatical scalar support a closed network of write edits?

10 September 2026. Original handoff and two unembedding views. Freeze previous
gerund e, all A1/A2/P/C rows and native checkpoint. No direction, layer, rank,
mean or gain selection. This tests a global candidate interface, not a newly
identified native module. Prior v185 already established lambda-weighted writer
accounting and the danger of mistaking direct attribution for live causation.
Read grouped front/transition/deep/readout and attention dossiers. New per-module
findings must be checked against their dossier before pursuing a module claim.

For block l, x_(l+1)=a_l*x_l+b_l*x0+A_l+M_l. Actual lambdas are scalar learned
coefficients, not assumed one. At the semantic position define

    beta_l = product(a_j for j>l)
    gamma = product(a_j for all j) + sum_l beta_l*b_l
    h_final = gamma*x0 + sum_l beta_l*(A_l+M_l)
    e^T h_final = gamma*e^T x0 + sum_l beta_l*e^T(A_l+M_l).

This is exact real arithmetic; floating-point bridges are measured. The same
last token within each cue pair makes x0 and its entire direct re-entry term
identical. It does not make routed first-value contributions identical.

Capture each full attention-output and MLP-output vector at the semantic
position. For groups G=attention, MLP, all, intervene at EVERY eligible module:

* donor: y_live <- y_live + e*(e^T y_native_donor - e^T y_live);
* zero: y_live <- y_live - e*(e^T y_live).

These are live replacements; later native computation changes in response.
Attention's separate first-value cache is retained, and input positions other
than the final semantic position remain native. No full-module/whole-sequence
swap or independently extracted producer is implied.

Compare each live edit to the frozen-background prediction

    h_hat = h_native + e*sum_(l,kind in G) beta_l*delta_s_native(l,kind),

then execute the actual final RMS, unembedding and softcap. Donor delta_s is
donor-minus-base; zero delta_s is minus the corresponding native scalar.
This prediction ignores changes to later non-e writes, so it can fail even
though the scalar accumulation identity holds exactly on the changed run.

For all-group donor swaps, e^T h_live must equal e^T h_donor up to numerical
error, because every output e-coordinate is prescribed and x0 matches. This
is an algebraic instrument control, NOT evidence that the remaining state or
full vocabulary is donor-equivalent. All-group zero leaves gamma*e^T x0.

Per panel: native base/donor (2), all-site native-base scalar no-op (1), three
group donor swaps (3), three group zero edits on EACH native endpoint (6).
Total48 body forwards768 sequences; zero backward passes and no optimizer.
All545902902 nativeparameters retained plus the existing e artifact and18
propagation coefficients. Managed execution <=900seconds. Hooks visit36 sites
per body; report counts and every source scalar rather than choosing best sites.

Predicates:

* A instrument:48/768 counts, finite, unit e<=1e-5; reconstructed final state
  relL2<=1e-5 on every run and reconstructed readout maxabs<=1e-3/relL2<=1e-5;
  no-op readout same bars; all-port donor/zero scalar error
  <=max(1e-3+1e-5*abs(expected),1e-6*norm(live_final_state)), because a nearly
  zero coordinate is obtained by cancelling large FP32 residual entries;
  report absolute errors too. Parent scalar-final per-row recovery and zero CE
  replay maxabs<=1e-3. Tiny FP64 accumulation controls<=1e-10.
* B capability: all64 pairs both native answer/foil margins correct, positive
  A1/A2/C cue denominators, paired x0 vectors exactly equal.
* C edit-prediction closure: A/B and frozen-background versus live centered
  full-vocabulary effect error<=.10 for ALL attention/MLP/all donor and zero
  base/donor arms on BOTH target panels. Denominator is each live effect norm,
  which must exceed1e-4. Report per-arm errors on P/C without promoting tiny
  control effects to reliable ratios. Failure rejects this fixed scalar-only
  prediction model, not every possible larger or context-dependent circuit.
* D distributed scalar sufficiency: A/B; all-group donor raw recovery>=.80 on
  BOTH targets; P/C meanabsCE<=.10 and |C mean raw recovery|<=.10. This retains
  the parent sufficiency bar; no partial-effect threshold substitution.
* E selective distributed zero removal: A/B; all-group zero mean correct-CE
  damage across both endpoints>=.10 on both targets; C meanabsCE<=.10.

Report attention and MLP group effects, weighted native contributions, and
state feedback without ranking a winner into a circuit. These are current
conditional writes; their own production remains opaque. If closure fails,
quantify the missing changed-state contribution before attempting a new model.
CPU successor: paired uncertainty on closure/removal/recovery and signed native
attention/MLP scalar accounting. No best-site/gain/mean rescue.
