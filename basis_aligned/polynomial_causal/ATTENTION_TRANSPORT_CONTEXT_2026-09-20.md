# Context-dependent transport, then an exact interaction census

The registered transport test passes instrumentation and integrated-response checks, but the joint-state reader fails the all-cell10%number gate:63/64pass, worst14.237%. Crucially all32beside_subject cells pass, worst0.1714%, versus baseline-reader worst251.63%. That template contains all13previous composition failures. Baseline-context mismatch explains the large reader errors there much better than finite curvature along the small local removal.

Midpoint readers pass64/64number cells, worst0.03997%. Three-node integration passes its tighter all-output1e-4relative or1e-8absolute gate; maximum absolute error norm across four contrasts is5.264e-14. The residual map is smooth on these measured paths; this is not a proof of universal quadrature accuracy. The single joint-reader miss is unlike_nearby at full radius. Preserve this failed registered prediction.

Every estimate still uses the actual joint native state and a native suffix derivative. Midpoint also needs the exact removed write; integration requires three derivative evaluations. Counts12prefix232suffix68attention16baseline reverse calls128directional JVPs,16.95s execution. This is a diagnostic, not a simpler extracted program. Exact first-edge removal still fails its0/13sufficiency test.

## Executed next step: separate generated from incoming interactions

For quartet x0,xs,xt,xst, define xadd=xs+xt-x0 and I=xst-xadd. For a residual write W:

    generated = W(xadd)-W(xs)-W(xt)+W(x0)
    transported = W(xst)-W(xadd)
    I_next = I + generated + transported

This exact identity prevents incoming state interactions from being counted again as newly generated ones. It also retains nonlinear dependence of transport on I. It is not a unique causal attribution: the chosen additive reference and intervention path matter. Fixed cache/backgrounds must be identical across the quartet. Residual lambda scaling multiplies incoming I by lambda0 when x0 background is fixed.

Implemented finite_mixed_residual_split.py and executed planted quadratic, linear, zero-incoming and RMS-normalized controls. Closure7.11e-15, analytic quadratic generation2.66e-15, transport5.33e-15. A native layer census remains to be run. It should report generated/transported terms and downstream relevance separately; large state norms alone cannot identify the circuit. This path targets the missing composition behavior rather than improving another isolated ray fit.

Receipts: ATTENTION_TRANSPORT_CONTEXT_AUDIT_V1.json; ../bilinear_quotient/circuits/followups/attention_edge_transport_native_v1_result.json; FINITE_MIXED_RESIDUAL_SPLIT_CONTROL.json.
