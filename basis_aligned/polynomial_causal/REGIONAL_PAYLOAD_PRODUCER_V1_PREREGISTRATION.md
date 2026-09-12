# Regional payload producer mediation through MLP16

Use frozen32V2regional cue prompts and the identified all-source block. No fitting.
At every source position split the child tuple as u=(m+p+e)/rho+f:
m is folded MLP16quadratic+bias; p is lambda17[0]times pre-MLP16residual read;
e is lambda17[1]times embedding re-entry read; rho is actual pre-attention17RMS;
f is the first-attention-stream read. Keep parent, QK gates, query writers and
recipient downstream background fixed. Pair donor/recipient by equal positions.

A: compiled MLP16numerator versus native MLP16output read<=1e-5relative; sum of
parts versus native child tuple<=1e-5; all-five-port donor swap versus previous
child-only donor write<=1e-5. B:A and eachfamily all-port transplant transfer
>=10%native cue gap with>=5/8positive pairs. C:A/B and MLP16numerator-only transfer
>=50%all-port transfer eachfamily, unrelated contrast meanabs<=.5regional meanabs.
This is a registered producer-dominance hypothesis, not an assumption. Preserve
Cfail if preceding residual or normalization dominates; report all5single-port
transfers. Donor MLP numerator uses recipient divisor for its isolated swap.

7bodyforwards32sequences4-12tokens;7suffix arms (native,5single,all),180seconds.
Capture native MLP16input/output and pre-attention17residual for exact controls;
retain all normalization dependencies, no inverse-RMS constant approximation.
This is conditional upstream tracing, not independent text-to-answer extraction
or a whole-model replacement. Existing parent1producer algebra is reused.
