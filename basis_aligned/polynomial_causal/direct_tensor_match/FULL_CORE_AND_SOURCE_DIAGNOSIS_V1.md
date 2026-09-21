# Full-core closure does not repair component fidelity

21 September 2026. Two completed CPU diagnostics distinguish a graph restriction from downstream error concentration. Both concern the existing six-read target on448 opened native states, not a new circuit or fresh validation.

With the final learned352-dimensional input spans fixed, project each quadratic form onto the complete span and compile the unrestricted pair core. Covariance error improves7.8931% to7.6482%; original-coordinate error58.6696%. Both coefficient bars and Jacobian bars pass. Component errors2.1338%,2.1077%,10.6753% still fail acceptance. Actual independent-pair compilation uses1056 nonlinear products and1219680 source multiplications, failing the original1064448 ceiling. The original-coordinate full-core fit also fails fidelity. Thus omitted within-span cross terms explain part of the coefficient gap, but removing that restriction is insufficient for component fidelity. This is not a lower bound over other input spans or functional objectives.

Next, replace each approximate source read with its exact native counterpart while keeping h and RMS fixed. For the primary covariance graph:

| Component | Original approximation error | Exact first read | Exact second read |
|---|---:|---:|---:|
| 1 |2.2181%|2.8358%|.8749%|
| 2 |2.2940%|2.3587%|1.0560%|
| 3 |10.6723%|10.4178%|4.7299%|

The second read is the more consequential approximation in each component. Replacing the first can worsen fidelity because errors cancel; it would be wrong to infer that each individual source approximation improves the composed function independently. The exact interaction identity -delta_a*delta_b/(2*s^2) matches numerical differences below1e-10 normalized error. Component3 interaction magnitude is.6260%; exact-both is zero against recomputed native RMS. Cached-truth/RMS drift is recorded separately.

This points toward asymmetric source corrections rather than simply adding pair capacity according to total coefficient energy. Prior equal-pair capacity allocation put all14 additional directions into component1, improving the coefficient objective but missing component fidelity. Any source-directed candidate must retain both coefficient geometries, all component/Jacobian requirements, physical pricing and later fresh validation. It must not be promoted using these already opened diagnostic rows.

Artifacts: [full-core result](FULL_CORE_CLOSURE_V1.json), [full-core executable audit](audit_full_core_closure.py), [source replacements](SOURCE_READ_REPLACEMENTS_V1.json), [replacement audit](audit_source_read_replacements.py).
