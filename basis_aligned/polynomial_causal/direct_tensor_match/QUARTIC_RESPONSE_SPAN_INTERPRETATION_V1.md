**Output refitting helps, but this fixed dictionary cannot meet10% finite-response error in the tested polynomial metric.**

Native capture and SVD oracle completed in3.41seconds. Controls/orthogonality/frame replay PASS. Capacity<10% FAIL in all four wide cells. Half-baseline-error prediction FAIL overall (FineWeb misses; code passes). These are in-sample optimal linear readouts, not trained replacements or OOD tests.

|Domain|Strength|Narrow baseline / oracle|Wide parent baseline / oracle|
|---|---:|---:|---:|
|FineWeb|.5|28.77 /26.16%|28.28 /16.58%|
|FineWeb|1|21.76 /19.13%|22.23 /12.84%|
|Code|.5|30.24 /22.68%|32.86 /13.61%|
|Code|1|27.41 /19.33%|30.70 /10.49%|

The wide design is full rank528; minimum relative singular values are at least5.4e-4. Least-squares residual orthogonality is below7e-15. The improvements are not evidence that a changed writer transfers: it sees every evaluated response target. One writer pooled over both strengths gives wide individual errors17.79/13.47%onFineWeb and14.76/11.10%oncode, consistent with the stricter joint constraint.

The target is the purequartic numerator difference in the unembedding's Euclidean geometry. Recipient RMS weighting and final nonlinearities are absent from this oracle, so these are not rigorous lower bounds on native-logit response error. Nevertheless, changing only the writer cannot reach the declared polynomial accuracy for the frozen feature dictionary. Exact graph rewrites cannot change that span either. Wider arbitrary rank fitting on the same features is demoted; finite-response feature learning is the concrete successor.

Implemented successor `quartic_finite_response.py` evaluates the exact quartic feature difference using quadratic increments and constructs a joint relative-energy value/response objective. Five CPU dimension/gradient controls agree with an independently expanded dense polynomial below1.1e-15; zero edits are exactly zero. These are algebra checks, not successful structural recovery or native training. Reuse the existing profiled readout/optimizer machinery next; preserve disjoint fitting/evaluation pairs and compare value-only versus response-aware feature learning at identical product budget.

[Preregistered scope](QUARTIC_RESPONSE_SPAN_PLAN_V1.md) · [Oracle rows](QUARTIC_RESPONSE_SPAN_V1.json) · [Response helper](quartic_finite_response.py) · [CPU controls](QUARTIC_FINITE_RESPONSE_CONTROLS_V1.json).
