# Weights-generated synthetic pushforward carrier

September14 10:45UTC. The isotropic-at-every-layer MLP mean failed; actual
propagation violates that independent-input assumption. One bounded countercheck:
256uniform-token sequences, length16, batches16, seeds170232000..170232015.
Compute actual native pre-MLP17 last-query residuals under the checkpoint,
without circuit removals. Average them; compare128-row half means.

This is a synthetic weights-generated expectation, NOT a fitted native-data
background or claimed natural-language mean. No validation states, labels,
CE, Fisher metric or corpus statistics enter it. No sample/length adaptation.

Freeze K=(T-That)(b,.) with the original sparse program before native scoring.
A correction-at-b FP32replay<=1e-6 and halfmean relative disagreement<=.05.
B bytes<=1%extra. C >=5%all12normalizedrawerror improvement on both
historical120 and fresh96 panels. Store carrier and seeds as provenance;
only1536correction values required at runtime. CPU120s/two threads.
Failure ends this constant-carrier promotion; no native fitting to rescue it.
