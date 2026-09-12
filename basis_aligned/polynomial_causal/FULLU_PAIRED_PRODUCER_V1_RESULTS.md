# Full-unembedding producer folding: rank32 is insufficient

12 September 2026. [Frozen registration](FULLU_PAIRED_PRODUCER_V1_PREREGISTRATION.md),
[primary result](FULLU_PAIRED_PRODUCER_V1_RESULT.json).

Folding the MLP16 producer metric into the full MLP17/unembedding tensor improves
the output-rank32 approximation, but does not make it sufficient. The run finished
at04:13:59 UTC, with2.76seconds of measured execution. Numerical controls pass;
the registered structural-improvement and native-fidelity predictions fail.

| Output coefficient metric | Rank32 captured energy | Rank128 captured energy |
|---|---:|---:|
| Full U, ordinary input metric |17.8212%|34.0472%|
| Full U, producer metric |20.1415%|35.9987%|
| Centered U, ordinary input metric |11.5460%|28.9985%|
| Centered U, producer metric |12.6999%|30.0145%|

The primary centered rank32 relative gain is **9.994807%**, below the registered
10% bar. Rounding it to10% would incorrectly change the verdict. More materially,
complete-path write error falls from82.66% to47.41%, still far above5%.
Producer-metric swap disagreements are77.21/69.15/61.24/36.41% across the four
developmental families; removal-CE disagreements are0.104/0.237/0.093/0.106nats.
These fail the unchanged10% swap and0.02nat removal bars.

## What was decomposed

Let $p(x)$ be the exact homogeneous MLP16 output with its residual coefficient.
The target is the entire pure producer/producer MLP17 contribution,

$$
y(x)=\frac{D_{17}[(L_{17}p(x))\odot(R_{17}p(x))]}{
\operatorname{mean}(\mathrm{pre}_{17}^2)+\epsilon}.
$$

This differs from the earlier selected two-output component. The new reference
contains every output direction of this path. Other source-pair paths, biases,
native background and final head remain outside it. The paired producer metric
uses exact weight-derived quadratic coefficients, with no text fitting.
Native validation uses the previously inspected128endpoint developmental cache;
it is not new OOD evidence.

The output eigendecomposition is an optimal output-rank truncation in the
specified paired coefficient norm. It is not a CP-rank, sparse-core or arithmetic
circuit optimum. A computation can have many output directions while sharing
cheap intermediate operations. These results therefore reject this restrictive
rank32 interface, not the possibility of structure in the composed weights.

The projected decoder uses184,320values. Keeping its native producer and MLP17
input readers adds26,542,080values, for26,726,400 in this conditional numerator
implementation versus31,850,496 for the two exact bias-free MLPs. Native
background, normalization and head are additional; this is not a whole-model
compression or extraction result.

## Negative-result check and next discriminating test

The ordinary metric and full versus centered output controls show that producer
weighting changes the spectrum, but does not concentrate enough output energy.
The exact formula and original full-U spectrum replay within numerical tolerance.
The next test changes the structural assumption: common input blocks while
retaining all output directions. Its exact contraction and quiet-direction
counterexample are executed in [the input-block note](FULLU_INPUT_BLOCKS_V1_MATH.md).
No additional rank or text-guided sweep is inferred from this failure.
