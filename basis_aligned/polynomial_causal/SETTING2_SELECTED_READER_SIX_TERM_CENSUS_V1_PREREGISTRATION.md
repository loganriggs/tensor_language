# Setting2 selected-reader six-term census V1

## Question

Which exact bilinear source terms account for changes in the twelve frozen UK/US
token-reader MLP17 numerators at the 64 already cached base/donor pairs? This
follows the small MLP16 × head17.2 result without fitting or opening new outcomes.

At each endpoint, decompose the native MLP17 input as

$$
x=g+p+a,
$$

where $p$ is the scaled MLP16 output, $a$ is head17.2's native projected write,
and $g=x-p-a$ contains the remaining incoming residual and other attention17
writes. For $q(x)=D((Lx)\odot(Rx))$, compute the three self terms and three
symmetrized mixed terms:

$$
q(x)=q_{gg}+q_{pp}+q_{aa}+q_{gp}+q_{ga}+q_{pa}.
$$

Project every term through the same twelve unembedding rows. Pair consecutive
cached endpoints as base/donor and evaluate term changes $\Delta q_s$ against
the full numerator change $\Delta q$.

## Frozen predictions and reporting

1. The six absolute terms and their base/donor changes must each sum to the direct
   native numerator with relative FP64 error at most $10^{-10}$.
2. Report for every term its change RMS, ratio
   $\|\Delta q_s\|_F/\|\Delta q\|_F$, aligned fraction
   $\langle\Delta q_s,\Delta q\rangle/\|\Delta q\|_F^2$, and per-reader ratios.
3. A term is **live** when its change-RMS ratio is at least `.05`. Freeze the
   descending change-RMS ranking. This is descriptive path selection, not circuit
   identification.
4. Report cancellation as the norm of the summed change divided by the sum of the
   six term-change norms. Do not interpret signed fractions as probabilities.

The next folding candidate is the highest-ranked live term other than `pa`, which
already failed its separate 2% absolute-reader relevance gate. If no other term is
live, close this three-source decomposition at these readers. No ranks, fits,
threshold changes, behavior/logit outcomes, fresh/OOD rows, model forwards,
gradients, updates, or quantization are allowed. Input RMS, Down bias, direct
residual, final RMS, and softcap remain outside this numerator census.
