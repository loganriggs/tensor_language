# Actual retained producer under uniform-token synthetic inputs

September 14, 2026, 10:10 UTC, before execution.

The arbitrary-Gaussian tail tangent test passes but does not make those
backgrounds representative. Replace that law with actual normalized embedding
rows sampled independently and uniformly from the 50304-row checkpoint table.
These are synthetic token sequences, not natural language, training-distribution
samples, or OOD validation. No fitting is authorized by this experiment.

Freeze 256 five-token sequences in 16 batches of16, seeds170224000..170224015.
Execute native FP32 blocks0–9; capture normalized MLP7 and attention8/9 inputs
and RMS8/9. Use the existing crossfirst_state_executor_v1 hierarchy to compute
child and remainder scalar fields. Remove each along the ORIGINAL stored
MLP9_CROSSFIRST_RESPONSE_V1 direction, without amplitude rescaling. Propagate
each corner through the previously verified functional tail in FP32.

At17, build both normalized QK factors and inherited/current values from
actual corners in FP64. Construct the additive raw corner C+R-N, not the fully
propagated joint-removal corner. Use all three retained contractions.
The background z is the final-query pre-MLP state C+R-N (including all nine
attention17 heads), as in the prior conditional composed-last-block boundary.
The evaluated compact state is z+W sum(a_k); its exact FP64 MLP17 and final RMS
provide the common denominator. This is NOT a full compressed-body rollout.

Compare the three already-frozen programs in PAIR_RANKING_V1: sparse,
ordinary shared-output, balanced shared-output. A: complete nine-term Gram and
direct summed-error replay <=1e-10, finite states, and compact three-group
executor replay <=1e-10. B: serialized program costs within1% of sparse.
C (opposed by reversal or insufficient precision): both shared total-error
energies exceed sparse by >3 paired standard errors and have positive gaps in
both128-sequence halves. Report total means, paired differences/SE, selected
token contrasts, relative error against the mixed contribution itself, input
amplitude RMS and all finite failures. No adaptive sample extension.

CPU/two threads/120second calculation cap. Exact body/hierarchy dependencies,
all backgrounds, masks and adapters remain charged; matched candidate storage
does not constitute full-model compression. A timeout is inconclusive.
