# Weight-folded calibration scalar: vocabulary and normalization consumers

Prior work found a corpus-frequency-correlated MLP17 output direction whose
removal changes common/rare-token loss, with random-removal controls. It did
not identify a literal log-frequency output bias (sanity correlation .529
failed its .8 bar). Massive-dimension RMS gain and task-specific MLP17 damping
are already known. This test asks which of two explicit consumers of the
calibration scalar accounts for its corpus-level causal effect, and translates
the scalar producer through the bilinear weights without a rank search.

FIT: cached FineWeb rows0:48. Freeze the direction w from covariance of MLP17
outputs with log(1+FIT target-token count), normalized once. Target labels are
used only to propose w on FIT; evaluation execution never reads the next token.
Top20 frequent tokens are fixed by FIT counts. Evaluate remaining42 cached
FineWeb rows and16 Pile documents beginning at document5000, first257 tokens
per eligible document. IDs/contents fixed in CALIBRATION_TWO_READERS_V1_ROWS.json.
FineWeb has row-disjoint, not proven document-disjoint, provenance; Pile is a
corpus shift, not a claim of absence from model training. No prefix64 collisions.

MLP17 is M(u)=D[(Lu)*(Ru)]+b, with u the native normalized input. Define

    q(u) = w^T M(u)/(w^T w) = u^T Q u + beta,
    c = D^T w/(w^T w), Q = sym(L^T diag(c) R), beta = w^T b/(w^T w).

Compute Q directly from weights inFP64; no tensor fit or low-rank approximation.
If h is the final residual, write e=q*w and h_minus=h-e. The same scalar affects
two consumers: the unembedding numerator U h and the final RMS denominator.

    z[n,d] = 30*tanh((U h - n*q*U w)/(30*rho[d])),
    rho[0] = sqrt(mean(h^2)+eps32),
    rho[1] = sqrt(mean((h-q*w)^2)+eps32).

Arms00 native,10 numerator-only removal,01 denominator-only removal,11 complete
removal. This factorial explicitly intervenes on uses of the shared quantity;
it is not four different producer fits. Three fixed random unit output axes
provide matched complete-removal controls. Native epsilon and output softcap
remain in every read. All earlier model computation and remaining MLP output
are retained and charged. No whole-model structural saving is claimed.

A, instruments:30 full native-body forwards /118 sequence instances at length256
(27 data batches, one independent native facade replay, two online removal
bridges on the first held-out batch). Producer q-fold/direct scalar relative
error<=1e-5 on both evaluation cohorts. Native formula/facade and native/compiled
online-removal logit bridges maxabs<=1e-3, relativeFrobenius<=1e-5. CPU exact
algebra controls must distinguish the two consumers, retain epsilon, reproduce
the joint read, and fail an omitted-normalization variant. Finite outputs.

B, held-out calibration signature: complete removal increases rare-token CE
by>=.10 nat and decreases frequent-token CE by>=.02 nat in BOTH cohorts.
Mean absolute random-axis rare CE change must be<=.10 times the complete
calibration rare change; mean absolute random frequent CE change<=.05 nat.
These are a new held-out screen of an old candidate, not a new rank discovery.

C/D, respective consumer sufficiency: numerator-only / denominator-only must
match the full removal effect with centered-vocabulary relativeFrobenius
error<=.10 and each common/rare meanCE effect error<=max(.002,.20*abs(full effect))
in BOTH cohorts. Both may fail, indicating the registered simple attribution
is insufficient. Report the exact nonlinear joint interaction separately;
do not sum losses as if they were additive or fit a compensating gain.

Runtime derives the producer once, uses token microbatches for large vocabulary
reads, and retains compact per-row losses, vector-error sums and producer weights.
No full-vocabulary dataset dump. If B fails, stop this fitted calibration
candidate on this split; do not search another direction/rank. If B holds,
preserve its limitations and require independent provenance/consumer-interchange
confirmation before claiming the full four-property circuit. If only C or D
holds, it nominates that consumer; if neither holds, preserve both uses without
claiming a simpler one-consumer computation. GPU execution through managed queue.
