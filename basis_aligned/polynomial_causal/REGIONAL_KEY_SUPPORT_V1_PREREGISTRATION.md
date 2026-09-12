# Shared key-update support across original and geographic prefixes

Localize dependencies of the frozen weight-discovered regional branch; this
does not discover a new unsupervised factorization. No model weights or scalar
amplitudes are fit. Use the four original source-panel prefixes for selection;
the eight geographic prefixes are held out of selection. Fresh panel prefixes
duplicate two original prefixes and are not independent validation examples.

The candidate universe is 26 native update ports: attention0/MLP0 through
attention12/MLP12. At a key layer j, only update indices below j apply. Learn
one shared support for all three key layers 8/9/13. The budget is 13 ports.
The background plus embedding term is always present and separately priced.

Forward greedy adds the port minimizing the worst joint-key error over the
four original prefixes and three layers. Ties use ascending port index. Stop
at exactly13 ports; preserve the full selection trajectory. A preregistered
red-team search starts with all26 and removes the least damaging port until13
remain; this addresses nonmonotone cancellation and forward-search traps.
Neither is a global optimization guarantee. Serialize both supports before
evaluating geographic prefixes. No post-hoc support changes from heldout results.

pred_a: all-port raw-sum/native key replay <=2e-5, and contracted joint outer-
product error agrees with dense evaluation within1e-10 on a fixed original case.
pred_b: pred_a and forward13 worst original key error <=.1.
pred_c: pred_a/pred_b and forward13 worst heldout key error <=.1.
Backward results are diagnostic and cannot rescue a failed forward prediction.
Null: a half-sized native update support does not suffice for this key interface.
A miss does not exclude rotated/shared features, changed boundaries, cancellation-
aware joint support search or a better weight-only decomposition.

CPU only, frozen saved native update terms, no full model forwards or GPU.
At most520 candidate supports across both trajectories; fixed-size projected
keys and Gram contractions, compact JSON receipts. Native-generated port cost
remains external: a support pass is not recursively pruned model execution.
