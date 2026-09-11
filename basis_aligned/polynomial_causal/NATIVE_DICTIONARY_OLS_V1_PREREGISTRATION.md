# Native shared-product OLS baseline

Prior turn implemented and checked multi-output orthogonal least squares (OLS).
This run selects among MLP17's4608trained products, sharing selected support
across every output. Full-U and centered-U metrics are separate arms. No corpus
or body forwards. Existing natural-data own-neuron selection is not this object.

Normalize each product's coefficient norm to one, preserving the function by
adjusting writers. Compute native Gram G and target cross C implicitly. At each
step choose maximum residualized gain ||C_j||²/G_jj, then orthogonalize remaining
features and target cross. Save supports at32,64,128,256,512,1024. At every budget
independently solve selected-Gram conditional least squares and compare its
captured energy with summed recursive gains. A fixed seeded random-support
control gets the same exact writer solves and budgets. Report conditioning.

A instrument: native full/centered total matches prior receipts, independent
conditional solve residual and capture/gain replay errors<=1e-8; captured energy
obeys exact output-rank bounds (1e-8 slack). All reported numbers finite.
B small baseline: full-U128product capture>=.08698912596, the existing locally
converged penalized free-product fit's raw capture. This is a function-quality
reference only: objective and initialization differ, no optimizer superiority.
C broader capacity: full-U1024products capture>=.50. A miss does not reject
non-native readers, larger supports, joint blocks or circuit structure.

Price: 0forwards0tokens. Atmost2048greedy selections, two1024x1024conditional
solves per metric plus smaller controls; 900s hard execution ceiling. Native
Gram4608²FP64~170MB; never materialize vocabulary quadratic tensor. Retain only
indices, diagnostics and checkpoints' content bindings, no multi-MB artifact.
If exported independently, kproducts require2*1152*k readers plus1152*k writers
before bias/other unchanged model weights; shared U remains charged. Index-only
receipt is not an independent program. Native products remain a restriction.

Null: support curve tracks random control or misses learned-reader reference;
greedy selection supplies no compact/high-coverage representation. Red-team
negative with exact conditional replay, output-rank ceiling, random comparison
and selected-Gram conditioning. Greedy support is not globally optimal, and
ending at1024 is a capacity limit, not a convergence certificate.

Representation/solver distinction: this is multi-output OLS, not correlation-only
OMP. Exact local gains do not imply sparse recovery in a correlated dictionary.
Reference: https://arxiv.org/abs/1602.06916. Broader weights-first hypotheses remain
active; this bounded baseline provides constructive initializations and capacity
context rather than replacing their coverage.
