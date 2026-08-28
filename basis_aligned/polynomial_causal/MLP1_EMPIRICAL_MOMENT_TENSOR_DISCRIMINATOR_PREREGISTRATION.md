# Prospective MLP1 empirical-moment tensor discriminator

Date: 2026-08-28

Status: design-only preregistration. It authorizes no row harvest, checkpoint or
model load, activation capture, fitting, backward pass, result, or scientific claim.
A later executable protocol must bind exact rows, model files, sources, seeds,
optimizers, tolerances, and publication transactions before any outcome is computed.

## 1. Question and prior boundary

For the bias-separated physical MLP1 map

\[
 F(x)-b=D((Lx)\odot(Rx))
       = T\mathbin{\lrcorner}(x\otimes x),
\]

the authoritative coefficient-HOSVD result found full numerical output and input
rank 1,152; 90% coefficient-Frobenius energy required output/input ranks 835/937;
and the largest registered $64^3$ core retained only 0.33212% of the full tensor.
This prunes low-dimensional Euclidean Tucker, not compression on the activations
that MLP1 actually receives or in directions the suffix actually reads.

This discriminator asks, in order:

1. Is MLP1 low-complexity under the empirical distribution of its natural physical
   inputs?
2. Is a covariance-matched noncentral-Gaussian surrogate accurate enough to be a
   computational proxy for that empirical objective?
3. Do learned product factors improve on original native gates, fixed PCA/Tucker,
   Down-only, affine, and random controls at matched executable price?
4. For a locally admitted candidate, is the same error small under downstream
   Fisher response?

Only questions 1--3 belong to the first result. Question 4 is a separately unlocked
screen. Live suffix, finite intervention, OOD, and MLP0/1/2 composition require later
fresh-role protocols.

## 2. Why covariance alone is not the natural metric

For a candidate $G$ with folded error tensor $E=T-\widehat T$ and exact copied bias,
the empirical natural-write loss is

\[
 \mathcal L_{\rm emp}(G)
 =\frac{\frac1N\sum_{n=1}^N\|F(x_n)-G(x_n)\|_2^2}
        {\frac1N\sum_{n=1}^N\|F(x_n)-b\|_2^2}.
\]

Its numerator contracts two copies of $E$ with the empirical fourth moment

\[
 \widehat M_4=\frac1N\sum_n x_n^{\otimes4}.
\]

It is therefore not determined by the mean and covariance. The primary objective is
the streaming sample average above; the implementation must never materialize
$1152^4$ entries.

For $X\sim\mathcal N(\mu,\Sigma)$, noncentral Wick/Isserlis gives the corresponding
fourth moment exactly from $(\mu,\Sigma)$. E8 supplied the empirical mean/covariance,
and E10 correctly implemented that formula for its Gaussian surrogate, including
the nonzero mean. Natural MLP inputs are
RMS-normalized, lie near a sphere/cone, and need not be Gaussian. Consequently
noncentral Wick is exact for the surrogate measure, not for the empirical activation
law. Adding E10's isotropic ridge

\[
 \Sigma_t=(1-t)\Sigma+t\,\operatorname{tr}(\Sigma)I/d
\]

changes the measure again; it is a robustness regularizer, not an empirical-moment
identity.

The empirical loss remains authoritative even if the Gaussian validation below
passes. Wick may then accelerate fitting; it never replaces held-out empirical
scoring.

## 3. Distinction from E8/E10

E8 captured 200k inputs to MLP8, projected the layer to the top 96 centered PCA
directions plus its mean, and reported that this projected target reproduced 0.818
of the original layer's write on a small natural-input control. E10 fitted flat and
two-layer transcoders to that projected MLP8 target under a 5%-ridged noncentral
Gaussian metric. It did not compare that surrogate loss with an independent
empirical fourth-moment loss, did not use train/validation/replication row roles, did
not target full physical MLP1, and did not test a zero-native-call candidate in the
suffix or in composition.

Those results motivate the present question but supply no row, metric, rank, PCA
dimension, threshold, or causal evidence here. Reproducing high Gaussian
tensor-similarity would duplicate E10 and is not a success criterion.

## 4. Frozen data roles and units

The later authority must allocate three mutually document-disjoint roles before any
capture:

- **FIT400:** exactly 400,000 eligible MLP1 positions, with nested deterministic
  prefixes `FIT100`, `FIT200`, and `FIT400` of 100,000, 200,000, and 400,000 positions.
  The order is a hash-seeded ordering of documents followed by positions, not corpus
  order or an activation-dependent ordering.
- **VALIDATION400:** exactly 400,000 positions. It scores every frozen fit, validates
  empirical versus Gaussian geometry, and selects the smallest-price passing
  program. It never updates a basis, support, coefficient, rank, ridge, optimizer,
  or threshold.
- **REPLICATION400:** exactly 400,000 untouched positions, opened once for the single
  selected program plus its predeclared controls. It cannot change the selection.

Eligible positions use one frozen support rule, provisionally positions 64:256 of a
256-token window with an exact token mask and common denominator. Documents, not
positions, are independent sampling units. All intervals and comparisons use one
shared source-document cluster bootstrap that recomputes pooled numerators and
denominators, with 20,000 fixed-seed draws and simultaneous 95% basic max-error
bands over every promotive comparison. Exact row-to-document and row-to-position maps, ordered unit hashes,
source/license receipts, prior-role collision checks, and model/source hashes are
required before execution.

The 100k/200k/400k ladder changes only FIT prefix size. Each rung starts from the
same seed-indexed initialization; no warm start from a larger or smaller rung is
allowed. The candidate grid and optimizer budget are identical across rungs. This
distinguishes sample scaling from extra optimization.

## 5. Cheapest branch discriminator: no learned replacement

Before the product-fit campaign, use `FIT100` only to estimate $\mu$, $\Sigma$, and
the following fixed physical input subspaces:

- `mean+PCA-k`: the normalized mean direction plus the top $k-1$ eigenvectors of
  covariance projected orthogonal to the mean;
- `PCA-no-mean-k`: the top $k$ centered PCs, a mean-ablation diagnostic;
- `mean+random-k`: four fixed-seed Haar subspaces containing the mean direction and
  otherwise lying in its orthogonal complement; and
- identity and mean-only controls.

If the FIT mean norm is at most $10^{-8}$ times input RMS, the mean direction is
declared absent: `mean+PCA-k` becomes the top $k$ PCs and `mean+random-k` becomes a
$k$-dimensional Haar subspace. This fail-closed branch is fixed before validation.

Use $k\in\{64,96,128,256\}$. On `VALIDATION400`, evaluate the nonpromotive oracle

\[
 x\mapsto F(P_kx),
\]

with the exact native polynomial and common denominator. This oracle deliberately
retains all 4,608 native products and is never called a replacement. It asks only
whether a small input subspace can support the natural write.

The literal projected-native branch survives this screen only if some `mean+PCA-k` with
$k\le256$ has document-bootstrap upper bound
$\mathcal L_{\rm emp}\le0.10$ and improves on the best-performing of the four
`mean+random-k` controls by a replicatewise simultaneous lower bound of at least
0.10 in normalized loss. Otherwise that literal projection is not used as a Tucker
initialization or positive control.

Failure cannot prune regression on PCA coordinates: in general
$E[F(x)\mid P_kx]\ne F(P_kx)$, because omitted quadratic terms may have predictable
conditional means. Fixed-PCA Tucker, direct CP, native-gate selection, learned
non-PCA Tucker, and Fisher-specific bases therefore remain separate branches.

This is the cheapest decisive test of the **literal projected-native/input-support
branch**: one fit-moment pass and one validation polynomial evaluation, no optimizer
and no backward pass. It is not a universal test of natural tensor compressibility.

## 6. Empirical-versus-Gaussian metric validation

Before fitting with Wick, freeze a 48-member quadratic residual-probe bank. Its
projection probes use the `FIT100` bases permanently, so changing moment sample size
does not change the measured functions:

- eight `mean+PCA` and `PCA-no-mean` projection residuals;
- 16 `mean+random` projection residuals, four per $k$;
- eight fixed-seed signed native-gate perturbation maps;
- eight fixed-seed native-gate dropout maps; and
- eight fixed-seed scale-matched random paired-factor maps.

No probe may depend on a validation activation or candidate outcome. For probe
outputs $H_j(x)$, compute on `VALIDATION400` the empirical Gram

\[
 K^{\rm emp}_{ij}=N^{-1}\sum_n H_i(x_n)^TH_j(x_n)

\]

and the noncentral-Wick Gram $K^{G,N}$ from $(\mu_N,\Sigma_N)$ estimated separately
on `FIT100`, `FIT200`, and `FIT400`. Normalize both by the empirical target-write
energy. Report all matrices and per-probe diagonal losses.

Wick is an allowed fit accelerator only if, at the 400k rung:

1. $\|K^{G,400}-K^{\rm emp}\|_F/\|K^{\rm emp}\|_F\le0.10$;
2. every probe with empirical diagonal at least $10^{-4}$ has relative diagonal
   error at most 0.10;
3. Spearman correlation of the 48 diagonal losses, using frozen average-rank tie
   handling, is at least 0.95; and
4. from 100k to 200k to 400k, each error statistic may increase by at most 0.01
   and Spearman correlation may decrease by at most 0.01.

The same three primary statistics are recomputed on `REPLICATION400` for the 48
frozen probes, and the selected candidate residual is appended as a separately
reported 49th calibration function. Failure means the Gaussian surrogate is invalid
for this purpose: stop all Wick-selected fits and retain empirical-minibatch fits.
If the selected program was Wick-fit, its own empirical-versus-Wick diagonal error
must also be at most 0.10 or replication fails.
Passing licenses only computation under this activation law, not Gaussianity or
causal fidelity. A separate `t=0.05` ridged-E10 arm may be reported descriptively,
but it cannot select a candidate unless it independently passes the same empirical
calibration gates.

## 7. Candidate families

All learned candidates copy the exact output bias and are fit only on `FITN`.
Native inputs and teacher writes are captured in a distinct reference-only phase.
Candidate scoring then consumes those immutable inputs/labels and executes its own
serialized program with zero calls to, or capability for, the native MLP1 module.
Reference and candidate call ledgers are separate and exact; only the former may
contain native calls. A later live phase cannot access cached teacher writes.

### 7.1 Learned CP/product program

For $q\in\{64,128,256,512,1024,2048,4096\}$ fit

\[
 G_q(x)-b=C((Ax)\odot(Bx)).

\]

Use the empirical streaming loss as primary. A Wick fit is a separate arm only after
metric validation. Use fixed optimizer schedules, five seeds, native-subset and
balanced-random initializations, and validation selection by physical function loss,
not training loss.

### 7.2 Original native-gate subset

At every CP rank $q$, fit a full multi-output decoder on the native product features
with a frozen group-lasso path, penalizing the output-column norm of each gate. The
features are first standardized to unit FIT empirical RMS, with the inverse scale
absorbed into the decoder; this makes selection invariant to scalar gauge of the
native factor pair. Order gates by first entry on that FIT-only path, take the first
$q$, and refit an
unpenalized/ridged full output decoder on the same FIT objective. The executable
copies those factor rows; retaining their original Down columns is forbidden.
Include fixed-seed random native subsets with the same decoder refit. Learned CP and
selected native gates then have identical standalone floating price and product
count. The later executable protocol must freeze the lambda path, solver tolerance,
ridge, and tie rule before capture.

### 7.3 Symmetric Tucker

At each data rung, recompute `mean+PCA-k` from that FIT prefix; validation never
updates it. For surviving fixed bases and separately registered FIT-only
Stiefel-learned orthogonal bases at the same $k\in\{64,96,128,256\}$, fit
output ranks $r_o\in\{64,128,256,512,1024,1152\}$ and either a dense symmetric core
or a top-COO core selected using FIT only. The physical input basis contains the
mean direction; the candidate is homogeneous in $x$ and does not receive a free
affine feature. Core coefficients, active input pairs, input/output bases, and bias
are all serialized.

### 7.4 Activation-weighted Down

Fit the best reduced-rank decoder from all 4,608 native product features at
$r_D\in\{128,256,512,846,970,1152\}$ under the empirical objective. Its standalone
program copies both complete native factor banks. It is a storage/write-map baseline
and always pays 4,608 products; it cannot win a product-rank claim.

### 7.5 Nonquadratic and null controls

Fit reduced-rank affine programs $UVx+c$ at the largest rank with no greater
standalone bytes than each matched tensor candidate, because the natural input cone
can make a quadratic locally linear. Also include output mean,
deranged-document labels, `mean+random` Tucker, random native gates, and scale-matched
random CP factors with the same decoder refit. A quadratic family receives no credit for a gain matched by the
affine control.

## 8. Exact executable prices

Primary serialization is float32 with int64 indices; report real/int counts and
bytes separately. There is no inherited-weight discount in the primary comparison.
For $d=o=1152$, $h=4608$:

- native: 15,926,400 floats and 4,608 products;
- learned CP or standalone selected-native $q$:
  $3456q+1152$ floats and $q$ products;
- standalone Down rank $r_D$:
  $10,617,984+5760r_D$ floats and 4,608 products;
- dense symmetric Tucker $(r_o,k)$, $p=k(k+1)/2$:
  $1152k+1152r_o+r_o p+1152$ floats and $p$ products;
- COO Tucker with $s$ core coefficients and $a$ distinct active pairs:
  $1152k+1152r_o+s+1152$ floats, $3s$ int64 indices, and $a$ products; and
- rank-$r_A$ affine: $r_A(1152+1152)+1152$ floats and zero products.

Multiply-adds, bias additions, depth, peak activation memory, and actual serialized
bytes are also mandatory. PCA computation is fit cost, while its executable input
basis is runtime/storage cost. A source gate index is metadata only when factors are
copied; an inherited-library count may be reported secondarily but cannot replace
standalone price.

Promotive comparisons are either exact equal-price CP versus native-gate comparisons
or coordinatewise no-greater standalone bytes and products. No decoder-only,
replacement-only, or float-only sparse price may be called matched.

## 9. Selection, doubling, and conditioning gates

For each family and price rung, validation records normalized natural loss and
document-bootstrap intervals at all three FIT sizes. `VALIDATION400` nominates a
single local candidate only if all conditions hold:

1. the `VALIDATION400` upper confidence bound on $\mathcal L_{\rm emp}$ is at most
   0.10;
2. it beats every required matched random, affine, and deranged control by a
   simultaneous loss-reduction lower bound of at least 0.02;
3. at equal $q$, learned CP beats selected native gates by at least 0.02 before any
   claim that new factors help; otherwise the native subset is preferred;
4. validation loss does not worsen by more than 0.01 from FIT100 to FIT200 or FIT200
   to FIT400, and the FIT200/FIT400 physical predictions differ by at most 0.05 of
   target-write RMS on validation; and
5. the candidate is the smallest standalone-byte program satisfying the preceding
   gates, with products as the deterministic tie-break.

`REPLICATION400` then repeats conditions 1--3 for that candidate and its already
frozen controls. Local admission requires the replication upper bound to remain at
most 0.10 and both simultaneous control margins to remain at least 0.02. Failure is
terminal; replication cannot nominate a different program.

Every fit must additionally pass:

- finite losses, gradients, factors, and serialized replay;
- exact scalar-gauge balancing with physical-function replay at relative tolerance
  $10^{-6}$ in float64 and $10^{-5}$ in float32;
- normal-equation or local Gauss--Newton condition number at most $10^8$ after the
  frozen ridge;
- validation cancellation ratio
  $\sum_a E\|g_a(x)\|^2/E\|\sum_ag_a(x)\|^2\le100$;
- the largest balanced gate norm at the final checkpoint is at most twice its
  minimum over the final 20% of optimization;
- deployed-precision loss no more than 10% relatively worse than float32 and still
  inside the absolute natural-loss gate; and
- agreement of at least three of five seeds within 0.02 validation loss. Seed is not
  a selectable hyperparameter; take the fixed median-loss fit among passing seeds.

These are border-rank/ill-conditioning guards. A low empirical loss achieved by
divergent cancellation is not a finite executable simplification.

## 10. Registered stop rules

1. If the PCA oracle gate fails, stop only the literal projected-native seed/control
   at $k\le256$; do not prune fitted PCA-Tucker, CP, or Fisher compression.
2. If Wick calibration fails, stop Gaussian/ridged-surrogate selection. Continue only
   empirical fits.
3. The first direct-product screen is $q\in\{64,128,256,512\}$ at FIT100. Extend it
   through the complete data ladder only if CP or selected-native $q=512$ has
   validation loss at most 0.25, or improves on $q=256$ by at least 0.05. Otherwise
   stop the direct-product branch before $q\ge1024$. This is a prospective futility
   rule, not evidence that unsearched ranks cannot work.
4. If no family passes the 100k/200k/400k stability gates, stop before replication.
5. If replication misses the absolute or matched-control gate, report
   `no_admitted_natural_metric_program` and do not unlock Fisher or suffix work.
6. If learned CP fails to beat the equal-price native-gate subset, prune the claim
   that factor learning discovers a simpler basis; the native-gate program may still
   survive independently.

All stopped branches and all evaluated rungs remain reported. No threshold, rank,
ridge, support, seed, or optimizer may be added after viewing validation.

## 11. Fisher screen for a frozen natural survivor

Natural admission unlocks, but does not itself perform, a fresh teacher-state Fisher
screen. On source-document-disjoint roles, let $J_c$ be the frozen suffix Jacobian
from the MLP1 physical write to scored future logits and
$F_c=\operatorname{diag}(p_c)-p_cp_c^T$. For write error $e_c$ define

\[
 \mathcal L_F=\frac{\sum_c e_c^T J_c^TF_cJ_c e_c}
                    {\sum_c (F(x_c)-b)^TJ_c^TF_cJ_c(F(x_c)-b)}.

\]

Here $c$ denotes a complete scored window/document block, $e_c$ concatenates all
MLP1 write errors in that block, and $J_c$ maps that joint edit to all registered
future logits. It is not a sum of independent one-position Jacobians; causal
cross-position terms are retained.

Two independent 32-probe fixed-seed categorical-Fisher halves estimate this quantity;
probes are Monte Carlo replicates, while documents remain the sampling units. The
screen must retain cross-position future reads, use common denominators and states,
and compare activation-selected, Fisher-refit, shuffled-probe, random-factor, affine,
and native-gate candidates at matched price. The Fisher candidate passes only if both
probe halves and an untouched replication role have simultaneous upper bound
$\mathcal L_F\le0.10$, half-to-half point estimates differ by at most 0.05, and all
matched-control improvement lower bounds exceed 0.02. A Fisher-refit program must
also retain natural-loss upper bound 0.10 on its untouched role; Fisher improvement
cannot free-ride on lost natural behavior. A nonfinite or near-zero native Fisher
denominator is a hard failure, with the numerical floor frozen before execution.

Because $J_c^TF_cJ_c$ varies with context, this is not another fixed fourth-moment
metric and is not made exact by noncentral Wick. A teacher-state Fisher pass is a
local consequence-weighted approximation, not an executable replacement.

## 12. Later zero-native-call, causal, OOD, and composition gates

Only a frozen natural-plus-Fisher survivor may enter later protocols, each on fresh
roles:

1. **Single-site live replay:** install the serialized program at MLP1 with exact
   native parent state, require exactly zero native MLP1 calls, and score local write,
   downstream block states, final KL, and CE against native and deletion controls.
2. **Finite interventions:** use fixed signed gate/input perturbations and amplitudes;
   compare predicted Fisher response with observed future KL using the registered
   $\tfrac12\epsilon^2$ convention. Include document-deranged and sign controls.
   Teacher-forced labels receive no executable credit.
3. **OOD:** replay the unchanged program on a separately licensed corpus, including
   code, with no refit or rank/support selection.
4. **Composition:** cross independently frozen MLP0, MLP1, and MLP2 candidates in all
   eight cells. Use common-document cluster bootstrap, all conditional increments,
   and pair/triple Möbius interactions from the common comparison contract. Large
   compensating interactions imply a joint behavioral program, not modular causal
   equivalence.

Promotion to “simpler MLP1 replacement” requires the live absolute KL/CE gates,
pointwise collateral, OOD retention, exact call/price integrity, and composition
no-free-rider gates. The first empirical-moment result may conclude only that a
priced local natural-write program exists—or that none exists in the frozen families
and budgets.
