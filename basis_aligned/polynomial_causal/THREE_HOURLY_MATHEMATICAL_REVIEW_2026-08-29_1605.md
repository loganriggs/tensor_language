# Three-hour mathematical review — 2026-08-29 16:05 UTC

## Update: the first proposed model run was stopped before execution

The 15:45 review correctly identified useful causal-mechanism-reduction (CMR)
algebra, but ranked the first model target incorrectly.  A source/result audit found
that the proposed MLP1 native-channel selector substantially repeats the completed
MLP1 global-gate experiment.  I froze a natural-text MLP1 protocol, then stopped it
without opening a model outcome.  The preserved decision is
`MLP1_BILINEAR_CMR_DISCOVERY_NO_RUN.md`.

The prior experiment already asked whether a context-independent subset of MLP1's
4,608 native products could preserve downstream behavior.  It compared full-suffix
Fisher/leverage response, response energy, activation-times-`Down`, deranged-factor,
and random selectors at budgets 32, 128, and 512.  The gate rankings were stable,
but no candidate passed the consequence/control family; its status was
`no_admitted_support`.

The new CPU audit shows that the local activation/Down family was not independent
of the response-based family:

| budget | comparison with activation/Down | score Spearman | support Jaccard |
|---:|---|---:|---:|
| 32 | Fisher/leverage primary | `0.728` | `0.333` |
| 32 | response energy | `0.725` | `0.333` |
| 128 | Fisher/leverage primary | `0.734` | `0.313` |
| 128 | response energy | `0.725` | `0.313` |
| 128 | deranged factor | `-0.074` | `0.012` |
| 128 | random | `-0.047` | `0.004` |

This does not prove that centering the products gives an identical ranking, or that
retaining 50% of the channels fails.  It does show that another MLP1 local-score
screen is too redundant to be the top new mathematical experiment.

## Correction: what the CMR score does and does not measure

For one native bilinear product,

$$
a_j(x)=(L_jx)(R_jx),
$$

and its immediate residual write is $D_{:j}a_j(x)$.  Constant replacement deletes
that product and adds its mean write to the bias:

$$
b'=b+D_{:j}\mathbb E[a_j].
$$

The diagonal score

$$
s_j=\operatorname{Var}(a_j)\lVert D_{:j}\rVert_2^2
$$

is the expected squared error of this immediate write when the channel is treated
alone.  It is invariant to the exact fixed-channel rescaling

$$
L_j\mapsto uL_j,\quad R_j\mapsto vR_j,\quad
D_{:j}\mapsto(uv)^{-1}D_{:j}.
$$

That is useful, but it is not final-logit risk.  The transformer suffix contains
RMSNorms, residual additions, attention, and later bilinear MLPs, so its response is
nonlinear, state-dependent, and coupled across positions.  CMR's top-1/interchange
certificate must therefore use the **actual joint final-logit distortion** of the
compiled finite intervention:

$$
D_2=\mathbb E\lVert \ell_{\mathrm{compiled}}-
\ell_{\mathrm{native}}\rVert_2^2.
$$

For any threshold $\epsilon>0$, the measurable certificate is

$$
\Pr(\text{top-1 mismatch})
\leq
\Pr(\text{native margin}\leq2\epsilon)+\frac{D_2}{\epsilon^2}.
$$

The local score may propose deletions; only the finite final-logit measurement can
certify them.  The relevant CMR source is the recent preprint
[Asiaee (2026), arXiv:2602.24266v2](https://arxiv.org/html/2602.24266v2).  Its exact
folding algebra is independently tested here, but its empirical transfer claims are
not assumed.

## Current whole-model position

The strict balance sheet is unchanged by this mathematical correction:

- typed exact algebra covers every model site;
- named causal-path coverage is `10.92%` of global ablation headroom;
- the current shipped composite is still `0.8976` nat above its paired clean model;
- the held-out unexplained effect is dominated by the joint MLP0--MLP2 group
  (`0.728` of `0.873` global Shapley nats);
- registered early-stack interaction fractions are 43--64%.

What is genuinely working is narrower: C512 makes MLP0 `Down` 72% smaller, preserves
the layer-8 copy latent with $R^2=0.9955$, and composes with the rank-256 copy-gate
program with only `0.00064` nat aggregate interaction on the exposed cross.  The
main unresolved object is still the MLP0--MLP2 joint interface, not the already
negative question of a tiny fixed MLP1 native support.

## Revised top three mathematical moves

### 1. MLP2 finite causal mechanism reduction with a full-suffix selector

**Object.**  MLP2's 4,608 native product channels, first on the native upstream
trajectory.  Retain one fixed budget; delete the rest; replace each deleted product
by its fit-set mean and fold that write exactly into the MLP2 bias.  The executable
price with $K$ retained channels is $3456K+1152$ stored scalar values plus indices
and precision metadata, and $K$ multiplications per token.

**Operational mathematics.**  Use local gauge-invariant write mass only as one
control.  The primary selector must use a trajectory-complete suffix response or
Fisher block that sums effects across all token positions.  Final evaluation uses
actual finite post-softcap logits, shared-document simultaneous inference, and the
margin certificate above.

**Why MLP2.**  There is no equivalent completed global-gate assay at MLP2.  MLP2 is
also the known compensating/attenuating part of the MLP0--2 interaction, so a
successful program directly attacks the largest early-stack causal gap.

**Assumptions that may fail.**  Native channels may be the wrong basis; constant
means may be insufficient; singleton scores may ignore large covariance; a native
trajectory program may break under C512; a Fisher tangent may mispredict a 0-to-1
deletion.

**Measurable consequence beyond reconstruction.**  At equal executable price, the
candidate must beat local mass, uncentered activation/Down, invariant weight mass,
random, and deranged controls on final CE/KL/logit error; signed small edits must
predict finite direction; its actual-logit margin bound must be nonvacuous; the same
frozen program must later survive a native-versus-C512 background cross.

**Cheapest valid falsifier.**  One budget, constant folding only, source-document-
disjoint fit/validation rows, followed by conditional replication.  Stop if the
primary selector lacks a simultaneous positive lower confidence bound over every
control, if covariance makes the diagonal prediction inaccurate, or if the
certificate cannot guarantee at least 90% top-1 agreement.  Existing exposed rows
cannot supply fresh authority, so new outcome-blind row roles are the current launch
dependency.

### 2. Response-conditioned multi-view moment factors

**Object.**  A shared early-stack latent viewed simultaneously through MLP1 `Left`,
MLP1 `Right`, and a downstream view containing the validated copy latent plus MLP2
response.  After centering, whitening, and fixed projection to 32 dimensions, form

$$
T=\mathbb E[u\otimes v\otimes w].
$$

**Operational mathematics.**  Under a three-view latent-component model,
$T=\sum_j\kappa_j a_j\otimes b_j\otimes c_j$.  Kruskal's sufficient uniqueness
condition $k_A+k_B+k_C\geq2r+2$ fixes factors up to scaling/permutation; robust
versions quantify perturbation stability
([Bhaskara, Charikar, and Vijayaraghavan, 2014](https://proceedings.mlr.press/v35/bhaskara14a.html)).

**Assumptions that may fail.**  The deterministic views are not conditionally
independent, components coactivate, third moments may vanish, and a stable factor
may still be causally inert.

**Consequence/falsifier.**  A useful factor must match across data halves, predict a
held-out third moment, and predict which upstream edit changes all three views.  Run
ranks 8/16/32 on a $32^3$ tensor and stop on split instability or failed causal
swaps.  This is retained but second because a prospective MLP1 empirical-moment
protocol already exists; it should be repaired/finished rather than independently
reinvented.

### 3. Empirical balanced realization at the MLP0--MLP2/copy cut

**Object.**  Treat early-stack edits as inputs, the residual stream as state, and
the validated copy latent, copy-edge scalars, Fisher probes, and final logits as
outputs.  Estimate controllability $W_c$ from admissible early-MLP edits and
observability $W_o$ from downstream JVP/VJP responses.  Rank directions by the
balanced singular values of $W_cW_o$.

**Operational meaning.**  A retained direction must be both writable by an allowed
upstream mechanism and readable by a protected downstream consumer.  This is an
empirical nonlinear analogue of balanced realization, not an LTI error-bound claim
([Condon and Ivanov, 2004](https://arrow.tudublin.ie/scschmatart/70/)).

**Assumptions that may fail.**  The result depends on the edit/output batteries;
local tangents can fail for deletions; rare OOD states may be absent; classical
balanced-truncation bounds do not transfer through this nonlinear transformer.

**Consequence/falsifier.**  At ranks 16/32/64/128, balanced coordinates must predict
held-out C512, MLP1, and MLP2 edit responses better than activation PCA and the copy
HOSVD basis.  Stop if the spectrum/coordinates are unstable when documents double
or if response prediction does not beat PCA at equal rank.

## Pruned branches in this review

- **Another MLP1 native-gate CMR sweep:** stopped as substantially duplicated.
- **Local write MSE advertised as a logit certificate:** mathematically invalid.
- **Generic raw CP/HOSVD or norm minimization:** already screened; canonical HOSVD
  added only `0.93` recovery point and missed its registered two-point bar.
- **Generic SAE/dictionary atoms:** reconstruction-efficient but atom identity has
  been seed-unstable; require downstream multi-view identification before more scale.
- **Generic Hankel/automata splicing:** prior splice was OOD and had `+3.54` nat
  shift.
- **Information bottleneck:** its noise/quantization choice is arbitrary here and
  can discard rare causal variables.
- **MDL/prequential code:** retain only as a final selector among executable programs;
  it does not find the causal representation.

## Action completed in this review

The highest-priority safe CPU action was the hash-pinned duplication audit:

- `audit_mlp1_cmr_duplication.py`;
- `mlp1_cmr_duplication_audit_results.json`;
- five CMR algebra tests still pass in `1.90s`.

This changed the experiment order before spending GPU time.  The next model run is
not authorized merely by this document: the MLP2 experiment needs fresh
source-document-disjoint roles, exact finite-logit scoring, and a frozen simultaneous
inference contract.  In the meantime, its pure folding/gauge and certificate kernels
are already proof-checked and the GPU is occupied by the independent compiler lane.

