# Hourly strategic review — 2026-08-30 03:05 UTC

## Bottom line

The strict amount of the native model explained is unchanged:

- **29,196,288 / 545,904,054 = 5.348245316%** of stored native values have certified
  simpler component programs;
- **0.57968 / 5.30682 = 10.923302467%** of deletion CE belongs to named causal
  mechanisms;
- **4.72714 nat = 89.076697533%** of deletion CE remains unnamed;
- **0 / 68** terminal actions jointly pass extraction, selective removal, and OOD
  transport.

One useful branch did move.  A genuinely smaller $P=512$ sparse MLP1-Down program
recovered 86.51% of that component's CE effect and failed its frozen 90% gate, with
FINAL correctly sealed.  A fast $P=768$ capacity probe then improved recovery to
88.48%, but did not cross 90% and left only 5.55% of complete MLP1 storage removed.
This makes another wider Down-only dictionary poor return.  The next sparse-tensor
experiment should try to remove the native bilinear gate computation itself.

Separately, the compiled-model retraction became stronger rather than weaker at the
second coverage: the recently selected build loses to S1959 by 32.650 milli-nats at
coverage 16,110, versus 11.770 at 5,419.  The older S1959 build still beats the deployed
design by 98.768 milli-nats.  This validates the fresh-window instrument and reinforces
the rule that small selected margins are not results until transported to new rows.

## What the sparse computation is

MLP1 first produces 4,608 bilinear products

$$
g(x)=(Lx)\odot(Rx),
$$

then its native Down matrix writes $Dg(x)$ into the 1,152-dimensional residual stream.
The tested replacement keeps all of that upstream work but replaces $Dg$ with

$$
\widehat D(g)=c+A\operatorname{TopK}_{32}(Eg).
$$

`E` scores a bank of $P$ atoms; TopK retains the 32 largest positive scores at each
token; `A` decodes those 32 coefficients; and `c` is a constant intercept.  Sparse
activation therefore does not automatically mean cheap execution: all $P$ scores and
all 4,608 native products are still computed.

The causal success measure is

$$
R_{CE}=\frac{CE_{zero}-CE_{sparse}}{CE_{zero}-CE_{native}}.
$$

`zero` removes only the bias-free MLP1 Down action while retaining native `Down_bias`.
$R_{CE}=1$ means the sparse replacement recovers all of the loss caused by that
deletion; zero means it is no better than deletion.  Output $R^2$ measures local write
reconstruction and is reported separately because better local reconstruction need not
imply better final-token prediction.

## New measured result and price

The prospective P512 run used 96 FIT and 96 disjoint SELECT documents, positions
64--255, three seeds, 2,400 Adam steps, exact model/call accounting, and a sealed
96-document FINAL role.  All seeds converged near $R^2=0.621$; SELECT CE was

| arm | CE |
|---|---:|
| native MLP1 | 2.959766 |
| P512 sparse | 3.101662 |
| zero Down action | 4.011502 |

This gives $R_{CE}=0.865084$, below the frozen 0.90 admission threshold.  No factorial
or FINAL result exists.

The discovery-only P768 run reused only those already-opened FIT/SELECT roles and never
loaded FINAL.  One seed was sufficient for a capacity decision because P512's seed
spread was only $0.000216$ in $R^2$.  It ran for 97.33 seconds and returned:

| P | SELECT $R^2$ | SELECT CE | CE recovery | stored reals | full-MLP storage saved |
|---:|---:|---:|---:|---:|---:|
| 512 | 0.621211 | 3.101662 | 0.865084 | 2,950,272 | 14.8065% |
| 768 | 0.639748 | 3.080967 | 0.884761 | 4,424,832 | 5.5479% |

So capacity helps, but not efficiently enough.  P768 spends 1,474,560 more constants
to recover another 0.02070 nat.  P896 would save only 0.9186% of full MLP1 storage
before router metadata, so threshold-chasing there would not deliver a meaningfully
simpler component.

The historical P2048/k32 result at 0.9384 is not comparable confirmation.  It used a
four-times-wider bank, one seed, a smaller adjacent FineWeb evaluation split, all token
positions, and no exact checkpoint/row provenance.  Its bank stores 11,797,632 reals,
2.222 times native Down.  It proved that sparse per-token support exists in a large
bank; it did not prove storage compression.

## Largest gaps

1. **The routing scores are still expensive.**  The sparse program computes all native
   Left/Right products and a dense $P\times4608$ encoder before selecting 32 atoms.
2. **No simpler MLP1 has composed with simpler MLP0 and MLP2.**  P512 failed before the
   cube; P768 is discovery-only and also below the target.
3. **MLP0's continuous lexical/context code is not semantically factored.**  We can
   compress substantial writes but cannot yet name stable atoms with verified readers.
4. **Finite downstream consequences are not predicted by the tested local geometry.**
   The Rayleigh/Fisher predictor failed badly on held-out document-level effects.
5. **Terminal behavior APIs remain absent.**  The copy screen found a strong four-head
   bundle, but its mean-replacement intervention had too much off-target damage; no
   extraction/removal/OOD terminal action passed.

## Ranked next actions

### 1. Low-rank pre-gate quadratic router screen

For each learned detector $e_a$, fold it exactly into

$$
Q_a=\tfrac12\bigl[L^T\operatorname{diag}(e_a)R+
R^T\operatorname{diag}(e_a)L\bigr],
$$

so $e_a^Tg(x)=x^TQ_ax$.  Approximate the routing scores by rank 1/2/4/8 bilinear forms
and measure held-out score error, TopK support agreement, physical CE, storage, and
multiplies.  This is highest priority because it attacks the dominant missing execution
interface and has a cheap falsifier.  If low rank cannot reproduce routing decisions,
prune flat/tree/DAG router training before spending on it.

### 2. Finite residual-response system identification

Treat native-minus-P512 MLP1 writes as controlled finite interventions, under both
native and C512 upstream states.  Fit a small response state using signed amplitudes,
then predict held-out documents, amplitudes, one withheld consumer, and unseen paired
edits.  This tests downstream equivalence rather than local MSE and could expose the
correction that the sparse program actually needs.  It ranks second because the needed
consumer bank is not yet complete and earlier tangent predictors failed.

### 3. Interaction-resolved terminal copy circuit

The four-head copy family has a large target effect but failed the off-target gate under
position-mean replacement.  Test input-conditional or interaction-resolved
interventions on new roles rather than a powerset of the same mean ablation.  A
successful terminal circuit supplies concrete downstream equations for early writers
and can validate whether a sparse decomposition supports extraction and removal.

### 4. Joint MLP0 writer / downstream-reader sparse decomposition

Optimize atoms and downstream couplings together, with reconstruction only as an
anchor and literal graph degree/price charged.  Existing joint evidence reduced
incoming degree from about 291 to 70 at CE recovery 0.9446, but covers only one link.
The next version must predict an unseen reader or composed edit; atom labels alone do
not count.

### 5. Consumer-common blocks across several verified endpoints

Use the common reducing subspaces of signed downstream pullback forms as a
gauge-invariant state decomposition.  The planted commutant toy works, but exact
commutants are noise-fragile and the real consumer panel is incomplete.  Keep this
behind the finite-response and terminal endpoint work.

## Action executed this hour

- Audited the historical 0.9384 result against the new P512 result.  They share the
  same bias-free Down target, native-bias convention, TopK32 grammar, and CE-recovery
  formula, but differ in capacity, data, positions, optimization, and provenance.
- Implemented a reusable configurable capacity runner with explicit storage and
  multiply prices; its three CPU contract tests pass.
- Ran P768 on the shared RTX 5090 after the existing fresh-window job released it.
  Runtime was 97.33 seconds; all native/replacement/attention call censuses matched;
  FINAL opened zero times.
- Preserved the numerical receipt in
  `mlp1_sparse_capacity_frontier_discovery.json` and updated the sparse tensor direction.

The next safe action is the real-model low-rank quadratic routing-score screen.  Its
algebra and hybrid tensor/CE objective are already validated by
`toy_sparse_routed_interaction_tensor.py`; the real test must not award executable
credit to any oracle that still computes the full native gate vector.
