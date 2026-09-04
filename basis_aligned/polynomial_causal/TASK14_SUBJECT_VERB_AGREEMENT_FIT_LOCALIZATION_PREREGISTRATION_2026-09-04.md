# Task 14 FIT localization: grammatical subject-number state and its readers

**Frozen prospectively:** 2026-09-04 09:17 UTC. **Status:** CPU-only design and preregistration. No localization
implementation, activation, model, checkpoint, GPU, queue, result, or later-phase artifact was opened or created in
this unit.

## Decision and scientific target

The task-14 capability result at commit `90c5b1606f6eb309ea9fca0042414c9146d8c455` is used only as a binary
opener: the frozen FIT behavior is capable enough to justify a separately reviewed localization experiment. Capability
does not identify a circuit.

The target variable is the grammatical number of the complete subject:

$$
s(x)=
\begin{cases}
-1,&\text{the grammatical subject is singular},\\
+1,&\text{the grammatical subject is plural}.
\end{cases}
$$

The registered high-level computation is

$$
y(x)=
\begin{cases}
\texttt{ is},&s(x)=-1,\\
\texttt{ are},&s(x)=+1,
\end{cases}
$$

while the number and lexical identity of a non-subject attractor must not determine $y$. In C, two singular noun
conjuncts joined by “and” jointly set $s=+1$; the state is a grammatical property of the full subject, not simply the
plural suffix of one head token.

The desired result is a task-conditioned residual-stream coordinate which carries $s$, transfers across noun identity
and syntax, is causally sufficient to move the model toward the answer associated with a donor's $s$, and is causally
necessary for the native agreement decision. The second target is the downstream interval that reads this state.

This is deliberately **not** a rank-reduction experiment. Rank one is primary because $s$ is binary; ranks two and four
are controls asking whether the rank-one causal abstraction is inadequate. Nor are native attention heads or MLPs
treated as the semantic basis. The primary sites are residual-stream boundaries. A physical block or tensor factor is
examined only after a residual state and a reader interval pass causal tests.

## Frozen inputs and what was learned from capability

This design binds only the already frozen FIT authority:

- FIT authority artifact SHA-256:
  `e88fd860c28c9b369abe4a8ec28372f93bb94b6e841265206c43e6929a25ac2f`;
- FIT logical-row SHA-256:
  `3cf3315a77b3176418739e7a9357c0dbd9b95724d6b276038f53691b873377d1`;
- full logical-authority SHA-256:
  `1cf6cf12668c7428719134bbee03ab84f57cc150f2653cc12ffc4a71566c8db1`;
- generator SHA-256:
  `33d7b62b3a0ffb4c798e75f085b7e96988e09b07be16667c5f9f8871c6339f94`;
- capability preregistration SHA-256:
  `06a9747b4707999e11637a45cf83588bfd9cb8671d6b3a25790518af62900f8b`.

The post-capability CPU audit reported that the six ordinary A1/A2/P side cells were 32/32, whereas C was 28/32 on
base and 29/32 on donor. It also reported nonzero raw P/C paired-margin differences. These facts are used only to
prevent an invalid control definition and postselection: all C rows remain in the experiment, including every native
error and every `key`/`dog` lexical case. P and C are controls for zero change **through the proposed subject-number
coordinate**, not claims that two natural prompts must have identical full logits. No correctness filter is allowed.
These capability values are not localization evidence and do not select a boundary, direction, rank, component, or
reader.

SELECT, TEST, and OOD authorities and outcomes remain closed. Nothing below licenses opening them.

## The four semantic families

The fixed logit contrast is always

$$
m(x)=\ell_{\texttt{ are}}(x)-\ell_{\texttt{ is}}(x).
$$

This makes the sign convention independent of which side of a pair is called base: positive is plural/` are`, and
negative is singular/` is`.

- **A1:** changes only subject-head number in a prepositional-phrase construction. The answer changes and a valid
  subject-state interchange must have the donor sign.
- **A2:** makes the same answer-changing edit in a relative-clause construction. It supplies the cross-syntax test.
- **P:** changes attractor noun identity while subject number and answer stay fixed. Its paired arm is a zero-coordinate
  control. In addition, P prompts receive opposite-number donors from other noun groups, making P a positive transfer
  test across noun and attractor identity rather than merely a no-change control.
- **C:** changes attractor number under a coordinated plural subject while the answer remains ` are`. It is kept
  separate from P and tests that an attractor-number edit does not travel through the learned subject-number
  coordinate. Raw output equality and native correctness are not required.

## Frozen within-FIT discovery and validation split

The authority deliberately uses the same head/attractor noun roles in groups $g$ and $g+16$, with complementary
number states. Those mirror groups must stay on the same side; splitting them would put exact natural prompts in
both discovery and validation. Treat each $\{g,g+16\}$ for $g\in\{0,\ldots,15\}$ as one indivisible unit. Stratify
the 16 units by whether $g$ is even or odd; these are respectively the congruent-state and incongruent-state number
cycles. Within each stratum, units are ordered by

```text
sha256("task14-fit-localization-v1|discovery-validation-pair-coherent|"
       + group_id(g) + "|" + group_id(g + 16))
```

and the first four mirror units enter DISCOVERY; the other four enter VALIDATION. The exact group numbers are:

```text
DISCOVERY = [0, 1, 4, 6, 9, 10, 11, 15, 16, 17, 20, 22, 25, 26, 27, 31]
VALIDATION = [2, 3, 5, 7, 8, 12, 13, 14, 18, 19, 21, 23, 24, 28, 29, 30]
```

The canonical partition record has SHA-256
`125b744d311088b3b6a41b144be51bacd81478212c71b5b82d04fef3548612ec`. Each half contains four groups in every
subject-number $\times$ attractor-number cell and eight disjoint head-noun pairs. Each contains 64 authority rows: 16
linked A1/A2/P/C panels and 128 distinct natural endpoint prompts. Noun identities can recur in different semantic
roles because the authority is Latin-balanced, but no `group_id`, row, head-noun pair, prompt, or learned activation
crosses the discovery/validation boundary.

DISCOVERY may be used for site screening, optimization, hyperparameter health checks, and choosing a single frozen
candidate. VALIDATION is evaluated once after that choice. A failed validation cannot be repaired by moving a group,
changing a donor, selecting another seed, rank, site, or checkpoint, or relaxing a threshold.

## Multiple frozen donor constructions

An endpoint is one natural base or donor prompt, identified as `row_id:base` or `row_id:donor`. Donors are formed
separately inside DISCOVERY and VALIDATION; they never cross the partition. The seed label is
`task14-fit-localization-v1|donors`.

For every A1 and A2 endpoint, freeze four opposite-subject-number donors:

1. `paired`: the other endpoint of the same authority row, which changes one head-number token and preserves noun
   identity and syntax;
2. `cross_noun_1` and `cross_noun_2`: the first two SHA-ranked endpoints with the same family and attractor number,
   opposite subject number, and a different group;
3. `cross_syntax_1`: the first SHA-ranked endpoint in the other answer-changing family, with the same attractor
   number, opposite subject number, and a different group.

For every P endpoint, freeze two SHA-ranked P donors with the same attractor number, opposite subject number, and a
different group. These are the P positive-transfer rows. For P and C paired controls, base is the target and donor is
the donor; both have the same registered subject number.

Candidate ordering for a target $t$ and candidate $d$ is lexicographic order of the following digest, using the
literal separators shown:

```text
sha256(seed_label + "|same|"   + partition + "|" + family + "|" + t + "|" + d)
sha256(seed_label + "|syntax|" + partition + "|" + family + "|" + t + "|" + d)
sha256(seed_label + "|p|"      + partition + "|" + t + "|" + d)
```

The resulting logical donor contract has 704 records: 352 per partition. Per partition these are 64 paired A1/A2,
128 cross-noun A1/A2, 64 cross-syntax A1/A2, 64 P positive-transfer, 16 paired P controls, and 16 paired C controls.
The canonical record with keys sorted, compact JSON separators, records ordered by partition, arm, family, target,
and matching has SHA-256
`25a1f09d5947301f573b223abfbcae1699555ddf809f2b137eabffcbe776f3dc`. A later CPU compiler must independently
reconstruct this manifest and either match the digest or stop. This document does not materialize the manifest.

This donor multiplicity is part of identification. A direction that works only for the one-token paired edit is a
lexical edit direction, not an established grammatical-number state.

## Residual sites and the finite intervention

Let $D=1152$. The residual boundaries are $b=-1,0,\ldots,17$: $b=-1$ is the normalized embedding residual presented
to the block stack, and $b=l$ is the residual after complete block $l$. No attention or MLP output is called the
semantic unit at this stage.

At every boundary use two semantic token positions, read from the authority rather than fixed integer positions:

- $H$: the subject-head position for ordinary A1/A2/P prompts;
- $Q$: the final prompt position at which the copula logits are predicted.

Thus the initial grid has $19\times2=38$ sites. A C prompt has two conjunct-head coordinates, so C is not used to
pretend that $H$ is a single-token subject. C enters DAS fitting and required invariance gates at $Q$ sites only. At
an $H$ site, P supplies the required same-state control; projections at both C conjuncts are reported separately as
descriptive measurements and cannot satisfy or fail a gate. This avoids inventing a single-token representation of a
coordinated subject.

For a rank-$k$ orthonormal matrix $U_{b,p}\in\mathbb R^{D\times k}$, let

$$
P_{b,p}=U_{b,p}U_{b,p}^{\mathsf T}.
$$

The finite interchange from natural donor $d$ into target $x$ is

$$
r'_{b,p}(x\leftarrow d)
=r_{b,p}(x)+P_{b,p}\bigl(r_{b,p}(d)-r_{b,p}(x)\bigr).
$$

Only that semantic position is changed. All other target positions and the rest of the forward computation remain
native. The sign of a rank-one $u$ is fixed after fitting by

$$
u^{\mathsf T}(\mu_{+1}-\mu_{-1})>0,
$$

where class means use DISCOVERY only.

For an opposite-state pair define its signed interchange effect

$$
E_P(x,d)=\frac{s(d)-s(x)}{2}
\left[m\!\left(x\leftarrow_P d\right)-m(x)\right].
$$

Positive $E_P$ means movement toward the donor's answer in both singular-to-plural and plural-to-singular directions.
The full-position ceiling $E_I$ uses $P=I_D$. For any arm $A$,

$$
\operatorname{direction}(A)=\frac1{|A|}\sum_{(x,d)\in A}\mathbf1[E_P(x,d)>0],
$$

$$
\operatorname{recovery}(A)=
\frac{\operatorname{mean}_{A} E_P(x,d)}
     {\operatorname{mean}_{A} E_I(x,d)}.
$$

Recovery is invalid rather than infinite if the denominator is nonpositive, nonfinite, or if the corresponding full
interchange has direction below 0.80.

For a same-state P or C pair define output leakage

$$
L_P(x,d)=\left|m\!\left(x\leftarrow_P d\right)-m(x)\right|
$$

and coordinate leakage

$$
Z_P(x,d)=\left\|U^{\mathsf T}\left(r_{b,p}(d)-r_{b,p}(x)\right)\right\|_2.
$$

Output leakage is normalized by the median absolute output change produced by paired A1/A2 projected subject swaps
at the same site. Coordinate leakage is separately normalized by the median $Z_P$ magnitude of paired A1/A2
opposite-state residual differences; activation units are never divided by logit units. This asks whether the
proposed grammatical-number coordinate changes under an answer-preserving nuisance edit. It does not ask
all other residual directions or full output margins to remain equal.

## Discovery screen: gradients can nominate, not establish

At all 38 sites, DISCOVERY A1/A2 paired rows receive:

1. a full-position interchange ceiling $E_I$; and
2. a native gradient score

$$
G_{b,p}=\frac{\left\|\operatorname{mean}_{x}\left[s(x)\nabla_{r_{b,p}}m(x)\right]\right\|_2}
{\sqrt{\operatorname{mean}_{x}\left\|\nabla_{r_{b,p}}m(x)\right\|_2^2}}.
$$

A site is screen-eligible only if the full-position interchange direction is at least 0.80 separately on A1 and A2,
and both mean signed ceilings are positive. Among eligible sites, compute

$$
S_{b,p}=G_{b,p}\min_{f\in\{A1,A2\}}
\frac{\operatorname{mean}_{f}E_I}
{\operatorname{mean}_{f}|m(d)-m(x)|}.
$$

The ratio is clipped to $[0,1]$ only for ranking. Keep the top three $H$ sites and every eligible $Q$ site. Keeping
the complete $Q$ trajectory is required to locate the onset of downstream use rather than choosing a late site with
the largest gradient. Break $H$ ties by earlier boundary. If fewer than three $H$ sites are eligible, keep all of
them; if no $H$ site or no $Q$ site is eligible, stop with `no_intervention_ceiling`.

$G$ and $S$ are screens only. They are not causal evidence, do not name a writer or reader, and never enter a success
predicate. This guards against treating local gradients as the circuit while still avoiding an unbounded DAS search.

## Rank-one DAS fitting and cross-fitting

First fit `joint` rank-one directions on DISCOVERY at the retained $H$ sites and every eligible $Q$ site. After the
DISCOVERY-only site choice below, fit two cross-fit directions only at the selected $H$ and selected $Q$ sites:

- `joint`: A1 and A2 answer-changing rows together;
- `A1_only`: A1 answer-changing rows only;
- `A2_only`: A2 answer-changing rows only.

At a $Q$ site, the joint objective is the equally weighted mean of normalized A1 and A2 signed effects, plus half
weight on P positive transfer, minus half weight on each P and C same-state output leakage. Each family and donor
matching is averaged first, so a large arm cannot dominate by record count:

$$
J(U)=\tfrac12(\bar E_{A1}+\bar E_{A2})
+\tfrac12\bar E_{P+}
-\tfrac12\bar L_P-\tfrac12\bar L_C.
$$

At an $H$ site, the objective is instead

$$
J_H(U)=\tfrac12(\bar E_{A1}+\bar E_{A2})
+\tfrac12\bar E_{P+}
-\tfrac12\bar L_P,
$$

with no C term. Both coordinated-subject conjunct projections are retained for descriptive reporting only.

Every term is divided by its site's mean paired A1/A2 full-interchange ceiling. `A1_only` and `A2_only` use the same
formula but omit the other answer-changing training family; that omitted family is used only for cross-syntax
evaluation.

Use five preregistered optimizer seeds `14001, 14002, 14003, 14004, 14005`, 400 stratified minibatch steps per fit,
32 logical donor records per step, Adam at learning rate 0.03 with cosine decay to zero, and QR orthonormalization on
every forward. The batch schedule is a SHA-256 ordering of record IDs under the optimizer seed and cycles without
replacement within every arm/matching stratum. The model weights remain frozen. A fit is unhealthy if its projector
moves less than 0.02 from initialization, its final 50-step objective does not exceed its first 50-step objective, any
gradient or value is nonfinite, or any required arm is absent. An unhealthy fit is invalid, not a negative result.

For each site and fit kind, choose a seed medoid using only the pairwise distances between DISCOVERY causal-effect
vectors. Select the $H$ site with largest median joint $J$ across healthy seeds; ties within 0.01 go to the earlier
boundary. Preserve the complete eligible $Q$ curve for the state-formation rule below. The selected $Q$ site is the
later boundary of the first sharp transition defined there; if there is no sharp transition, it is the boundary with
largest median DISCOVERY joint $J$, with ties within 0.01 going earlier, and reader status is forced unresolved.
Validation never chooses a seed, checkpoint, or site. At least four of five seeds must be healthy and must pass the
seed-level validation direction predicates below. Report all five.

Ranks two and four are fitted only at the selected rank-one $H$ and $Q$ sites, with the identical rows, five seeds,
steps, batches, optimizer, and model calls. They are matched-opportunity falsifiers, not alternative success routes.
Their extra dimensions do not earn a compression claim.

## Necessity, sufficiency, and transfer

### Sufficiency

The interchange above is the sufficiency test. On locked VALIDATION, the selected joint rank-one coordinate must pass
every donor construction separately:

- A1 paired, cross-noun-1, and cross-noun-2;
- A2 paired, cross-noun-1, and cross-noun-2;
- A1 target with A2 donor and A2 target with A1 donor; and
- both P cross-noun positive-transfer matchings.

The two cross-fits are also mandatory: `A1_only` is tested on A2 and `A2_only` on A1 without refitting.

### Necessity

Let $a(x)=u^{\mathsf T}r_{b,p}(x)$ and let

$$
a_0=\tfrac12\left(\operatorname{mean}_{s=+1}a+
                         \operatorname{mean}_{s=-1}a\right)
$$

on DISCOVERY. Neutralization sets only this coordinate to $a_0$:

$$
r^{\rm neutral}_{b,p}(x)=r_{b,p}(x)-u\bigl(a(x)-a_0\bigr).
$$

Its signed necessity effect is

$$
N(x)=s(x)\left[m(x)-m\!\left(x^{\rm neutral}\right)\right].
$$

Positive $N$ means removing the proposed number coordinate reduces support for the native subject-number answer.
Necessity is evaluated separately for A1 and A2, both subject-number directions, both attractor-number states, and
every natural endpoint. It is not inferred from sufficiency and does not require a donor.

### Fixed validation bars

The rank-one hypothesis passes only if all of the following hold on VALIDATION:

1. `joint` has direction at least 0.80 and recovery at least 0.50 in every A1/A2 same-syntax matching named above;
2. each cross-syntax arm has direction at least 0.75 and recovery at least 0.40;
3. each P positive-transfer matching has direction at least 0.75 and recovery at least 0.40;
4. P at both selected $H$ and $Q$ sites, and C at the selected $Q$ site only, each have mean normalized output leakage
   at most 0.20, median normalized coordinate leakage at most 0.20, and 90th-percentile coordinate leakage at most
   0.50; C conjunct projections at $H$ are descriptive and do not enter this predicate;
5. A1 and A2 necessity each has positive mean $N$, at least 0.65 of rows with $N>0$, and removes at least 0.25 of the
   native correctly oriented mean margin;
6. every one of the four subject-number $\times$ attractor-number cells has interchange direction at least 0.70;
7. `A1_only` on A2 and `A2_only` on A1 each has direction at least 0.70 and recovery at least 0.35;
8. at least four of five optimizer seeds pass all direction bars, and their median recovery passes every recovery bar;
9. no rank-two or rank-four fit improves median validation recovery by more than 0.10 on either A1 or A2, rescues a
   failed cross-syntax/P arm, or is required to make P/C leakage pass; and
10. all 16 C groups and both natural sides are reported, including native errors and lexical strata. No row-level
    exclusion is permitted.

These thresholds concern the subject-number intervention, not generic reconstruction error or a reduced model's CE.
Failing item 9 specifically rejects the simple binary-state account; it is not a reason to relabel rank four as the
successful circuit.

## Locating state formation, then testing a reader

The selected $H$ site asks where subject number can be taken from the head token. The full $Q$ curve asks where a
task-relevant number state becomes available at the prediction position. The earliest pair of consecutive residual
boundaries in the DISCOVERY trajectory across which median rank-one $Q$ recovery reaches at least 90% of the best
later $Q$ recovery while the preceding boundary is below 50% of that best value is a **Q-state formation or transport
interval**, not yet a reader. An ineligible preceding boundary counts as zero only because its full-position causal
ceiling already failed. If no such sharp transition exists, the state is recorded as gradually transported.

For a proposed upstream site $i$ and later site $j$, run two finite two-site interventions. First patch $u_i$ from an
opposite-state donor, then at $j$ restore the selected $u_j$ coordinate to its unpatched target value. Let $E_i$ be
the upstream signed effect and $E_{i\to j\,\mathrm{reset}}$ the remaining effect. The mediated fraction is

$$
M_{i\to j}=1-\frac{E_{i\to j\,\mathrm{reset}}}{E_i}.
$$

Second, neutralize $u_i$ and patch the donor coordinate at $j$. Its rescue fraction is the resulting signed effect
divided by the ordinary $j$-only signed effect. Calling the ordered $H\to Q$ handoff a reader requires median
$M_{i\to j}\ge0.70$ and median rescue at least 0.70 on A1 and A2 separately, plus the same P/C leakage bars. These are
tests of a residual-state handoff, not evidence that a named native component is itself the semantic object.

The upstream site must be the selected $H$ boundary and must strictly precede the candidate $Q$ boundary. If the
DISCOVERY-selected $H$ site is not earlier than any qualifying $Q$ formation boundary, no alternate $H$ site is chosen
using validation; the terminal is `fit_state_supported_reader_unresolved`. A sharp $Q$ increase without passing the
ordered reset/rescue test is likewise state formation without an identified reader.

### Redundancy and hidden interaction effects

Single-site activation patching can include interaction with other mediators. Therefore a failed single-site
necessity test does not by itself reject a distributed or redundant implementation. For every pair among the top two
DISCOVERY $Q$ sites, measure neutralization damage

$$
D_S=\operatorname{mean}_x s(x)\left[m(x)-m(x^{\rm neutral\ at\ }S)\right]
$$

for $S=\{i\},\{j\},\{i,j\}$ and the finite interaction

$$
I_{ij}=D_{\{i,j\}}-D_{\{i\}}-D_{\{j\}}.
$$

If each singleton removes less than 0.25 of the native oriented margin but the pair removes at least 0.50 and
$I_{ij}$ is positive by at least 0.20 of that margin, the result is `two_site_redundant_candidate`, not
`rank1_absent`. Conversely, if each singleton is large and the pair is no larger than their maximum within 0.10, the
sites are consistent with two points on one serial path. Every interaction classification must repeat on VALIDATION
with the pair frozen from DISCOVERY. Other patterns are `interaction_unresolved`; no additive natural-indirect-effect
interpretation is allowed.

## Opposing predictions

| Hypothesis | Required observation | Observation that defeats it |
|---|---|---|
| Shared grammatical subject-number state | Rank one transfers across A1/A2 syntax, multiple donors, and P noun identity while P/C same-state edits have low projected leakage. | Only the within-row edit works, cross-syntax or P transfer fails, or C attractor number moves the coordinate. |
| Binary state is sufficient | Rank one passes necessity and sufficiency and ranks two/four do not materially improve held-out transfer. | Higher rank rescues a failed arm or improves recovery by more than 0.10. |
| State is read at a localized residual transition | Upstream patch, downstream reset, and downstream rescue satisfy the reader bars. | Effects grow gradually, reset does not remove them, or rescue fails. |
| Two redundant routes carry the state | Singleton neutralizations are weak, joint neutralization is strong, and positive interaction repeats on validation. | Joint damage is additive/weak, or the pair was selected only by validation. |
| Lexical/template shortcut | Strong paired A1/A2 fit but poor cross-noun, cross-syntax, or P transfer. | Transfer bars pass across all frozen constructions. |
| Nearest-noun-number state | C attractor-number edit strongly changes the projected coordinate/output. | C projected leakage passes while ordinary subject swaps remain strong. |

No basis overlap, gradient magnitude, probe accuracy, component rank, or native head label can decide among these
hypotheses alone.

## Exact translation to bilinear weights after causal identification

Residual DAS identifies a finite causal state; it does not automatically say which weights compute or read it. After
the FIT residual and reader gates pass, a separately preregistered translation may open only inside the frozen reader
interval.

For a bilinear MLP receiving its actual normalized input $z$, write

$$
F(z)=W_D\left[(W_Lz)\odot(W_Rz)\right].
$$

Because RMS normalization is nonlinear, a direction learned before RMSNorm is not silently treated as a linear
direction in $z$. The residual intervention must first be repeated at the stored normalized MLP input and pass the
same answer-changing and P/C gates. Let $q$ be that causally validated normalized-input direction and let $u$ be the
validated downstream output/read direction. Then

$$
u^{\mathsf T}F(z)=z^{\mathsf T}Q_u z,
$$

where the exact symmetric quadratic weight matrix is

$$
Q_u=\frac12\left[
W_L^{\mathsf T}\operatorname{diag}(W_D^{\mathsf T}u)W_R+
W_R^{\mathsf T}\operatorname{diag}(W_D^{\mathsf T}u)W_L
\right].
$$

With $P_q=qq^{\mathsf T}$, this computation has the exact, exhaustive decomposition

$$
Q_u=P_qQ_uP_q+
P_qQ_u(I-P_q)+(I-P_q)Q_uP_q+
(I-P_q)Q_u(I-P_q).
$$

Equivalently, for $z=\alpha q+c$ with $q^{\mathsf T}c=0$,

$$
u^{\mathsf T}F(z)=
\alpha^2q^{\mathsf T}Q_uq
+2\alpha q^{\mathsf T}Q_uc
+c^{\mathsf T}Q_uc.
$$

The first term is subject-state self-interaction, the second is subject-state $\times$ context interaction, and the
third is context-only computation for this downstream read direction. These are literal contractions of the trained
weights, not an SAE, PCA, quantization, or low-rank approximation. Their causal status still comes from repeating the
finite necessity, sufficiency, transfer, and control interventions after applying the corresponding weight terms.

If the reader interval is attention rather than an MLP, the analogous exact pre-softmax object for head $h$ is

$$
B_h=\frac{W_{Q,h}W_{K,h}^{\mathsf T}}{\sqrt{d_h}},
\qquad
\operatorname{score}_{h}(i,j)=z_i^{\mathsf T}B_hz_j,
$$

and $P_qB_h$, $B_hP_q$, and their complements separate query-slot and key-slot use of the validated state. Heads are
physical summands, not the semantic basis: candidate terms are grouped by their common downstream causal effect
through the OV path. Softmax makes the complete attention map nonlinear, so only the QK score decomposition is
algebraically exact; the full attention claim must again pass finite interventions.

## Decision rule and later-stage boundary

The future FIT localization terminal is one of:

- `fit_rank1_state_and_reader_supported`: every validation bar passes and either a reader handoff or a preregistered
  two-site redundancy account passes;
- `fit_state_supported_reader_unresolved`: state transfer/control bars pass but reader localization does not;
- `fit_binary_state_rejected`: ranks two/four materially rescue the result or required transfer/control bars fail;
- `no_intervention_ceiling`: full residual interchange cannot produce the registered effect;
- `instrument_invalid`: hashes, rows, donors, runtime, optimizer health, completeness, or finite-value gates fail.

Only the first terminal may motivate a new, independent SELECT localization preregistration. The second permits a new
FIT-only interaction design, not SELECT. The third permits a new scientific hypothesis, not a threshold change. The
last two open no scientific continuation. In all cases TEST and OOD stay closed, and bilinear translation requires a
separate prospective contract and independent review.

## Required price freeze before any execution

This design freezes 38 candidate residual sites, the exact 704-record logical donor contract, five optimizer seeds,
400 steps, and the evaluation arms, but it deliberately does not authorize a physical execution plan. Before any
model-facing implementation, a separate CPU compiler must enumerate and hash every native, full-interchange,
gradient, DAS-training, necessity, two-site, and validation call; its batch shapes and order; the exact forward,
backward, and update counts; every retained array and dtype; raw numeric bytes; cache policy; and hard maximum GPU
time. It must prove that SELECT/TEST/OOD and unrelated activations are absent. A producer and adapter may be built only
after independent approval of that exact compiler. No job may run from this document, and an implementation that
changes batching, steps, sites, donors, ranks, or retained evidence requires a new prospective amendment and review.

## Prior work used without duplicating it

The older task-14 locus result suggested an early number-sensitive head-position residual and a later read point, but
the natural two-head removal result was weak. Those facts motivate a boundary-wide residual search and explicitly do
not preselect L11H3, L15H5, or any whole head. Earlier generic DAS work also showed that single-seed derived ratios can
vary substantially and that rank sweeps can recover generic component damage without isolating a causal variable.
Accordingly this design uses five seeds, held-out donor/syntax transfer, direct finite effects, and ranks two/four only
as falsifiers. It does not repeat generic component-damage DAS or use low rank as its success criterion.

## CPU-only design verification

Before this freeze, a model-free reconstruction of the 128 FIT rows verified:

- 16 discovery and 16 validation groups with exactly four groups in every base subject-number $\times$ attractor-number
  cell on each side;
- every mirror pair $\{g,g+16\}$ remains together;
- 128 distinct endpoint prompts per half and zero prompt overlap between halves;
- eight distinct head-noun pairs per half and zero head-pair overlap;
- exact regeneration of partition digest
  `125b744d311088b3b6a41b144be51bacd81478212c71b5b82d04fef3548612ec`; and
- 704 complete donor records, 352 per half, with every answer-changing/P-positive donor opposite in subject number
  and matched in attractor number, and every P/C control matched in subject number; canonical donor digest
  `25a1f09d5947301f573b223abfbcae1699555ddf809f2b137eabffcbe776f3dc`.

This verification read only the frozen FIT authority. It did not load model values, capability evidence, checkpoint
bytes, activations, or any later split.

## What this document licenses

This document licenses only independent CPU review and construction of a future outcome-blind FIT localization
compiler/preregistration successor. It does not license implementation of the interventions, activation caching,
model or checkpoint access, GPU work, enqueue, result publication, component/head/MLP selection, bilinear weight
extraction, or opening SELECT/TEST/OOD. A future compiler must freeze exact calls, batches, retained arrays, numeric
bytes, runtime gates, and create-only namespaces before any model-facing producer is built.
