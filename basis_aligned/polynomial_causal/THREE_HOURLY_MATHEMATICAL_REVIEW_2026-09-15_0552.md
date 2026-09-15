# The retained three-block QK1 subtotal has exact bilinear rank two

Review clock: **2026-09-15T05:52:35.598373180Z**. Next mathematical
deadline: **2026-09-15T08:52:35.598373180Z**. At the first clock check, the
preceding review clock `2026-09-15T02:50:24.482516196Z` was about 175 minutes
25 seconds old, so the requested less-than-175-minute freshness stop had just
expired. This is one bounded active review and one executed CPU consequence,
not a review for an offline hour.

I read `session_recovery/bilin18-research-driver/SKILL.md` completely, plus the
installed copy, the current startup guide, `NEXT_CODEX_PROMPT.md`, the board
protocol and newest tail, the 03:02/04:01/05:01 hourly receipts, both registries,
the subject-number machine record, relevant attention/MLP dossiers and indexes,
and the current setting2 preregistrations, corrections, bindings, results and
Logan explanations. Historical `/workspace/tensor_language` paths were resolved
against this checkout for reading. I did not mutate queues, services, timers,
registries, dossiers, explanations, primary receipts, or concurrent work.

## Verdict

The current selected QK1 routing subtotal is not intrinsically a three-product
program. If the late group is (D=A_5+M_5+M_6+M_7), the remainder is (R), and
(B) is the first head9.8 QK score with its native denominators supplied, then

\[
B(D,D)+B(D,R)+B(R,D)=B(D+R,D+R)-B(R,R).
\]

Its group coefficient matrix is

\[
C=\begin{bmatrix}1&1\\1&0\end{bmatrix},\qquad \det C=-1,
\]

so its exact bilinear rank is two. At head width 128, two dot products, or 256
scalar multiplications, are both sufficient and necessary for a standalone
exact bilinear algorithm under the assumptions below; the direct three-dot
implementation uses 384. If the native complete QK1 parent score is already
live, the incremental selected score needs only the additional (R\times R)
dot and one subtraction. This is an exact execution simplification. It does
**not** repair the failed MLP16-to-MLP17 causal prediction, prove that the group
is a native semantic unit, or establish causal fidelity.

## Native computation as mathematical objects

### Whole model and common indices

Let vocabulary index (v=1,\ldots,50304), block
\(\ell=0,\ldots,17\), sequence positions (t,s=0,\ldots,T-1), residual
coordinates (i,j=1,\ldots,d) with (d=1152), heads
\(h=0,\ldots,8), head coordinates (a=1,\ldots,128), and MLP product
channels (m=1,\ldots,4608). The untied embedding and unembedding are
\(E,U\in\mathbb R^{50304\times1152}). A batch residual is
\(x_\ell\in\mathbb R^{B\times T\times1152}).

Each block uses two tied-across-position learned residual scalars
\(\lambda_{\ell,0},\lambda_{\ell,1}\):

\[
u_\ell=\lambda_{\ell,0}x_\ell+\lambda_{\ell,1}e,
\qquad \widehat u_\ell={u_\ell\over
\sqrt{1152^{-1}\|u_\ell\|^2+2^{-23}}}.
\]

For each head, native maps
\(Q_{1,\ell h},K_{1,\ell h},Q_{2,\ell h},K_{2,\ell h},V_{\ell h}
\in\mathbb R^{128\times1152}\) and
\(O_{\ell h}\in\mathbb R^{1152\times128}\) are reused at every position.
Queries and keys receive separate width-128 RMS normalization and the stored
BF16-rounded half-split rotary maps \(R_t\). With causal mask (s\le t),

\[
p_{\ell hts}={1\over128}
\langle R_t\widehat{Q_1u}_{\ell ht},R_s\widehat{K_1u}_{\ell hs}\rangle
\;{1\over128}
\langle R_t\widehat{Q_2u}_{\ell ht},R_s\widehat{K_2u}_{\ell hs}\rangle.
\]

There is no softmax. Values mix the current projection with the correspondingly
indexed layer-0 value using the learned scalar \(\mu_\ell\), which is not
assumed convex. The causal contraction graph is

\[
u\to(Q_1,K_1,Q_2,K_2,V)\to(s_1,s_2,v)\to
p=s_1s_2\to z_{ht}=\sum_{s\le t}p_{hts}v_{hs}
\to \sum_hO_hz_h.
\]

The MLP has
\(L_\ell,R_\ell\in\mathbb R^{4608\times1152}\),
\(D_\ell\in\mathbb R^{1152\times4608}\), and
\(b_\ell\in\mathbb R^{1152}\):

\[
M_\ell(y)=D_\ell[(L_\ell\widehat y)\odot(R_\ell\widehat y)]+b_\ell.
\]

The output is
\(30\tanh(U\widehat x_{18}/30)\). Consequently, the literal deployed program
is nonpolynomial because residual/head RMS square roots and the final tanh are
live. With all denominators fixed, one QK factor is degree two in its residual
sources, the two-factor attention numerator with current value is degree five,
and each MLP is degree two in its supplied normalized input. Polynomial degrees
quoted below are always for an explicitly fixed-background restriction, not the
complete native model.

Tied parameters include every position-shared linear map, each block's two
residual scalars, and the layer-0 value reused by later attention. Embedding and
unembedding are not tied. MLP-channel permutations, swapping Left/Right, and
reciprocal Left/Right scaling with decoder transport are gauges. A rank-one
write permits a sign gauge (u\mapsto-u,\alpha\mapsto-\alpha). Attention
query/key coordinate changes are gauges only when they preserve their separate
RMS forms, both score contractions, and the rounded rotary tables; an arbitrary
dense basis change is not licensed. The named residual-source partition below
is an intervention definition, not a free gauge.

### Current circuit object: subject number at L11H3

The canonical circuit record is `grammatical_subject_number.v34`. Its current
rank-one output object uses the top left singular direction
\(u\in\mathbb R^{1152}\) of the native L11H3 output slice. For an MLP8-input
background (x_b\in\mathbb R^{1152}), grouped MLP6/7 donor displacement
\(\delta_b\in\mathbb R^{1152}\), and the native head function
\(H:\mathbb R^{1152}\to\mathbb R^{1152}\), define

\[
z_b=u^TH(x_b),\qquad
s_b=u^T[H(x_b+\delta_b)-H(x_b)].
\]

The selected coefficient program is the degree-two scalar polynomial

\[
\alpha_b=\beta_0+\beta_z z_b+\beta_s s_b+\beta_{zs}z_bs_b,
\qquad w_b=\alpha_bu.
\]

Its four coefficients are fitted by leave-one-construction-out least squares;
the output to preserve is the signed finite-intervention change of the
`are`-minus-`is` logit contrast after the native suffix. The approximation norm
is relative (L_2) over the 512 opened-authority cells, with cosine and sign
agreement as secondary measures. The interaction form reaches `.37689`
relative (L_2), versus `.60931` for `[1,z]`. However (s_b) uses the exact
donor-dependent head response. The two-vector donor-free proxy predicts (s)
at `.28804` error but the composed coefficient fails at `.51585`; a rank-two
recipient-state proxy changes that only to `.51072`. Thus this is a causal
rank-one write and a predictive opened-row interaction screen, not a closed
donor-free generator.

The frozen interface receipt prices 3,460 scalars versus 13,824 for the earlier
12-write interface and four coefficient scalars versus ten, but the head,
upstream state generator, native suffix, and all model weights remain charged.
The response assay itself used one physical forward/96 role sequences plus
1,024 offline head-function evaluations and 12 scalar fits. Literal adoption
still costs the 545,902,902-parameter model because no independently executable
producer for (s_b) has passed.

### Current folded-path object: regional head9.8 QK1 routing

The task endpoint is a row-specific UK-minus-US reader
\(r_n=U_{v_{UK,n}}-U_{v_{US,n}}\in\mathbb R^{1152}\), evaluated on 96
controlled regional rows for discovery and 48 later rows for the recursive
edit. The MLP17 input split used earlier residual (e), propagated MLP16 write
\(p\), other attention17 output (o), and head17.2 output (a). Its ten
ordered self/cross terms are

\[
ee,pp,oo,aa,ep,eo,ea,po,pa,oa,
\]

where each unordered label in the receipts contains both Left/Right orderings,
for example

\[
B_r(e,p)=(r^TD_{17})[(L_{17}e)\odot(R_{17}p)
+(L_{17}p)\odot(R_{17}e)].
\]

On task-matched rows (ep) led at `.6184`; attention9 supplied `.26094` of
that term, and head9.8 supplied `.97762` of its aligned attention9 component.
This circuit-selected head is therefore the live folded target.

At block 9, the raw carry is exactly decomposed with learned propagation:

\[
c=\gamma_EE+\sum_{\ell=0}^{7}
(\gamma_{A,\ell}A_\ell+\gamma_{M,\ell}M_\ell)=D+R,
\quad D=A_5+M_5+M_6+M_7.
\]

For final query (t), causal key (s), and fixed native residual/head
denominators, put

\[
B(X,Y)_{ts}={1\over128}
\langle R_tQ_1\widehat X_t,R_sK_1\widehat Y_s\rangle.
\]

The 289-source census explicitly contained every self term and both ordered
cross terms. Fine-grained top-ten replay failed at `.66738`. The proposed
retained subset is the three terms touching (D):

\[
S_D=B(D,D)+B(D,R)+B(R,D),
\]

while (B(R,R)) is the matched omitted branch. QK2, current/inherited value,
head9.8 output projection, learned residual propagation, MLP16, MLP17, the row
reader, final RMS, softcap, and native suffix are backgrounds that remain live
or charged. In the restricted QK1 variables (D,R), (S_D) is exactly
bilinear (degree two). The complete native route is nonpolynomial.

The grouped attribution replayed selected rows at `.11186` error relative to
the parent paired change. On fresh recursive removal, it remained material
(`.52843/.59816` of whole-head effect), but the fixed MLP16×MLP17 folded suffix
predicted the opposite direction: cosine `-.92536` and `0/24` sign agreement.
The subsequent exact response census found direct propagated attention9
`.54223`, attention17 `.24952`, and opposing MLP17 `-.13785` aligned fraction;
the complete pre-RMS numerator predicted the logit effect at cosine `.99903`.
Accordingly, the current path object is a causally active routing edit plus an
unresolved downstream response graph, not an adopted folded circuit.

Literal local price before this review was 17 source states, 289 ordered terms
for exact discovery, then four grouped terms and three retained dot products.
The exact rank-two form reduces the retained score from 384 to 256 scalar
multiplications per query-key cell (33.3%), plus linear additions. When the
native parent score is already computed, only one extra 128-wide dot is needed.
All projections, states, QK2/value products, causal reductions, suffix weights,
and full-model storage remain charged; no total-model saving is claimed.

## Primary theorem and exact object mapping

[Grigor'ev's primary paper on rank and multiplicative complexity](https://logic.pdmi.ras.ru/~grigorev/pub/rank.pdf)
states, for a bilinear form over a field, that the smallest number of products
of linear forms is its ordinary coefficient-matrix rank. Map the live grouped
variables as

\[
x=(q_D,q_R),\qquad y=(k_D,k_R),\qquad
A=C\otimes I_{128},\qquad S_D=x^TAy/128.
\]

Because (operatorname{rank}(C)=2),
\(operatorname{rank}(A)=256\). The displayed two-dot identity attains this
lower bound. Rank computation is polynomial time by elimination; for this
Kronecker object it is immediate. Rank factorization is not unique because of
an internal (GL(2)) gauge, but the minimal multiplication count is unique.

Assumptions: real exact arithmetic; a scalar bilinear output for each query-key
cell; (q_D,q_R,k_D,k_R) allowed to vary independently; only multiplication
of linear forms is priced; and the native denominators are supplied. The actual
model violates independence on natural text and computes denominators from the
same residual. QK2, values, causal accumulation, downstream RMS, MLPs and tanh
also lie outside this theorem. It therefore gives an exact restricted executor
and lower bound, not recoverability, identifiability, or causal adoption.

The singular values of (C) are `1.61803` and `.61803`; Eckart--Young on this
matrix gives a best one-product relative Frobenius error `.35682` under an
isotropic coefficient norm. Approximate bilinear algorithms can behave
differently from exact rank in higher tensor settings, as emphasized by
[Bini, Lotti and Romani (1980)](https://doi.org/10.1137/0209053). Here the
two-by-two matrix rank-one set is closed, so a one-product exact executor is
impossible and its isotropic error cannot vanish. Natural-text covariance may
make empirical effect error smaller, so `.35682` is not a native-behavior lower
bound.

## Executed CPU consequence

The new [control](three_block_bilinear_rank_control_20260915_0549.py) and
[exclusive receipt](THREE_BLOCK_BILINEAR_RANK_CONTROL_20260915_0549_RESULT.json)
bind the current valid late-group result by SHA256. With CUDA hidden and two CPU
threads, deterministic FP64 arrays of shape `[7,19,128]` gave relative error
`5.1548e-16` for both the two-product identity and parent-minus-remainder form.
The coefficient determinant, rank, expanded (256\times256) rank, singular
values, exact/naive multiplication prices, and one-product Frobenius lower bound
were computed. Runtime was about 0.1 seconds; no model, text, fit, queue, or GPU
was used.

Executable consequence for the next folded implementation: represent the
selected three-block score as `complete_carry_score - remainder_self_score`.
Do not materialize or execute three separate block products except when their
individual attribution is itself the measurement. A one-product proposal is
mathematically falsified under the declared independent-source restriction.

## Organization reconciliation

- `CIRCUIT_REGISTRY.md` directly links both `grammatical_subject_number.v34`
  to `PATH-SUBJECT-001` and the inherited-city head8.2→head9.8-O circuit to
  `PATH-SET2-001`. The machine subject record's latest claim is v34 with status
  `weights_translated`; aliases `subject_verb_agreement`,
  `grammatical_subject_number`, and `task14` agree with the L11H3 dossier.
- `COMPUTATION_PATH_REGISTRY.md` now includes the carry-source, late-group,
  fresh-routing and downstream-response primary results and preregistrations.
  Its PATH-SET2 handoff correctly rejects the MLP16×MLP17 suffix and requests a
  direct-propagation plus attention17 split. This resolves the link debt stated
  in the 05:01 review through concurrent work that landed before this review.
- `MODULE_DOSSIERS.md` contains the subject response/proxy nulls and the regional
  attention9/head9.8, attention5/MLP5–7, fresh routing, and response-census facts.
  `MLP17_CURRENT_UNDERSTANDING.md` and the MLP index remain the relevant aliases
  for the late bilinear reader. The current seven primary results checked here
  each have three or four direct references across registries/dossiers/
  explanations; none is orphaned.
- V1/V2 invalid runners and correction notes remain preserved and bindings point
  to them where required. Registries correctly point to the frozen V1
  preregistration and the valid V3/V2 result rather than relabeling a repaired
  run as the original outcome.
- The Logan explanation indexes now point to the 05:34 update. In contrast,
  `NEXT_CODEX_PROMPT.md` is intentionally a September-13 migration snapshot and
  its claim that the setting2 native objective was not built is stale. The
  startup guide already says current board/receipts override it, so copying or
  rewriting that historical handoff would add risk and duplication.

## Efficiency audit

From the 05:01 review to the 05:34 publication, four substantive folding
receipts landed. Valid model executions were roughly three seconds each:
carry-source 05:03:49–05:03:52, late-group 05:11:32–05:11:35, fresh-routing
V3 05:24:42–05:24:45, and response-census V2 05:32:29–05:32:31. Compute and
queue capacity were not the bottleneck. At review time both managed services
and both review timers were active, both queue files had zero lines, and no GPU
job was launched by this review.

The main waste is repeated runner ceremony and shape-sensitive reporting. The
carry and grouped-attribution runners are 144/153 lines and repeat binding,
row bucketing, propagation, projections and scoring. Fresh routing required two
pre-result repairs (rotary broadcasting, then row-axis reporting); the response
census required one FP32 closure-tolerance correction. A new shared
`regional_grouped_interaction_tools.py` with rotary, explicit paired-row-axis,
source-partition and ordered-bilinear helpers plus model-free tests landed at
05:36. Completed bound runners should remain immutable; retrofitting them would
not speed an already completed experiment and would weaken receipt provenance.
Future attention17/QK splits should import that helper instead of cloning the
same 10–17 kB runner skeleton. This review therefore spends its one mutation
budget on the higher-value rank-two CPU consequence rather than a second code
refactor.

Other operational findings: the local namespace shim remains necessary for old
`/workspace` assertions, which is brittle but currently validated; direct
relative paths in the new control avoid adding another such dependency. The
25,000-line append-only board is expensive to scan, but its tail is authoritative
and truncating/rewriting it would violate coordination. Runner2's log contains
old nonmonotonic append segments across service restarts; queue files and current
service state, not raw tail order, are the live authority.

## Cross-track consequences and handoffs

Circuit evidence selected head9.8 through the inherited-city head8.2 value edge;
weight folding independently selected head9.8 inside the regional MLP17 path.
The fold then split that native head into a QK1 routing group (D/R), a distinct
QK2 branch, and a separately identified head8.2-derived value branch. The fresh
edit shows the QK1 group is causally material, while the response census says its
effect travels mainly through direct propagation and attention17, with MLP17
opposing it. Thus the folded algebra proposes a cross-module group
`{attention5, MLP5, MLP6, MLP7} -> head9.8 QK1`, but the causal circuit should
not merge that group with the head8.2 value transport or call MLP16×MLP17 its
suffix.

**WEIGHT_FOLDING handoff:** split the `.24952` attention17 response by all nine
heads, then by QK factor only for a stable head. Use the exact rank-two
`parent - R×R` executor for the existing QK1 edit; keep QK2 and value controls
distinct. Kill the proposed suffix if no attention17 head transfers across the
two fresh families or if direct propagation plus the selected head cannot
predict the signed numerator response.

**CIRCUIT handoff:** for subject number, derive a donor-free predictive-state
coordinate oriented by the frozen L11H3 response operator, not activation PCA
or wider lexical prototypes. For regional spelling, compare the QK1 rank-two
routing edit, head8.2 value midpoint edit, and their joint edit with opposing
predictions: multiplicative composition predicts the joint signed effect from
the two frozen branches; redundant/correlated attribution predicts subadditivity
or control spillover. Require fresh task effects and unrelated-reader controls.

## Limitations and stopping boundary

The rank result concerns one fixed-denominator QK1 restriction. It neither
compresses its source generators nor bounds empirical error on the native state
manifold. The causal edit is active but its proposed MLP suffix is falsified;
the attention17 decomposition is not yet performed. Subject-number response
interaction remains donor-dependent. No new OOD dataset, full-vocabulary
prediction, extraction, selective adoption, total runtime saving, or whole-model
price improvement follows. Algebraic replay and compression do not prove native
causal fidelity. Quantization was not used. This bounded review creates no goal,
agent, GPU job, timer/queue change, commit/push, or external contact.
