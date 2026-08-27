# Early-MLP matched-objective and transported-code discriminator v1

Status: **prospectively frozen for implementation after independent mathematical and
lifecycle GO.** It may not load a new fit, validation, intervention, or final row until every listed
artifact/lifecycle field is implemented, tested, committed, pushed, and independently
re-audited. It never reopens compiler-v2.1's spent roles or negative decision.

## Immutable inherited evidence

The only inherited numerical objects are:

| object | bytes | SHA256 |
|---|---:|---|
| `early_mlp_state_complete_compiler_v21_final_authority.json` | 11,767 | `659051ed8e2d34a2d755d1942f4112161294831e724d6697f4c3e2ef466f6987` |
| `early_mlp_state_complete_compiler_v21_final_result.pt` | 1,594,891 | `c73f2a7f6099de9e28550b02d7d02904fe37477c65cb8c5c9c6f4beed9bfb5cd` |
| `early_mlp_state_complete_compiler_v21_programs_receipt.json` | 11,269 | `c9c67bdd14a34dd83192a02d49705d0ed7043e2f9751d042250f44395f88ec2c` |
| `early_mlp_state_complete_compiler_v21_programs.pt` | 186,250,188 | `36a8e5203ec72d8c8f30909dba9241d1bf2a4a2d3fd980d8c558e28c3c0b614e` |
| `joint_early_mlp_pca_composition_authoritative_v3_bases.pt` | receipt-bound | `0eee01f39087548a479486d068404f78c4bdc2fd930932add162212da31fe4d9` |
| `joint_early_mlp_pca_composition_authoritative_v3_basis_receipt.json` | receipt-bound | `b81adb4c78255613997de4cbfc8ffd9e8eec233b40950915a14005ba3efcba0f` |

All paths are under `basis_aligned/bilinear_quotient/`. The inherited scientific
source commit is `bd9a58207b41b05c1b76f2be730771e89cab54ff`; all 60 source blobs bound by the
terminal authority must still hash exactly to that commit. The frozen ship identity
is `21ddc9ffdb7703aa570f88c5c7f4fa9fe007a988a1a7a3fd91058ee76a25ab8e`.

The canonical row source is `HuggingFaceFW/fineweb`, train, streaming order, revision
`9bb295ddab0e05d785b879661af7260fed5140fc`, ordered-manifest SHA256
`ba5e92b0d157f47cc6f8656eb1c37e46b7aac6957be8be68c1596736b98e6f90`,
source SHA256 `c84e6941d787b50959521df6d6894a91397c8b2db13f8a9c8fe0f8782872e930`,
`datasets==5.0.1`, and GPT-2 encoding SHA256
`0be287937901b1baae837369293dd6f63da1bece9609006e6485b57a3de37335`.

Compiler-v2.1's sealed final remaining-KL ratios were
$R_0=0.37329$, $R_1=0.56451$, and $R_{01}=0.66308$. Its executable pair gained
$0.05914$ CE against the projected-oracle pair's $0.22658$, or $26.10\%$.
Only the ratio and half-oracle gates failed. The best alternative same-family final
arm improved selected B by only $0.00257$ CE, so this protocol does not reopen the
old ridge/native-K grid.

## Question and identifiability

This experiment asks, in order:

1. On identical new rows, initialization, parameterization, optimizer, trial budget,
   and validation lifecycle, does direct suffix KL training outperform local
   coefficient-MSE training?
2. Conditional on that matched local baseline, does an explicit executable parent
   code $p_0^L$ improve observational and interventional MLP1/suffix response?

If both answers are no, v1 rejects only these two executable routes. It does **not**
distinguish predictor grammar from fixed-subspace inadequacy; that would license a
separate prospective oracle residual rank curve, not a post-hoc subspace conclusion.

## Fixed physical interface and gauge

Reuse immutable $B_0,B_1\in\mathbb{R}^{1152\times64}$ and Q0/Q1 only as the common
initialization. Neither basis, rank, deployed complement, precision, nor normalization
may change. Every local or suffix program retains

$$
\Delta m_l=\left(\widehat p_l(z_l)-m_lB_l\right)B_l^\top.
$$

The transport arm is defined on the **executable post-L0 code**
$p_0^L=\widehat p_0^L(z_0)$, never on native target coefficients. With the matched
local L0/L1 programs frozen, T changes only the MLP1 target code:

$$
\widehat p_1^{LT}(z_1,p_0^L)=\widehat p_1^L(z_1)+p_0^L A,
\qquad A\in\mathbb{R}^{64\times64}.
$$

It is intentionally uncentered, so no unpaid $\bar p_0$ exists. The physical linear
operator is $\tau=B_0AB_1^\top$. Under $B_l\mapsto B_lQ_l$, executable row codes
transform as $p_l\mapsto p_lQ_l$ and $A\mapsto Q_0^\top A Q_1$.

The incremental T price is exactly 4,096 float values and 4,096 multiplies/token,
conditional on fully pricing L0/L1 and both bases. Standalone and conditional prices
are both reported. There is no factorized-rank option in v1.

Gauge matrices are integrity replays, never nulls. Freeze exactly eight in the
preflight receipt:

- four signed permutations, seeds `2026082801+i`, `i=0..3`, using CPU float64
  `torch.randperm(64)` followed by 64 CPU `torch.randint(0,2)` signs;
- four Haar orthogonal matrices, seeds `2026082810+i`, `i=0..3`, from CPU float64
  standard normals and reduced QR, with each Q column flipped so the corresponding
  R diagonal is nonnegative.

Serialize all eight tensors and raw-tensor hashes before fitting. A complete replay
rewrites $B_0,B_1,A$, both affine-program weights and biases, all executable and
teacher-label code tensors, and every intervention $\delta$ by the corresponding
$Q_l$. It must preserve physical maps and every scored row to tolerance
$2\cdot10^{-6}$. Rewriting only the bases and $A$ is an integrity failure.

## Deterministic fresh roles

Candidate triple $j\ge0$ is

$$
(384,43000+12000j),\quad(192,47000+12000j),\quad(192,51000+12000j)
$$

for fit, validation, and final. A CPU-only builder constructs one triple at a time in
canonical source order, then checks ordered document IDs, full rows, and prefix-32
rows against one another and every provenance set in the v3 basis, compiler-v2,
compiler-v2.1, frozen-ship, code-OOD, and 36-site held-out registries. On collision it
records only hashes/provenance in a collision manifest, deletes temporary candidate
files, increments $j$, and repeats. No model process may deserialize a candidate
before the first collision-free triple's receipt is the last-written CPU authority.
The chosen $j$, skips, tensors, source records, registry census, tokenizer/corpus
identities, and all hashes are then immutable.

## Matched L and R programs

Teacher labels are exact projected native coordinates recomputed at each student's
current autoregressive state. OON teacher logits use exact live rank-64 projected
restoration at MLP0/1 with deployed MLP2. Original MLP calls are allowed only in the
separate label/teacher route during fitting; student L/R/T calls are poisoned and
must remain zero. Validation and final permit original calls only in named O arms.

Both L and R initialize from the exact frozen Q full products
$W_l=L_lR_l\in\mathbb{R}^{1152\times64}$ and biases. Means, scales, and bases stay
fixed. Both use float32 parameters/forward/backward, float64 accumulated reporting,
AdamW betas `(0.9,0.999)`, epsilon `1e-8`, zero weight decay, gradient-norm clip 1,
batch size 4, three epochs, and learning rates `[1e-5,3e-5,1e-4]`. For trial `t`
and epoch `e`, the fit-row permutation uses CPU `torch.randperm` seed
`2026083000 + 100*t + e`; batches transfer in that order. No early stopping or
scheduler is permitted.

L and R have identical trainable tensors, batches, steps, initialization, and trial
count. Before either optimizer step, an initial-trajectory label pass computes each
site's centered second moment in float64 using the initialized Q programs on all fit
rows. These two denominators are serialized in the fit receipt and never updated.
During optimization, exact labels are recomputed at the current autoregressive
student state and immediately detached: no gradient may pass through label or teacher
construction. Their sole scientific difference is loss:

- **L local comparator:** sum of site-0 and site-1 coefficient MSE, each divided by
  its frozen fit-label second moment. Labels are captured at the current
  autoregressive student state.
- **R suffix program:** token-weighted
  $\mathrm{KL}(\mathrm{OON}\Vert\mathrm{RRN})$ through the complete suffix.

Fit joint L0/L1 and R0/R1 packages. Also fit suffix site0-only S0 with L1 frozen and
suffix site1-only S1 with L0 frozen, under the same three trials, to localize
independent opportunity. R0/R1 factorial removals are explicitly co-adapted package
ablations, not independently fitted site effects.

Select L by lowest unrounded validation local loss and R/S by lowest unrounded
validation suffix KL, subject only to finite values, exact support, student poison,
and copy worsening $\le0.01$. Ties use smaller learning rate, then lexicographically
smaller logical tensor hash. After selection, serialize each full W by deterministic
CPU float64 SVD: `U,S,Vh=torch.linalg.svd(W.double(),full_matrices=False)`, flip each
U column and matching Vh row so U's largest-absolute loading is nonnegative, and store
`left=U*S`, `right=Vh`. Reloaded products must match W within $2\cdot10^{-6}$.

Each route also trains a same-budget document-block-shuffled-label control: projected
coordinate labels for L and OON teacher-logit document blocks for R/S. Before fitting,
documents are partitioned into equal-row-count strata. Every stratum must contain at
least two distinct documents or preflight fails. Seed 2026083050 selects a nonzero
cyclic offset independently within each stratum; the resulting permutation must have
no fixed document IDs. New fit means are separately scored controls, not inherited
v2.1 controls.

## Transport fit and true nulls

Freeze selected L0/L1. Fit dense A against
$\mathrm{KL}(\mathrm{OON}\Vert\mathrm{LTN})$ with the identical optimizer grid,
epochs, batch order, and selector/tie-break used for R. A initializes to zero; L0/L1
never update. Fit 20 matched A-null programs after permuting whole-document parent
code blocks relative to their MLP1/suffix labels. Reuse the frozen equal-row-count
strata above. Null `i` uses seed `2026083100+i` to select an independently sampled
nonzero cyclic offset within every stratum; every realized map must have no fixed
document IDs. At evaluation, every A-null receives the true live executable L0 code;
only its learned pairing is false. Zero-A and a new-fit mean program are deterministic
controls. Orthogonal gauge rewrites are exact replay checks and are forbidden from
the null bank.

## Observational lattice

On validation and one-shot final, score common raw sufficient statistics for QQ; LL;
S0-L1 and L0-S1; RR and co-adapted removals R0-L1 and L0-R1; LT, zero-A, all 20 A
nulls, shuffled L/R controls; and N/N, O/O, and new-fit mean controls. Score all named
primary arms under deployed MLP2 N and exact-restored MLP2 E.

Use identical integer support, CE, OON teacher KL, copy, and the inherited nine token-
frequency bins. OON teacher KL is defined and gated only with deployed MLP2 N; exact
MLP2 E is a CE-only alternate-background retention check. Bootstrap 2,000 shared
source-document draws with seed 20260832.

## Non-tautological transport intervention

Validation chooses one amplitude from `[0.01,0.03,0.1,0.3,1.0]` times a frozen fit
RMS. After L selection, let $X$ contain every selected-L0 executable code recomputed
on the fit rows with L1 present; it is not the inherited Q0 trajectory. In float64 compute
$\bar x$, $C=(X-\bar x)^\top(X-\bar x)/(n-1)$, and
$r=\sqrt{\operatorname{mean}((X-\bar x)^2)}$. Eigendecompose $C$ and clip each
eigenvalue below $10^{-12}\operatorname{tr}(C)/64$ to that floor. Seeds
`2026083200+i`, `i=0..31`, generate CPU float64 Rademacher rows $s_i$; define
$v_i=s_iC^{1/2}$ and normalize each to
$\sqrt{\operatorname{mean}(v_i^2)}=1$. Both signs are used, so the intervention bank
contains 64 antithetic directions. The covariance, eigensystem, clipped spectrum,
$r$, raw signs, and normalized directions are serialized before calibration.

Selection uses **only the exact OON teacher**: among amplitudes whose median teacher
edited-vs-unedited suffix KL is in `[0.01,0.20]`, choose the one nearest geometric
center $\sqrt{0.01\cdot0.20}$, ties to the smaller amplitude. If none lies in the
band, choose the amplitude closest to that center anyway and set the calibration gate
false. Candidate/null responses are not observed during calibration. A failed
calibration makes the transport route fail but does not prevent the objective route
from receiving its registered final evaluation.

Validation and final each contain 192 rows. For validation, seed 2026083240 samples one
integer edit position per row uniformly from inclusive token indices `[64,255]`, and
seed 2026083241 produces a permutation of row indices; row at permutation position $k$
gets direction $k\bmod32$. Final uses position seed 2026083250 and permutation seed
2026083251 by the identical rule. Therefore every base direction occurs exactly six
times per role. Both signs are evaluated in separate arms, giving 384 edited
occurrences per amplitude and role. Every compared program sees the same realized
positions, directions, signs, and amplitude. All assignments are serialized before
teacher calibration. For a realized direction $v$, $\delta=\alpha r v$.

On final, intervene on executable code and physical write together:

$$
p_0^L\mapsto p_0^L+\delta,
\qquad m_0\mapsto m_0+\delta B_0^\top.
$$

The exact OON teacher receives the identical physical MLP0 edit and recomputes its
exact projected MLP1 response. Teacher and student changes are measured relative to
their own unedited baselines.

The explicit-T difference-in-differences

$$
[p_1^{LT}(\delta)-p_1^{LL}(\delta)]
-[p_1^{LT}(0)-p_1^{LL}(0)]=\delta A
$$

is an implementation identity and must replay within $2\cdot10^{-6}$; it is not
scientific transport evidence. Scientific metrics compare total LL/LT/null responses
with exact teacher response: MLP1-coordinate NRE, centered-logit-response NRE,
response $R^2$, cosine, and output KL. Logit deltas are compared with logit deltas;
they are never compared directly with $\delta A$.

For each occurrence $o$, let $t_o$ and $s_o$ be respectively the teacher and student
edited-minus-unedited response vectors. MLP1-code responses use projected exact MLP1
coordinates; logit responses first subtract the per-token vocabulary mean. Pool over
occurrences before taking ratios:

$$
\operatorname{NRE}=\sqrt{\frac{\sum_o\lVert s_o-t_o\rVert_2^2}
                               {\sum_o\lVert t_o\rVert_2^2}},\qquad
R^2=1-\operatorname{NRE}^2,
$$

$$
\operatorname{cos}=\frac{\sum_o s_o^\top t_o}
 {\sqrt{\sum_o\lVert s_o\rVert_2^2}\sqrt{\sum_o\lVert t_o\rVert_2^2}}.
$$

Any NRE denominator or cosine norm product $\le10^{-12}$ is an integrity failure.
The output-KL response ratio is

$$
\frac{\sum_o\mathrm{KL}(p^{\mathrm{teacher,edit}}_o\Vert
                         p^{\mathrm{student,edit}}_o)}
     {\sum_o\mathrm{KL}(p^{\mathrm{teacher,edit}}_o\Vert
                         p^{\mathrm{teacher,base}}_o)},
$$

with denominator $\le10^{-12}$ also an integrity failure. The 2,000 shared
source-document bootstrap draws recompute every pooled numerator, denominator, ratio,
and difference from sufficient statistics; they never bootstrap already-computed
occurrence ratios.

The single finite-null statistic is

$$
G_T=\operatorname{NRE}_{\mathrm{logit}}(LL)
-\operatorname{NRE}_{\mathrm{logit}}(LT),
$$

higher being better. Its integer finite-null rank is
$1+\#\{i:G_i\ge G_T\}$ among the 21 fixed values. This is reported as a finite-null
rank in `[1,21]`, not an asymptotic or randomization p-value. Other response metrics
are separately gated but do not choose the null orientation.

## Gates and interpretation

Common gates require exact source/artifact/row bindings, baseline replay, one final
load and callback, student original-call poison, full support, eight gauge replays,
SVD replay, protected snapshots, hook restoration, unchanged component tree, and
create-only terminal authority.

The **objective route** passes only if:

1. RR beats LL in suffix CE and OON teacher KL with paired 95% intervals wholly
   favorable;
2. R joint remaining-KL ratio is $\le0.50$ and its CE half-oracle margin has 95%
   lower bound above zero;
3. RR beats both co-adapted removals, and at least one independently fitted suffix
   singleton beats LL with a positive Bonferroni-adjusted 97.5% two-sided interval
   across the two singleton tests;
4. RR's CE advantage over LL remains positive under N and E backgrounds, while its
   OON teacher-KL advantage remains positive under N;
5. RR beats its same-budget shuffled-teacher control and new-fit mean, copy worsening
   is $\le0.01$, and every nonempty frequency bin worsens by $\le0.01$.

Only RR>LL identifies the objective; RR>QQ alone is descriptive because Q used old
rows and a different solver.

The **transport route** passes only if calibration passes; LT beats LL in
observational CE under N and E and OON teacher KL under N with positive full
intervals; intervention improvement in both MLP1-code and logit NRE has positive
point estimate and 95% lower bound; logit-response NRE has point estimate and 95%
upper bound $\le0.50$; response $R^2$ has point estimate and 95% lower bound
$\ge0.75$; $G_T>0$ and its integer finite-null rank is exactly `1`; copy/frequency bounds
also hold. The exact difference-in-differences identity is integrity only.

- Objective pass with L failing its registered ratio or half-oracle gate: local
  coefficient loss is identified as a primary failure of the previous compiler.
- Objective pass with L also passing those gates: suffix KL is the superior matched
  objective, but local loss is sufficient and is not identified as the old failure.
- Objective fail, transport pass: the matched local program lacked explicit
  executable parent-code transport.
- Both fail: reject these two routes only and preregister an oracle residual rank
  curve before changing grammar or subspace.
- Integrity failure: no scientific interpretation or outcome authority.

No outcome moves current-ship, 36-site, named-behavior, named-causal, semantic, OOD,
edit, or whole-model ledgers without a separate common-denominator experiment.

Every interval above uses the same 2,000 frozen source-document bootstrap replicates,
recomputed in float64. A two-sided 95% interval is
`torch.quantile(replicates, tensor([0.025,0.975], dtype=float64),
interpolation="linear")`; equivalently it linearly interpolates sorted indices
$(B-1)q$. The Bonferroni-adjusted singleton intervals use quantiles
`[0.0125,0.9875]`. No standard-error, row-level, BCa, or post-hoc interval may replace
these rules.

## Artifact DAG and terminal namespaces

Use lock `/workspace/runs/.early_mlp_suffix_transport_v1.lock` and exact create-only
outputs:

1. `early_mlp_suffix_transport_v1_rows_receipt.json`, rows manifest, collision
   manifest if needed, and `.rowcache_early_mlp_suffix_transport_v1/`;
2. `early_mlp_suffix_transport_v1_fit_ledger.pt`, fit manifest, and fit receipt;
3. `early_mlp_suffix_transport_v1_programs.pt` and programs receipt;
4. `early_mlp_suffix_transport_v1_final_attempt.json` before final deserialization;
5. `early_mlp_suffix_transport_v1_final_result.pt`;
6. `early_mlp_suffix_transport_v1_final_manifest.json`;
7. last-written `early_mlp_suffix_transport_v1_final_authority.json`, positive or
   scientific-negative, only after all hashes and protected snapshots revalidate;
8. `early_mlp_suffix_transport_v1_integrity_failure.json` only when integrity fails,
   with no outcome authority.

Every implementation, test, model loader, row builder, inherited runtime, statistics,
and transitive model-forward source must be enumerated in the source closure and
committed/pushed before the CPU row builder. Protected snapshots include every
inherited object above, all row registries, the frozen ship pair, and prior compiler
outputs. Final is forbidden before the immutable programs receipt; training is
forbidden after it. Outer model return, hook restoration, protected equality, and
result/manifest hashes are revalidated immediately before the authority's sole last
write.
