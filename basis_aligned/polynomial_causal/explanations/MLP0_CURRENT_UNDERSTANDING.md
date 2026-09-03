# What We Currently Understand About MLP0

*Last updated: 2026-09-01. Compiler-v2.1's sealed final evaluation is complete and
negative at the registered joint-composition gate. Rungs 394–401 subsequently closed the token-only anatomy and
established an exact natural-context causal grammar; no whole-model executable claim is made from that attribution.*

*2026-09-01 addendum: rung394 exhaustively measured the token-only bias-free-write removal response for all
50,257 tokens. The new result is summarized in §13 below and supersedes the “small correction” intuition without
changing the earlier contextual/compiler receipts.*

## Short version

MLP0 is not best understood as assigning every token to one discrete class.

It is better understood as a **quadratic feature generator**. It computes many
continuous features from the current residual stream, and then writes a mixture of
those features back into the residual stream.

Tokens such as numbers, punctuation, and related word forms really do share some
structure. But members of one such group are not mapped to exactly the same state.
They share part of their representation while retaining token-specific and
context-specific information.

The simplest current picture is therefore:

$$
\text{MLP0 output}
=
\text{shared lexical structure}
+
\text{token-specific refinement}
+
\text{contextual refinement}.
$$

This resolves the apparent conflict between:

- “numbers form a cluster”; and
- “downstream computation distinguishes the tokens.”

Both can be true. Numbers can share a common feature without becoming identical.

We also already have direct evidence that this continuous computation is
compressible. An older 256-feature quadratic surrogate reproduced about 97.8–97.9%
of MLP0 under its legacy behavioral denominator, and the later C512 program replaced
the native `Down` map by a rank-512 factorization with about 72% fewer `Down` bytes
while retaining small ordinary final-output point errors. These are approximate
behavioral low-rank results, not a proof that the original `Down` matrix or the exact
stack of all downstream readers has low algebraic rank.

---

## 1. What MLP0 computes exactly

Let $x$ be the residual-stream state immediately before MLP0's normalization. First,
MLP0 applies RMSNorm:

$$
z = \operatorname{RMSNorm}(x).
$$

It then applies two linear maps, called `Left` and `Right`:

$$
\ell = Lz,
\qquad
r = Rz.
$$

These two vectors are multiplied coordinate by coordinate:

$$
h_k = \ell_k r_k
    = (L_k z)(R_k z).
$$

Finally, the `Down` matrix mixes the product features into the 1152-dimensional
residual stream:

$$
m_0(z) = Dh(z) + b.
$$

Putting it together:

$$
\boxed{
m_0(z)
=
D\big((Lz)\odot(Rz)\big)+b
}
$$

where $\odot$ denotes coordinatewise multiplication.

This is an exact description, not an approximation.

### Why this is a tensor or polynomial computation

Each output coordinate of MLP0 is a quadratic polynomial in $z$:

$$
m_{0,j}(z)
=
\sum_k D_{jk}(L_kz)(R_kz)+b_j.
$$

Equivalently, each output coordinate can be represented by a quadratic matrix
$Q_j$:

$$
m_{0,j}(z)=z^\top Q_j z+b_j,
$$

with

$$
Q_j
=
\sum_k D_{jk}\operatorname{sym}(L_k^\top R_k).
$$

Thus the invariant mathematical object is a tensor containing the quadratic forms
$Q_j$. The individual hidden units are one factorization of that tensor.

---

## 2. Why the hidden-unit decomposition is not unique

MLP0 has gauge freedoms. For example, for any nonzero scalar $a$,

$$
L_k \longrightarrow aL_k,
\qquad
R_k \longrightarrow a^{-1}R_k
$$

does not change the product:

$$
(aL_kz)(a^{-1}R_kz)=(L_kz)(R_kz).
$$

We can also permute hidden units if we apply the inverse permutation to `Down`.

Therefore statements such as “hidden unit 417 is the number unit” are not generally
gauge-invariant. A more robust explanation should be expressed in terms of:

- quadratic subspaces;
- output subspaces;
- causal effects;
- or a chosen canonical gauge.

This is one reason the tensor structure is useful: it lets us distinguish the exact
function from an arbitrary factorization of that function.

---

## 3. What the token clusters actually mean

The clustering experiments have found real lexical organization. Tokens with related
roles often have nearby or partially shared representations. Examples include:

- numbers and number-like tokens;
- punctuation;
- related morphological forms;
- function words;
- related subwords.

The lexical assignments also beat controls in which the same centroids are retained
but their token assignments are deranged. This is evidence that the lexical grouping
contains real information.

But this does **not** imply

$$
m_0(\text{``1''})=m_0(\text{``2''}).
$$

A better model is

$$
m_0(t,c)
=
\mu
+
S\,a(t)
+
U\,q(t,c)
+
\varepsilon(t,c),
$$

where:

- $t$ is the current token;
- $c$ is its context;
- $a(t)$ is a shared lexical code;
- $q(t,c)$ is a continuous token-and-context refinement;
- $\varepsilon(t,c)$ is the remaining error.

Two numbers may share much of $a(t)$ while differing in $q(t,c)$. Downstream
computation can use either part.

So the useful statement is:

> Token groups share components of their representation.

The overly strong statement is:

> Every token in a group is replaced by exactly the same state.

We have evidence for the first statement, not the second.

### 3.1 How the “shared lexical code” and “continuous refinement” are computed

These names describe a fitted decomposition; they are not labels read directly from
one native hidden unit.

On fit-only documents, we capture the native MLP0 write $m_0(t,c)$. The dense
token code is the conditional fit mean

$$
T[t]
=
\frac{1}{N_t}\sum_{i:t_i=t}m_0(t_i,c_i),
$$

with a registered backoff for unseen tokens. It answers: “what part of the write is
predictable from token identity alone?” It is evaluated on different documents, so
memorizing fit positions cannot by itself produce held-out fidelity.

For a lexical partition $g(t)$, the shared class atom is the fit-frequency-weighted
mean of the token codes assigned to that class:

$$
C[g]
=
\frac{\sum_{t:g(t)=g}N_tT[t]}
     {\sum_{t:g(t)=g}N_t}.
$$

The within-class token refinement is then $T[t]-C[g(t)]$. The hierarchy experiments
physically serialized the class assignments and occupied centroids, and compared
them with assignment-preserving derangements. That is why we can say lexical
organization is real: the meaningful assignments beat those nulls. But meaningful
organization did not beat a matched-byte continuous map or satisfy the causal
interface gate, so it has not earned simplicity credit.

The continuous context term is fitted to what the token code misses. In the
token-plus-context family this is a held-out low-rank regression

$$
R(t,c)=m_0(t,c)-T[t]
\approx [a_0(t,c);x_0(t,c)]W_R,
$$

where $a_0$ is the same-block attention output and $x_0$ is the inherited embedding
state. In the tensor-native family, the residual is instead predicted by learned
quadratic features $(a_r^\top z)^2$. In C512, we retain the exact native product
state $h=(Lz)\odot(Rz)$ and fit a rank-512 continuous map from $h$ to the write.

So the decomposition is operational:

$$
\text{class mean}
+\text{within-class token residual}
+\text{context-predicted residual}
+\text{unexplained residual}.
$$

It is not yet a unique latent ontology. Different gauges can move information among
these terms; the terms become scientifically useful only when their total producer,
consumer, and residual price is smaller at matched causal fidelity.

### 3.2 Do not conflate the lexical decomposition with the rank-64 causal code

The symbol $a(t)$ above is a descriptive token/class factor inferred from native
MLP0 writes. The later rank-64 vector $p_0(z)=m_0(z)B_0$ is a continuous coordinate
in a causally selected output subspace. It is not a class label. A compiler is a
third object: an executable program that predicts such coordinates without calling
native MLP0, installs them in the live state, and composes with downstream consumers.

Compiler-v2.1 compiled the continuous rank-64 interface, not the class-centroid
decomposition. The class hierarchy remains a possible producer grammar, but current
matched-price and causal evidence did not justify selecting it. A focused explanation
and runtime audit is in `MLP0_COMPILER_AND_RUNTIME_NOTE_2026-08-28.md`.

---

## 4. Why “downstream computation separates everything” is not the end of the story

Suppose the declared downstream readers are linear maps $A_1,\ldots,A_n$. We could
define exact equivalence by

$$
u\sim v
\quad\Longleftrightarrow\quad
A_i u=A_i v
\text{ for every }i.
$$

Equivalently,

$$
u-v\in\bigcap_i\ker(A_i).
$$

For the combined block-1 readers, this exact kernel is effectively trivial. That
means almost any two distinct MLP0 writes are mathematically distinguishable by at
least one reader.

This is an exact, zero-error notion of equivalence. It does not say that all
distinctions are equally important.

For reverse engineering, the more useful question is approximate causal
equivalence:

$$
u\sim_\epsilon v
\quad\Longleftrightarrow\quad
D_{\mathrm{causal}}(u,v)\leq\epsilon.
$$

Here $D_{\mathrm{causal}}$ measures the downstream behavioral consequences of
replacing $u$ with $v$.

This is analogous to image compression. Two images are almost never exactly equal
pixel by pixel, but they can still share a compact description and be perceptually
indistinguishable at a chosen error tolerance.

Thus:

- exact downstream rank tells us whether a distinction is literally invisible;
- causal rate-distortion tells us whether retaining that distinction is worth its
  description cost.

The second is the relevant notion of simplicity for this project.

---

## 5. The earlier continuous and low-rank results

There is not just one “MLP0 is low-rank” result. There are several results about
different maps and different behavioral denominators. They should not be collapsed
into one number.

### 5.1 The pre-1500 tensor-native quadratic program

The earlier tensor-native program directly approximated the complete normalized
input-to-write function as

$$
\widehat m_0(t,z)
=
T[t]
+
\sum_{r=1}^{R}u_r(a_r^\top z)^2.
$$

This uses $R$ learned quadratic features rather than the original 4608 native
products. Under its legacy completeness-ledger denominator, the measured
substitution results were:

| program | final $\Delta\mathrm{CE}$ | legacy understood fraction |
|---|---:|---:|
| token table only | 0.34910 | 90.4% |
| token table + 64 quadratic features | 0.11366 | 96.9% |
| token table + 256 quadratic features | 0.07480 | 97.9% |
| 256 quadratic features without the table | 0.07807 | 97.8% |

This is the result summarized in the old commit as “256 quadratic features =
97.9%.” It is important because it shows that a small continuous quadratic program
can reproduce most of MLP0's ordinary substitution behavior. It also shows that the
quadratic program alone nearly matches the table-plus-quadratic program, so a giant
token table is not essential to that particular fidelity result.

Its `97.9%` is a legacy, mean-floor-relative score. It was not evaluated with the
later source-document authority, simultaneous worst-cell inference, physical byte
pricing, or C512's internal MLP1 interface gates. It is therefore strong discovery
evidence, but it is not interchangeable with the later causal certificate currency.

### 5.2 The canonical token-plus-continuous-context program

A later canonical program used

$$
\widehat m_0(t,a_0,x_0)
=
T[t]+[a_0;x_0]W_R,
$$

where $T[t]$ is a token-indexed write and $W_R$ is a rank-$R$ continuous correction
from the attention-0 output and embedding state.

On its frozen held-out FineWeb protocol, the token-only arm retained 84.33% of the
constant-to-live MLP0 gap. The rank curve rose monotonically:

| correction rank | retained fraction |
|---:|---:|
| 0 | 84.33% |
| 32 | 86.97% |
| 64 | 87.67% |
| 128 | 88.51% |
| 256 | 89.39% |
| 1152 | 90.26% |

This is a different result from the tensor-native 256-feature program. It decomposes
MLP0 into a token table plus a continuous inherited-context correction and uses a
more conservative held-out operational protocol.

There is also a useful structural bound on this decomposition. MLP0 is position-wise
and, at layer 0, its complete input is the current-token embedding plus attention-0's
write. Therefore $(t,a_0)$ is an information-complete regressor set for MLP0; adding
more upstream variables cannot reveal information MLP0 itself never receives. With
attention-0 frozen, the covered-token table ceiling is exactly 100.00%. With
attention-0 live, the same covered-position protocol assigns only about 9.73% of
MLP0's 0.855-nat stake—roughly 0.083 nats—to context plus unexplained residual.

This bound depends on coverage policy. Leaving MLP0 live for unseen fit tokens gives
a 90.27% covered-position table ceiling, whereas substituting a backoff everywhere
can lower the measured recovery by about 15.9 points through propagated uncovered
token errors. Any comparison of token and context terms must state which policy it
uses.

### 5.3 The immediate MLP0-to-MLP1 edge is approximately low-rank

At the declared MLP0-to-MLP1 edge, an earlier causal edge-rank screen found that a
rank-32 connection recovered about 81.4% of the measured edge gap and rank 128
recovered about 91.95%.

That is evidence that the part of MLP0 consumed by a particular downstream edge is
much thinner than the complete write. It does not imply that the stack of every
immediate reader is exactly low-rank: the literal combined reader stack has rank
$1152/1152$.

### 5.4 C512: a low-rank replacement of `Down` itself

The cleanest later result specifically about the native `Down` map is **C512**.

C512 retains:

- the exact RMSNorm input;
- the exact `Left` map;
- the exact `Right` map;
- all native coordinatewise product features.

It replaces only the original `Down` map with a rank-512 continuous program. Thus it
is narrower than the old 256-feature input-to-output surrogate, but its object and
physical price are much more explicit.

The original float32 `Down` matrix costs approximately

$$
1152\times4608\times4
=
21{,}233{,}664\text{ bytes}.
$$

The serialized C512 program costs

$$
5{,}904{,}640\text{ bytes}.
$$

Therefore C512 makes `Down` about

$$
\frac{21{,}233{,}664}{5{,}904{,}640}
\approx 3.60
$$

times smaller.

This is about a 72% reduction in the size of `Down`.

On 384 previously unseen FineWeb source documents, divided into two independent
192-document waves, C512 had small final-output point errors:

$$
\max\operatorname{KL}\approx0.005326
$$

against a point margin of $0.01$, and

$$
\max\Delta\operatorname{CE}\approx0.005492
$$

against a point margin of $0.0075$.

These results were stable across both waves.

This is strong evidence that much of the original `Down` matrix is unnecessary for
ordinary final-output behavior. Together with the earlier quadratic program, it
supports the user's original summary: **MLP0 has a useful continuous low-rank
description.** The remaining question is which low-rank description is causally
sufficient and composable, not whether any such description exists.

It is not yet a compression of all MLP0. The exact `Left` and `Right` maps still cost
42,467,328 bytes under the registered physical accounting.

---

## 6. MLP0 has a concentrated causal output subspace

A separate oracle experiment found that a rank-64 MLP0 output basis captures only
about 37% of the residual-error energy, but recovers about 79.9% of MLP0's held-out
causal CE effect:

$$
\frac{0.09254}{0.11584}\approx0.799.
$$

The selected subspace also beat all 20 matched random rank-64 subspaces under both
downstream-KL and raw-RMS strength controls.

This tells us something important:

$$
\text{causal importance}
\neq
\text{activation variance}.
$$

A relatively low-dimensional part of the MLP0 write carries most of its measured
causal usefulness.

This result is currently an oracle result. It identifies the target subspace, but it
does not yet give an executable program for predicting the correct coefficients in
that subspace.

### 6.1 What the new rank-64 continuous code means

Compiler-v2.1 turns that oracle subspace into an executable interface. Let

$$
B_0\in\mathbb{R}^{1152\times64}
$$

be the registered MLP0 output basis, let $z_0\in\mathbb{R}^{1152}$ be MLP0's
normalized residual input, and let $m_0(z_0)$ be the native MLP0 write. The basis
was obtained as the top 64 uncentered PCA directions of the native-minus-deployed
MLP0 write residual on the frozen basis split; downstream causal tests subsequently
showed that this subspace is far more useful than its raw energy fraction suggests.
The target code is

$$
p_0(z_0)=m_0(z_0)B_0\in\mathbb{R}^{64}.
$$

The selected `B_l5_r64` program predicts these coefficients with a rank-64 affine
map (the `l5` means ridge $\lambda=10^{-2}$, not layer or lag):

$$
\widehat p_0(z_0)
=b_0+
\left(\frac{z_0-\mu_0}{\sigma_0}\right)L_0R_0,
$$

where $L_0\in\mathbb{R}^{1152\times64}$ and
$R_0\in\mathbb{R}^{64\times64}$. At runtime, if $m_{\mathrm{old}}$ is the write
already present in that execution, the state-complete correction is

$$
m_0^Q
=m_{\mathrm{old}}
+\left(\widehat p_0(z_0)-m_{\mathrm{old}}B_0\right)B_0^\top.
$$

Thus the program replaces the write *inside the 64-dimensional causal interface*
while leaving its orthogonal complement as it was. It is continuous: tokens and
contexts may occupy distinct points in the code even when downstream computation
only reads a small number of shared directions. It is not a partition of tokens
into 64 clusters.

The standalone registered price is

$$
|B_0|+|L_0|+|R_0|+|\mu_0|+|\sigma_0|+|b_0|
=153{,}920
$$

float values, or $0.966\%$ of the registered $15{,}926{,}400$ values in native
MLP0. On mapped validation it recovers $66.07\%$ of the local OON-versus-NON
teacher-KL denominator and improves the copy collateral. The matched shuffled
pipeline is negative. This is evidence that the code predicts real input-dependent
state rather than merely adding a useful low-rank bias.

The $0.966\%$ figure must not be read as a complete $0.966\%$ replacement of all
MLP0 computation. The program is state-complete relative to an already deployed
write $m_{\mathrm{old}}$: it overwrites the 64 registered coefficients, while the
other 1,088 output directions remain whatever the deployed baseline produced. Its
strong claim is that an important missing causal slice is extremely cheap to
compile. A complete MLP0 compression claim would additionally have to price and
validate the baseline/complement, or show that the complement can be deleted.

The immediate MLP1 continuation is also promising but not yet final evidence. With
the compiled MLP0 program upstream, the same-price rank-64 affine family at MLP1
recovers $43.91\%$ of its local teacher-KL denominator and improves copy CE; under
the shuffled upstream pipeline it recovers $-32.96\%$ and harms copying. Both
selectors chose the same family and price, so the difference is not explained by
model-class size. Their denominators differ by context, however, and only the sealed
final common-row contrasts can admit the composed program.

The 64 individual axes do **not** yet have invariant semantic names. For every
orthogonal $Q\in O(64)$,

$$
B_0\mapsto B_0Q,
\qquad
p_0\mapsto p_0Q
$$

describes the same physical write. The invariant object currently identified is the
subspace plus its input-to-code map, not a privileged coordinate system. Calling
coordinate 17 “number” or coordinate 23 “punctuation” before resolving this gauge
would be arbitrary.

This nevertheless gives a concrete route to semantics:

1. Treat the 64-dimensional code as the shared contract between MLP0 and its
   downstream readers.
2. Fix its rotational gauge jointly, seeking a rotation in which the input program
   and downstream read maps have minimum conditional description length or maximum
   structured sparsity.
3. Factor the rotated input map against token, attention-context, and polynomial
   features, and factor downstream readers against the same coordinates.
4. Name only features whose code interventions predict held-out downstream effects
   and whose edits have bounded collateral.
5. Test the resulting program off the FineWeb manifold. An affine predictor can be
   excellent on the observed state manifold while missing the native quadratic
   extension away from it.

This is the principled joint optimization suggested by the user: simplicity is not
low rank at one module in isolation, but short input programs **and** sparse/simple
downstream reads in a common gauge, subject to causal and OOD fidelity.

---

## 7. MLP0 works jointly with MLP1 and MLP2

The exact MLP0/MLP1/MLP2 restoration cube shows that the early MLPs do not behave as
independent modules.

On the same model realization, the individual CE gains were approximately:

$$
\Delta_0=+0.119,
\qquad
\Delta_1=+0.167,
\qquad
\Delta_2=-0.230.
$$

Their singleton sum is only

$$
\Delta_0+\Delta_1+\Delta_2\approx0.056.
$$

But restoring all three jointly gives

$$
\Delta_{012}\approx0.514.
$$

Therefore the interaction surplus is approximately

$$
0.514-0.056=0.458.
$$

Most of the early-layer effect is therefore conditional interaction, not the sum of
three independent effects.

This changes the mechanistic story. MLP0 is not merely producing a final feature that
later layers read independently. It prepares a state that changes what MLP1 and MLP2
should compute.

The program is closer to

$$
\text{MLP0 state}
\longrightarrow
\text{state-conditioned MLP1 computation}
\longrightarrow
\text{state-conditioned MLP2 computation}.
$$

This is why a replacement can have small ordinary final-output error while still
showing a large internal MLP1 mismatch.

---

## 8. What the C512-to-MLP1 interchange resolved

C512's final-output point errors are small, but its internal error is much larger:

$$
\text{attention-1 nRMSE}\approx0.0544,
$$

and

$$
\text{MLP1-output nRMSE}\approx0.2323.
$$

There were two possible explanations.

### Candidate explanation A: downstream-null detail

C512 may discard information that is visible to MLP1 in activation space but has
little behavioral importance.

In that case, C512 is a useful behavioral compression, even though it does not
reconstruct the native interface exactly.

### Candidate explanation B: compensated error

The downstream network may compensate for C512's errors on ordinary inputs. The
discarded information could become necessary under interventions, composition, or
distribution shift.

In that case, C512's small ordinary KL and CE would not make it a manipulable causal
interface.

The completed MLP0-to-MLP1 interchange experiment distinguished these possibilities
more sharply.

For an exact MLP0 path $O$ and C512 path $C$, it constructs:

$$
OO=s_O+m_O,
$$

$$
CC=s_C+m_C,
$$

$$
CO=s_C+m_O,
$$

and

$$
OC=s_O+m_C,
$$

where $s$ is the state before the MLP1 write and $m$ is the physical MLP1 write.

On 384 new FineWeb documents, the pre-MLP1 state difference and the local
state-by-write interaction were small. Almost all of the measured internal mismatch
was carried by the changed physical MLP1 write. With MLP2 omitted, replacing that
write by the exact MLP1 write reduced the standardized family maximum by about

$$
3.34,
$$

with a positive simultaneous lower confidence bound. Thus the missing interface is
strongly localized to the MLP1 write in the MLP2-omitted suffix.

With the deployed MLP2 present, however, C512's ordinary final-output point error was
much smaller, and the registered sensitivity control did not license the same repair
claim. The best current interpretation is therefore:

$$
\boxed{
\text{C512 changes the MLP1 write, and deployed MLP2 suppresses or compensates the
resulting error.}
}
$$

This rules against treating C512's discarded directions as a certified downstream
null space. It also does not license a standalone MLP1 adapter. The next physical
interface question is where and how MLP2 performs the compensation.

## 9. What the MLP2 compensation factorial found

The completed experiment is a physical state-by-write factorial at MLP2. For exact MLP0
path $O$ and C512 path $C$, it separately crosses:

- the state entering MLP2;
- the physical MLP2 write computed on an exact or C512-conditioned state;
- omission, within-cell shuffled-write, and norm-matched native-write controls.

Its eight arms can distinguish four mechanisms:

1. the pre-MLP2 state already carries the remaining error;
2. MLP2's write on the exact state repairs it;
3. MLP2 adapts its write specifically to the C512-induced state;
4. apparent repair is merely write magnitude, generic sensitivity, or a shuffled
   alignment artifact.

The assay uses 384 source documents split into two 192-document waves, 1,256
evaluation windows, 16 frozen cells, source-document bootstrap resampling, and the
same inherited capped-logit currency as the MLP1 interchange. All model, row,
program, arm-routing, call-count, and inference contracts were frozen before its
current run.

The first execution, V1, failed closed after all forward passes but before any
sufficient statistic, bootstrap result, contrast, or decision was serialized. The
sole failure was an outer integrity check that demanded a fixed $10^{-6}$ absolute
norm error even for large writes. The underlying core had already enforced the
scale-aware per-position invariant

$$
|\|w_{\mathrm{control}}\|_2-\|w_{\Delta}\|_2|
\le 10^{-6}+10^{-5}\|w_{\Delta}\|_2.
$$

V2 changed only that redundant outer check, bound the exact V1 authority and failure
receipt, marked the reused rows as spent-but-outcome-blind rather than fresh, and
kept every scientific arm and threshold unchanged. It completed all 1,256 windows,
passed every runtime integrity check, and exactly replayed its frozen inference from
the serialized source-document ledgers.

One common gate was nevertheless false: float32-reported coverage differed from the
exact ledger fraction by only $3.3\times10^{-9}$ to $1.14\times10^{-8}$, but the
scorer required $10^{-12}$ equality. Because this was discovered after outcomes
were serialized, the registered labels remain false/inconclusive rather than being
post-hoc repaired.

Here `false` means “not promoted because a common gate failed,” not scientific
falsification of each mechanism.

The descriptive pattern replicated across both waves:

- omitting MLP2 exposes a large C512 mismatch: about 3.63 practical margins pooled;
- deployed MLP2 reduces it to about 0.79 margins;
- the simultaneous suppression reduction is positive in wave A, wave B, and pooled;
- the local state-by-write interaction point estimate is only about 0.07 margins
  (pooled UCB 0.616), but its dependency-gated component status is inconclusive;
- a shuffled within-wave/cell delta-write control is better than the aligned
  observational arm, so the specific aligned-repair story is not supported;
- observational equivalence is not certified: pooled UCB is 1.333 against a 0.8 bar.

The best current descriptive reading is therefore:

$$
\boxed{
\text{Deployed MLP2 and the ensuing suffix attenuate most C512-induced MLP1
mismatch, but the assay provides no support for a specially aligned repair write.}
}
$$

Because the sensitivity control was unpowered, this is failure to support or certify
alignment, not proof that aligned compensation is absent.

This narrows the joint compiler target, but it does not yet make C512 a certified
causal interface or add whole-model executable recovery credit.

## 10. A principled definition of simplicity

The evidence argues against defining simplicity as merely “the number of token
clusters.”

A better objective is joint causal description length:

$$
\mathcal{C}(P)
=
L(\text{producer})
+
L(\text{latent code})
+
\sum_i L(\text{consumer}_i)
+
\lambda D_{\mathrm{causal}}(P,M).
$$

Here:

- $P$ is the proposed simpler program;
- $M$ is the original model;
- $L$ is serialized description length;
- $D_{\mathrm{causal}}$ is behavioral distortion under natural inputs,
  interventions, composition, and OOD tests.

The optimization should also minimize over irrelevant gauge choices:

$$
\mathcal{C}_{\mathrm{gauge\text{-}invariant}}(P)
=
\min_{G\in\mathcal{G}}\mathcal{C}(G\cdot P).
$$

Under this definition, a number cluster earns credit only if its shared code reduces
the total price of the producer and downstream consumers at the same causal fidelity.

A cluster that requires a large token-specific residual has not compressed the whole
interface. Conversely, a continuous code can be simple if it makes both the producer
and consumers substantially cheaper.

---

## 11. Current best model of MLP0

The current evidence supports this qualitative program:

$$
\boxed{
\begin{aligned}
z &= \operatorname{RMSNorm}(x),\\
h &= (Lz)\odot(Rz),\\
u &= \text{compressed causal coordinates of }h,\\
m_0 &\approx D_{\mathrm{simple}}u,\\
\text{MLP1 write} &\approx g_1(u,\text{current state}),\\
\text{MLP2 write} &\approx g_2(u,\text{MLP1 state and write}).
\end{aligned}
}
$$

The lexical clusters describe shared organization inside $u$. They are not the whole
of $u$.

The main remaining task is to find an executable, causally sufficient coordinate
system $u$ whose producer and downstream consumers are jointly simpler than the
original network.

## 12. Current experiment and when to pivot downstream

We are no longer spending the main experimental budget asking whether MLP0 tokens
form hard clusters. That question has already been answered negatively in its strong
form and positively in its weaker shared-structure form.

Compiler-v2.1 asks the more useful question: can we predict the causally concentrated
MLP0 write coordinates without calling the original MLP, and can MLP1 then be
compiled in the state actually produced by that replacement? It compares five typed
families at each site:

- a $z$-only affine anchor;
- a state-complete affine Euclidean map;
- a state-complete affine causally weighted map;
- a native-product Euclidean program;
- a native-product causally weighted program.

The site0 stage is currently fitting and scoring both true-label and
document-block-shuffled 108-cell banks. Only after the complete ledgers are frozen
does it select an MLP0 program.
The site1 stage then captures and fits MLP1 under that frozen upstream program rather
than under native MLP0. This is therefore already a joint producer/consumer search,
not another isolated MLP0 probe. The first numerical launch produced no candidate
scores: concurrent commits changed the shared repository HEAD after launch, and the
frozen lifecycle bound both the relevant source hashes and the literal global commit.
Because the latter would fail at ledger freeze even when the former were unchanged,
the run was stopped before paying the full GPU search cost. This is a lifecycle
failure, not evidence for or against any MLP0 program family. The retry amendment
keeps launch synchronized to one committed source tree, then binds later transaction
boundaries to the exact scientific source hashes and inherited launch commit. This
allows unrelated descendant commits while still failing closed on any scientific
source, row, protected-artifact, or lock drift. The corrected retry passed independent
mathematical and lifecycle review and launched from `bd9a5820` at 22:24 UTC.

### 12.1 Compiler-v2.1 final outcome

The sealed final completed as an integrity-valid authoritative negative. The rank-64
affine MLP0 program removes $62.67\%$ of its local teacher KL, but conditional MLP1
removes only $43.55\%$, and the pair removes $33.69\%$ of joint teacher KL. The
pair's $0.05914$ CE gain is $26.10\%$ of the exact projected pair's $0.22658$ gain.
The required joint remaining-KL ratio $\le0.50$ and half-oracle gates therefore fail.

This is not a rejection of the continuous code. Both ordered increments have
positive document-bootstrap intervals, the pair beats both singletons, correct labels
beat mean/shuffle controls, and copy and every token-frequency collateral gate pass.
The failure says the present greedy local-loss compiler does not turn the code into a
sufficient executable MLP0-to-MLP1 interface.

The closed family bank narrows the next step. The best final same-family alternative,
causally weighted affine C, improves on selected Euclidean affine B by only $0.00257$
CE and still reaches only $27.23\%$ of the oracle gain; native-product families are
worse. The prospective next discriminator therefore compares local coefficient loss
with suffix KL on identical new rows and optimizer budgets, then separately tests an
explicit gauge-invariant transport $B_0AB_1^\top$ from the executable MLP0 code. Its
mathematical protocol and nine-test pure contract are now independently frozen, but
numerical work remains prohibited until the full lifecycle runner is source-closed and
re-audited; it has loaded no new rows. Only if both routes fail should we spend a new
experiment on rank or a joint suffix-Fisher basis.
See `EARLY_MLP_STATE_COMPLETE_COMPILER_V21_FINAL_FINDINGS.md` and
`EARLY_MLP_SUFFIX_TRANSPORT_V1_PREREGISTRATION.md`.

The decision rule after this run is:

1. If MLP0 and autoregressively fitted MLP1 compose on sealed final documents, treat
   their frozen coefficient interface as the current executable hypothesis and move
   immediately to MLP2 plus whole-model composition.
2. If MLP0 looks adequate locally but the pair fails, stop adding MLP0 clusters or
   regressors. The failure localizes the missing object to the transported
   MLP0-to-MLP1 state/write interface or to a jointly selected latent basis.
3. If the pair works until deployed MLP2 is changed or omitted, fit a conditional
   MLP2 program on the compiled upstream state. The existing factorial says MLP2 and
   the suffix attenuate the mismatch but does not support a specially aligned repair
   write, so the target should be the complete state-conditioned MLP2 map rather than
   a hand-designed correction vector.
4. If early-MLP compilation succeeds but whole-model recovery remains weak, shift to
   attention interfaces. Attention-0 is the only non-token contextual input to MLP0;
   attention-1 is an immediate consumer of the changed residual state. Routing can
   use a compact low-rank grammar, while values require a richer grammar: on untouched
   documents rank-64 routing retained about $63.0\%$, but rank-64 values retained only
   about $2.29\%$.

So it does make sense to interpret MLP1, MLP2, and the adjacent attention paths and
then return to MLP0. The precise version is stronger: use those downstream modules as
the operational definition of which MLP0 coordinates matter. Do not return to MLP0
unless a downstream composition failure identifies a missing interface that MLP0's
current program cannot express.

## 13. Exhaustive token-only downstream-equivalence result (2026-09-01)

Rung394 enumerated all 50,257 real tokens at sequence length one and removed only MLP0's bias-free write while
holding the raw token vector, attention0 value, MLP0 bias, and block1 remix fixed. This is the first exhaustive test
of what MLP0 adds beyond the architecture's explicit raw-token reinjection.

Block1's raw-token coefficient is 8.0 while its incoming-state coefficient is .012695, but MLP0's output scale
reverses the naive implication: after both coefficients are applied, the MLP0 term has 2.257 times the median norm
of the raw term. Their median absolute cosine is only .21495. MLP0 therefore contributes a large distinct nonlinear
token transform, not a small same-direction token copy.

The exact causal response separates by consumer. Attention1 has rank90 156 and participation rank 17.36; MLP1 has
rank90 734 and participation rank 121.83. A joined P256/k16 sparse code reaches only .3308 heldout response R²,
near the dense write-PCA256 ceiling .3535 and only .0107 above an activation-only code. It is seed-stable, but it
does not create the registered response-neighbor equivalence and no high-response/low-write pair crosses the frozen
threshold. All four positives failed and the strong null stayed false.

This result rejects one universal sparse token quotient for attention1 plus MLP1 and forbids immediate sparse-rank
tuning or TT/X promotion. It does not prove the token-only write is uninterpretable. The next distinct question is
whether that large write is a transformed representation from which exact token identity is linearly or
orthogonally recoverable. If so, the token-private component can be explained as a coordinate conversion and the
residual becomes the honest target for shared lexical features. If not, the token-specific quadratic map itself is
the object that must be described.

Rung395 answered this more sharply. A simple global linear/orthogonal copy fails its registered reconstruction bars
(write→raw R² .227, raw→write .397, Procrustes cosine .387), but nearest-vocabulary decoding recovers the exact
heldout token 90.4% top1 and 96.2% top5, versus zero under shuffled pairing. Physical injection of the fitted
raw-token component reached joint response R² .972, but rung396 exposed that this included a dominant common mean:
a mean-preserving shuffled token map already reaches .925. The stronger “token identity is 97%-sufficient” wording
is therefore withdrawn.

The corrected degree-one result remains important. Rank64 retrieves 97.58% of exact heldout tokens and reaches
joint response R² .964, removing about52% of error left after the mean; for attention1 it improves .720→.886,
removing about59% of remaining error. Exact-z and raw-x0 projection curves are nearly identical and saturate near
.397 write R². The current token-only decomposition is a large common write, a compact token-identity modulation,
and a broad orthogonal quadratic residual. The registered strong null fired, so no live TT transfer or rank tuning is
licensed until the common mean is explicitly conditioned out in every comparison.

Thus the supported token-only account is not “a rotated embedding” and not “a compact universal sparse quotient.”
It is a common operating-point write plus a nonlinearly warped token-identity modulation and a broad quadratic
residual. The modulation is incrementally causal after conditioning on the common write, especially for attention1,
but it is not sufficient on its own. The next legal object is a prospectively registered causal factorial over the
exact constant, complete degree-one, and orthogonal quadratic components; every score must be reported relative to
the common-write arm rather than as an unconditioned response R².

Rung397 performed that conditioning with an exact eight-arm causal factorial. The common write alone reaches joined
response R² .92492. Adding the correctly paired complete degree-one modulation recovers 62.19% of the joined error
left by the mean, versus -0.17% for a fixed shuffled-token control; attention1 recovery is 73.51%. Adding the correctly
paired exact quadratic residual after the mean recovers 47.95% joined error, versus 4.26% shuffled; attention1 recovery
is 63.52%. A wrong token's complete write reaches only .91539 joined R², below the registered .95 null. All exactness,
consumer-split, and aligned-component predicates held; null false.

The supported length-one account is therefore stronger than the provisional split: the common vector is an operating
point, the degree-one component is a compact token label/control modulation, and the broad quadratic residual is an
additional correctly paired token correction rather than disposable noise. Their block1 consequences are strongly
nonlinear—large pair and triple Möbius terms cancel—so component roles must be stated conditionally after the common
write. This is exact causal attribution, not compression: Q is an exact heldout residual. The next legal token-only
test is downstream-effect equivalence by cross-consumer physical interchange, not another activation sparse code.

Rung398 tested that interchange prospectively. Donors came only from the fitting set, receivers only from the heldout
set, and far-effect donors were forced to have component cosine <=.50. For L, attention-selected donors preserve
physical attention1/MLP1 conditional-effect cosines .837/.651 and MLP-selected donors .683/.719, far above random
.193/.166. But simple L-action neighbors score .880/.835 and raw neighbors .656/.691; the frozen cross-consumer and
control-margin bars fail. One L route nevertheless preserves both consumers above .80 for 5.30% of receivers across
7,000 donors. For Q, cross-consumer transfer is only .338 or .492 in the selected consumer pairing and global R² is
negative; only .43%–1.44% of swaps preserve both consumers above .80. Pred_a/d held, b/c failed, null false.

Therefore there are local far-direction semantic niches—decoded examples include `primarily`/`largely`,
`Although`/`While`, and several year-token pairs—but no supported whole-vocabulary equivalence code. L has weak
many-to-one structure already tracked by action geometry; Q is predominantly token-private. Retain exact identity.
The next object is a consumer-aware quadratic spectrum, which can seek a small set of Q directions read by each
consumer without asserting token interchangeability.

Rung399 closed that spectrum test. Separate consumer bases do cross over at rank64: attention-aware beats MLP-aware
on attention R² by .106, and MLP-aware beats attention-aware on MLP R² by .069. Q→attention has one dominant
cross-Gram strength 273 followed by33,30,23,20, whereas Q→MLP is broad at17.5,14.3,12.8,10.4,7.7. But every
response-aware joint basis loses to ordinary Q-PCA. At ranks16/64/256/512, response-aware joined R² is
.189/.375/.665/.827 versus PCA .207/.398/.698/.865. Rank512 costs26,322,560 values, 45% of the57,896,064-value
exact Q table, yet remains partial. Pred_a/d held, b/c failed, and the registered no-PCA-advantage strong null fired.

Thus token-only M/L/Q causal roles are identified, local interchange niches are bounded, and current sparse,
degree-one, grouping, and reader-weighted compression routes are closed. Attention1 reads a narrow Q signal; MLP1
reads broad token-private Q. Retain exact Q rather than tune ranks. The next program must reuse the existing exact
TT/X/CC real-context factorial and connect this length-one token anatomy to X and CC under new crossed controls.

## 14. Exact natural-context causal grammar (2026-09-01)

Rungs400–401 performed that connection without repeating the old TT/X/CC assay. Write the pre-normalization MLP0
state as token ray `e` plus attention0 context write `a`, and define the bilinear MLP map

`T(u,v) = Down((Left u) * (Right v))`.

For `G(e,a)=T(e+a,e+a)`, independently crossing the fitting token and context distributions gives the exact
functional ANOVA `G=mu+FT(e)+FC(a)+FTC(e,a)`. After restoring the observed normalization, the four intervened roles
are fixed-gain token main `T`, fixed-gain context main `C`, centered token×context interaction `I`, and observed
normalization-gain modulation `S`. The actual normalized vector has a small non-scalar residual
`r=z-s(e+a)`; retaining

`R=T(s(e+a),r)+T(r,s(e+a))+T(r,r)`

in every arm makes the identity exact rather than treating BF16 error as semantics. The repaired analytical relative
MSE is `2.88e-13` on FIT and `2.89e-13` on SELECT. It reproduces every rung400 arm CE and every Shapley value exactly.

The held-out SELECT Shapley CE benefits are `I=1.53753`, `T=1.49833`, `C=.41773`, and `S=.06749` nats; FIT gives the
same signs and order. Combined context-dependent contribution `C+I+S=2.02275` exceeds token main `T=1.49833`.
The preregistered standalone-context bar `C>=.50` honestly failed. The supported conclusion is therefore not “a
large additive context write,” but that context acts chiefly through the bilinear token×context interaction. This is
exact causal attribution, not compression: all 15,926,400 native MLP0 values remain in use, no FINAL rows were opened,
and no replacement is licensed.

The length-one `M/L/Q` and natural-context `T/C/I/S` decompositions answer different levels of the same mechanism.
At length one, attention0 is deterministic in the current token, so `M/L/Q` describes the complete token-fed
function. On natural text, `T` is the token effect averaged over independently crossed contexts, while `I` is the
additional computation that a particular attention0 context induces for a particular token. Rung401 prospectively
selects `I` as the next branch because it is largest on both document splits. The next legal interpretation step is
to audit prior attention0 work, then resolve `I` into attention-head/source-position contributions and validate the
largest terms physically; a complementary adoption link is to measure which branch the already adopted rank448
MLP0 context projection preserves or damages.

## 15. Attention-head carriers of the centered interaction (2026-09-01)

Rung402 uses the exact additivity of attention0's output projection. If `y_h` is head h's 128-coordinate pre-output
state and `O_h` the corresponding output-projection column block, then `a_h=O_h y_h` and the context write is the sum
of nine `a_h`, plus a separately retained BF16 arithmetic remainder. Since centered `I` is linear in context
deviation, it decomposes exactly as `I=sum_h I_h+I_eps` without fitting a probe or choosing a basis.

The repaired 21-arm assay keeps T/C/S and every numerical residual fixed and evaluates FULL, ZERO_I, NUMERIC,
SINGLE_h, and DROP_h arms. Its first run is preserved as an instrument failure: summing ten float terms before BF16
conversion made ZERO_I differ from the parent no-I boundary by `6.43e-5/1.04e-4` CE. The same-rung repair changed
only ZERO_I to subtract the identical parent I tensor. The repaired FULL and ZERO_I parent differences are exactly
0.0; I reconstruction relMSE is `2.24e-18`; BF16 head-write remainder relative squared energy is `2.75e-6`; the
numerical-only interaction effect is `-.00030` nat on SELECT; and all live calls/states hold.

Head3 is dominant and individually material: SELECT singleton/removal/endpoint-average benefits are
`.09957/.04128/.07042` nat. Heads8 and7 follow at endpoint averages `.01696/.01660`; heads6,0,4 are smaller but
nontrivial. The new ranking is stable across FIT/SELECT (Spearman `.9333`) and matches the old whole-head direct-cost
ranking (Spearman `.9000`). Pred_a/b/d hold and the strong null is false.

Pred_c honestly fails: positive top-two endpoint-average share is `.6224/.6046` on FIT/SELECT, below the frozen `.65`
bar and far below the old whole-head map's `.883`. The exact mechanism is therefore not licensed as a two-head sparse
router. Whole-head deletion overstated concentration because it changes every use of the attention write; isolated I
has one dominant local head plus a stable distributed supporting tail. Per the frozen rule, do not expand only head3
by source position. Next perform a branch-resolved physical audit of the rank448 MLP0 context program: determine
whether its error lies in T, C, I, S, or their downstream composition before designing another compressor.

## 16. Rank448 error anatomy and exact quadratic-producer closure (2026-09-01)

Rungs403–404 audited the fixed context-covariance rank448 replacement under the exact T/C/I/S grammar. On 384
distinct source documents its heldout CE damage is `.0071068` nat. The stable statement is token-grammar-led:
positive named damage is 38.0% T and 54.4% I, or 92.4% jointly; I alone leads T in only two of four waves, so the
earlier interaction-only headline is withdrawn. C is secondary and S compensates. Five input-metric families and
five routing levels subsequently failed to improve the priced program.

Rung409 changed object to output error. One train-only rank64 total-error basis reduces heldout p448 damage from
`.007947` to `.004769` as an oracle, but it reads the native error. Separate T32/I32 bases are worse than one joint
T+I basis (`.005876` versus `.004838`), so T and I retain distinct causal definitions but share a better output
interface. The old historical B0 basis removes only17% of damage and is not reusable.

Rung410 then contracted that output basis into the known native-minus-p448 bilinear weights. Every coefficient is
exactly `q_j(z)=z^T A_jz+beta_j`; the full derivation matches directly computed coefficients at relative MSE
`4.05e-12`. An executable independent rank24 approximation improves damage to`.006929` and beats affine/shuffled
controls, but recovers only32.0% of oracle gain and retains35.7% of weighted form energy. Its total price11,799,232
is only145,856 below p640, whose damage is `.002868`.

Rung411 tested a genuinely different joint Tucker factorization sharing input directions across all64 matrices.
At total prices10,436,800/11,036,864/11,930,240, Tucker96/160/226 damages are
`.007238/.007153/.006989`. Equal-or-cheaper ordinary covariance ranks494/552/638 achieve
`.006051/.004365/.002820`. No Tucker arm wins its price comparison; the strong null fires. Thus the residual is
**output-low-rank but input-broad**. Its exact quadratic tensor is a useful interpretation, but fixed low-rank U64
producers are closed as replacements unless a new grammar changes the computation rather than refactoring these
same forms.

## 17. Cross-head source relations are stable but not sparse circuits (2026-09-03)

Rung517 decomposed attention0 by query-to-source relation rather than by head. For every query, all source positions
across all nine heads were assigned exactly once to SELF, PREVIOUS, NEAR (lags2--7), DISTANT_SAME, or DISTANT_OTHER.
The32 possible subsets were passed through native MLP0 while attention0's direct residual write remained unchanged.
This is an exact causal test of what context MLP0 uses, not an attention-head ablation.

On held-out prose, the full-versus-empty context benefit is1.9004 nats. Shapley allocation assigns12.9% to SELF,
30.0% to PREVIOUS,26.4% to NEAR,3.6% to DISTANT_SAME, and27.0% to DISTANT_OTHER. Structured text is nearly identical
at12.8/30.8/26.2/3.4/26.8% of1.7467 nats. The ordering and absolute-position profiles are stable across document
splits, but the registered sparse hypotheses fail: SELF+PREVIOUS is only42.0% of the prose endpoint total, and
structured text does not shift ten percentage points away from that local pair.

The main mechanistic lesson is redundancy. PREVIOUS, NEAR, and DISTANT_OTHER alone recover78.5%,79.1%, and81.9% of
the prose full-context benefit when added to the empty boundary. Removing the same groups from full context loses
only5.64%,.80%, and.67%. Several source sets can therefore support much of the same MLP0 benefit in the presence or
absence of the others. Their effects must not be described as independent additive circuits.

The token-by-context vector identity splits cleanly across source groups. The context-only quadratic identity needs a
large47--52% closing term because its parent definition subtracts the fit-set average quadratic context response; this
is a centering term plus the measured arithmetic remainder, not evidence that half the computation is floating-point
noise. Immediate attention1/MLP1 response profiles are repeatable, but PREVIOUS does not beat matched random source
sets by the registered relation-specific margin. Thus source relation is useful stable anatomy, but not yet a
selectively manipulable or executable semantic decomposition. The next object should condition grouping and splitting
on the existing circuit tasks rather than refine the distance bins or optimize rank.

## Related authoritative write-ups

- `MLP0_QUOTIENT_STAGE0_V2_FINDINGS.md`
- `MLP0_NATIVE_DOWN_HIERARCHY_V1_FINDINGS.md`
- `MLP0_C512_MLP1_INTERCHANGE_SPEC.md`
- `MLP0_C512_MLP1_INTERCHANGE_V3_FINDINGS.md`
- `MLP0_C512_MLP2_COMPENSATION_SPEC.md`
- `MLP0_C512_MLP2_COMPENSATION_V2_FINDINGS.md`
- `explanation_2026-09-01_1358.md`
- `explanation_2026-09-01_1406.md`
- `explanation_2026-09-01_1415.md`
- `explanation_2026-09-01_1421.md`
- `explanation_2026-09-01_1432.md`
- `explanation_2026-09-01_1441.md`
- `explanation_2026-09-01_1502.md`
- `explanation_2026-09-01_1518.md`
