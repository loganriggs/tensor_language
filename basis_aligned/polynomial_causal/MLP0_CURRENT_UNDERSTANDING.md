# What We Currently Understand About MLP0

*Last updated: 2026-08-27 19:40 UTC. The authority-bound MLP2 compensation
factorial is queued but has not yet produced a scientific result.*

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

## 9. Current live discriminator: where MLP2 compensates

The next experiment is a physical state-by-write factorial at MLP2. For exact MLP0
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

V2 changes only that redundant outer check, binds the exact V1 authority and failure
receipt, marks the reused rows as spent-but-outcome-blind rather than fresh, and
keeps every scientific arm and threshold unchanged. Its source and authority are
committed, independently audited, and queued. Until it finishes, the MLP2 mechanism
above remains an unresolved hypothesis rather than a result.

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

## Related authoritative write-ups

- `MLP0_QUOTIENT_STAGE0_V2_FINDINGS.md`
- `MLP0_NATIVE_DOWN_HIERARCHY_V1_FINDINGS.md`
- `MLP0_C512_MLP1_INTERCHANGE_SPEC.md`
- `MLP0_C512_MLP1_INTERCHANGE_V3_FINDINGS.md`
- `MLP0_C512_MLP2_COMPENSATION_SPEC.md`
