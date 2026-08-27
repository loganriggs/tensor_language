# What We Currently Understand About MLP0

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

## 5. The strongest executable compression result so far

The best tested executable approximation to MLP0's `Down` map is called **C512**.

C512 retains:

- the exact RMSNorm input;
- the exact `Left` map;
- the exact `Right` map;
- all native coordinatewise product features.

It replaces only the original `Down` map with a rank-512 continuous program.

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
ordinary final-output behavior.

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

## 8. The current unresolved MLP0 question

C512's final-output point errors are small, but its internal error is much larger:

$$
\text{attention-1 nRMSE}\approx0.0544,
$$

and

$$
\text{MLP1-output nRMSE}\approx0.2323.
$$

There are two possible explanations.

### Explanation A: downstream-null detail

C512 may discard information that is visible to MLP1 in activation space but has
little behavioral importance.

In that case, C512 is a useful behavioral compression, even though it does not
reconstruct the native interface exactly.

### Explanation B: compensated error

The downstream network may compensate for C512's errors on ordinary inputs. The
discarded information could become necessary under interventions, composition, or
distribution shift.

In that case, C512's small ordinary KL and CE would not make it a manipulable causal
interface.

The current MLP0-to-MLP1 interchange experiment is designed to distinguish these
possibilities.

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

The decisive comparison asks whether placing the exact MLP1 write into the C512
upstream state repairs the behavioral error. If it does, we have evidence for a
small conditional MLP1 adapter. If it does not, C512's missing information is not
localized to that interface in a simple way.

---

## 9. A principled definition of simplicity

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

## 10. Current best model of MLP0

The current evidence supports this qualitative program:

$$
\boxed{
\begin{aligned}
z &= \operatorname{RMSNorm}(x),\\
h &= (Lz)\odot(Rz),\\
u &= \text{compressed causal coordinates of }h,\\
m_0 &\approx D_{\mathrm{simple}}u,\\
\text{MLP1 write} &\approx g_1(u,\text{current state}).
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
