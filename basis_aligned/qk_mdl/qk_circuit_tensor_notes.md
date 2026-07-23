# The head as a third-moment tensor: which object is the right decomposition target?

*Response to Logan's diagrammatic notes (spiders, moment tensors, the circuit tensor
$\mathcal{C}$). Mapping onto bilin18 layer 0, then: which of these objects should we
compress, and is there a "sparse bilinear layer" with lower MDL?*

## 1. The mapping (bilin18 has two branches, so it's a *mixed* third moment)

bilin18's pattern is a product of two distinct branch scores, not a squared single score, so
the source-side spider carries three *different* dressings. Per head, with folded factors
$k^{(1)}_t, k^{(2)}_t \in \mathbb{R}^{128}$ and value $v_t = W_v \hat e_t \in \mathbb{R}^{128}$:

$$\text{Out}_i \;=\; \sum_{j \in \text{ctx}} \big(q^{(1)}_i \cdot k^{(1)}_{t_j}\big)\big(q^{(2)}_i \cdot k^{(2)}_{t_j}\big)\, W_o v_{t_j}
\;=\; \mathcal{C}_{\text{ctx}}\big(q^{(1)}_i,\, q^{(2)}_i,\, \cdot\big),$$

$$\mathcal{C}_{\text{ctx}} \;=\; \sum_{j \in \text{ctx}} k^{(1)}_{t_j} \otimes k^{(2)}_{t_j} \otimes W_o v_{t_j}
\;\in\; \mathbb{R}^{128 \times 128 \times 1152}.$$

Everything from the notes transfers: the $j$-spider is capped (summed) so the source side
piles into a genuine 3-tensor, **linear in the context** (mixtures of corpora → mixtures of
$\mathcal{C}$); the $i$-side spider is open, so the destination side stays a Khatri–Rao
object (per-query, no pile-up). Because the branches are distinct there's no mode-1/2
symmetry, and the destination degree is (1,1) rather than $\mathrm{Sym}^2$ — otherwise
identical. Multilinear ranks: $\le 128$ in every mode, so the whole head Tucker-compresses
to a $128^3$ core (contract mode 3 through $W_o$'s column space).

## 2. Which object can be *the* decomposition target?

Three candidate objects, and they are not interchangeable:

**(a) The token-indexed CP factors** $\{(k^{(1)}_t, k^{(2)}_t, v_t)\}_{t \in V}$ *(plus the
query-side pairs)*. This is the only object that determines the full function
context ↦ $\mathcal{C}_{\text{ctx}}$ — the map is linear in the per-token rank-1 terms, so
you need all $V$ of them. **This is what we have been compressing all along**: the factor
tables *are* the CP factor matrices, and "the CP decomposition is the definition" is exactly
why the dictionary program targets them. The framing adds something real, though: our
current *grouping* (concatenate q-half ‖ k-half within a branch; two independent
dictionaries per head; OV untouched) is one arbitrary slicing of the CP factors. See §4.

**(b) The expected moment tensor** $M = \sum_t \pi_t\, k^{(1)}_t \otimes k^{(2)}_t \otimes W_o v_t$
(unigram weights $\pi$). This is *not* a substitute target — it's the **static component**
of the head, precisely: over i.i.d. length-$T$ contexts,
$\mathbb{E}\,[\mathcal{C}_{\text{ctx}}] = T \cdot M$, so $M$ carries exactly the $T^2$ term
of our delivery metric — the piece we measured at ~99% of delivered energy. Knowing $M$
exactly tells you the *average* context's behavior and nothing about fluctuations; it can't
reconstruct per-token structure. But it is tiny ($128^3$ per head in head space) and exact.

**(c) The Tucker core** ($128^3$): the right *analysis* object (all three weight matrices
plus embedding statistics in one small exact tensor), and per the notes, the wrong
*evaluation* object — the CP form term-by-term is literally attention.

## 3. Payoff 1: the static term of our training objective is computable EXACTLY

This is the immediately actionable consequence, and it reframes the tick-167 result.
The systematic term of eq. † is a moment-tensor contraction:

$$\mu_i \;=\; \sum_t \pi_t\, \Delta P(i,t)\, u_t \;=\; \hat M\big(\hat q^{(1)}_i, \hat q^{(2)}_i, \cdot\big) \;-\; M\big(q^{(1)}_i, q^{(2)}_i, \cdot\big),$$

and under rotary the rotation acts on the *query* side ($S_\Delta = (R_\Delta q_i)\cdot k_t$), so
**the same two moment tensors serve every offset**: $\mu_i^\Delta = \hat M(R_\Delta \hat q^{(1)}_i,
R_\Delta \hat q^{(2)}_i, \cdot) - M(\ldots)$. Working in head space ($128^3$ cores, $W_o^\top W_o$
Gram precomputed), building $\hat M$ costs ~100 GFLOP per step and evaluating
$\sum_i \pi_i \|\mu_i^\Delta\|^2$ over **all 50304 queries and all offsets** costs a few hundred
GFLOP — comparable to the current M=1024 *sampled* step. So we can replace the sampled
estimator of the dominant term (whose noise tick-167 just showed was binding: M=1024→4096
moved +0.0048→+0.0034) with the **exact full-vocabulary value**, keeping sampling only for
the scatter term (1/57 of the energy, and 6th-order — not moment-reducible). Prediction:
captures the full coverage benefit at M=1024 cost, possibly beating M=4096 since the
dominant gradient becomes noise-free. **Queued as tick 169a.**

## 4. Payoff 2: the "sparse bilinear layer" — yes, and it's a regrouping of the CP factors

The question "is there a sparse structure, like a sparse bilinear layer, with lower MDL?"
has a concrete answer in this frame: **sparse CP coding with the modes tied per token.**
Give each token ONE sparse code $z_t$ over atoms that are rank-1 *triples*
$(a_m, b_m, c_m) \in \mathbb{R}^{128} \times \mathbb{R}^{128} \times \mathbb{R}^{128}$:

$$k^{(1)}_t \otimes k^{(2)}_t \otimes v_t \;\approx\; \sum_{m \in \text{code}(t)} z_{tm}\; a_m \otimes b_m \otimes c_m.$$

Then the head *is* a sparse bilinear layer with $n$ interpretable "attention features":

$$\text{Out}_i \;=\; \sum_m A_m \,\big(q^{(1)}_i \cdot a_m\big)\big(q^{(2)}_i \cdot b_m\big)\, W_o c_m,
\qquad A_m = \sum_{j \in \text{ctx}} z_{t_j m}$$

— atom $m$ = a context feature (which tokens carry it, via the sparse code), a destination
selectivity (the bilinear form $a_m \otimes b_m$), and an output direction ($W_o c_m$); its
activation is a context-summed sparse code. Why this could be *lower* MDL than what we do
now: (i) **one code per token instead of two-per-branch** — index bits roughly halve at
matched $k$; (ii) it exploits **cross-branch (and cross-into-OV) correlation** — our atoms
came out semantic/topical, and a topical token is topical in *both* branches and in what it
delivers, so the current per-branch dictionaries pay for that structure twice or three
times. The pure-QK version (regroup as query-pairs $(q^{(1)}_t \| q^{(2)}_t)$ and key-pairs
$(k^{(1)}_t \| k^{(2)}_t)$, same raw object and bits conventions as everything so far) is a
clean matched-bits A/B against the current within-branch grouping; the triple version folds
$v_t$ in and is the composed-head object (baseline changes, kept separate). Rank caveat from
the diagrams: a rank-1-triple atom is a rank-1 constraint in the $128^2$ Khatri–Rao space —
richer than our current concat-atoms per branch, but the same token might genuinely need
different sparse structure on its two branches (the tick-164 residuals split ~evenly across
branches), so this is an empirical question, not a theorem. **Queued as tick 169b.**

The anchors survive unchanged in either scheme — exposure logic is grouping-independent —
and the caveat from the notes applies verbatim: the token-axis spider is basis-privileged
(fine, tokens are a real basis), and nothing here claims a privileged basis in the hidden
dimension.

## 5. What the moment view adds to interpretation (no new bits)

$\mathcal{C}$'s linearity in the context distribution means each head's static behavior
decomposes over corpus components: $M$ for a mixture is the mixture of $M$'s. With the
$128^3$ cores cheap to build per corpus slice, "which data is this head's average behavior
made of" becomes a small exact linear-algebra question (project a slice's core onto the
full-corpus core), which composes with the per-head attribution we already have. Worth a
tick when the frontier work settles.
