# The two ledgers: what the function frontier and the mechanism core each measure

*(tick 179; companion code with every number below: `qk_toy_ledgers.py`, run output in
`qk_toy_ledgers.out`. All toys are seconds on CPU and fully deterministic.)*

We have been maintaining two separate "ledgers" for the layer-0 QK circuit. They look
superficially similar — both compress token tables into sparse structure — but they answer
different questions, and tick 178 showed empirically that they rank things differently
(the heads that *fail* the mechanism gate cause the *least* function damage). This note
pins down what each object is, gives a minimal worked example of each, and finishes with
a two-token example where the two ledgers provably disagree about what to keep.

---

## 1. The two objects, precisely

### Ledger 1 — the function ledger (minimum description length of behavior)

The object is a **pair of numbers per compressed model**:

$$
\Big(\; L(\hat\theta)\ \text{bits}\,,\quad \Delta \mathrm{CE} = \mathrm{CE}(\hat\theta) - \mathrm{CE}(\theta)\ \text{nats} \;\Big)
$$

where $L(\hat\theta)$ is the number of bits needed to write down the compressed weights
(dictionary atoms + sparse codes + exact anchor rows), and $\Delta\mathrm{CE}$ is the
held-out damage to next-token prediction when those compressed weights replace the true
ones inside the full model (our standard audit: 307{,}000 predictions on FineWeb). The
*frontier* is the Pareto set over all our compression schemes: for each bit budget, the
smallest prediction damage anyone achieved.

What it measures: **how algorithmically simple the circuit's input–output behavior is.**
If 7{,}418 megabits of raw factor tables can be replaced by 493 megabits at a cost of
$+0.0023$ nats, then $\sim 93\%$ of the stored bits were not load-bearing for prediction.
The metric is *exposure-weighted by construction*: a token's misfit only costs what its
frequency (times downstream sensitivity) says it costs.

### Ledger 2 — the mechanism ledger (sparse symmetric third moment)

The object is, per head, the **third-moment tensor of the head's token rows** and its
sparse decomposition. With $y_t = [\,k^{(1)}_t \,\|\, k^{(2)}_t \,\|\, v_t\,] \in
\mathbb{R}^{384}$ (both key branches plus the value vector) and unigram weights $p_t$:

$$
\mathcal{M} \;=\; \sum_{t \in \text{vocab}} p_t \; y_t \otimes y_t \otimes y_t
\;\in\; \mathbb{R}^{384\times 384\times 384}
$$

(compressed in practice through a sparse code $y_t \approx \sum_a s_{ta} d_a$, giving a
small core $M_{abc} = \sum_t p_t s_{ta} s_{tb} s_{tc}$), followed by a symmetric
nonnegative CP decomposition

$$
M \;\approx\; \sum_{r=1}^{R} \lambda_r \; u_r \otimes u_r \otimes u_r .
$$

Why the *third* moment, and why this is the natural object for **bilinear** attention:
the model's attention pattern is $P(i,t) \propto (q^{(1)}_i \cdot k^{(1)}_t)(q^{(2)}_i
\cdot k^{(2)}_t)$ with no softmax, so what a head *writes* for query $i$ is, in
expectation over contexts,

$$
\mu_i \;=\; \sum_t \pi_t \,\big(q^{(1)}_i \cdot k^{(1)}_t\big)\big(q^{(2)}_i \cdot k^{(2)}_t\big)\, v_t
\;=\; \mathcal{M}\big(q^{(1)}_i,\, q^{(2)}_i,\, \cdot\big),
$$

a **contraction of the third-moment tensor against the two query vectors**. Everything
the head can do to the residual stream, averaged over data, factors through $\mathcal{M}$.
Each CP component $u_r$ is then an **archetype**: a direction in code space that
simultaneously names (a) a class of key-tokens, (b) the same class again on the other
branch, and (c) what gets written when both match. On the real circuit these came out as
case-invariant scaffold classes — a $\{$the$\}$ archetype, $\{$a/an$\}$, $\{$of$\}$,
$\{$and$\}$, punctuation families.

What it measures: **whether the head's interaction structure decomposes into a small
number of nameable types**, validated by planted-recovery gates, permutation nulls,
restart stability, and cross-corpus reproducibility — none of which are denominated in
nats.

---

## 2. Toy A — the function ledger in 8 tokens

Vocabulary of $V=8$ tokens with key vectors in $\mathbb{R}^4$, planted as **two clusters**
(four tokens near $c_1$, four near $c_2$, noise $0.05$). The toy language model's logits
for query-token $i$ are just its score row $S_i = Q\,K^\top$, and the "data distribution"
is the softmax of the true scores.

Raw storage: $8 \times 4 \times 32 = 1{,}024$ bits. Dictionary: 2 atoms (the cluster
means) at $2\times4\times32$ bits plus a 1-bit cluster index per token $= 264$ bits — a
**3.9× compression**. Measured damage:

$$
\mathrm{CE}_{\text{true}} = 1.8867, \qquad \mathrm{CE}_{\text{dict}} = 1.8897,
\qquad \Delta\mathrm{CE} = +0.0030 \ \text{nats}.
$$

That is the entire function ledger in miniature: one point $(264\ \text{bits},\
+0.0030\ \text{nats})$ dominating the raw point $(1{,}024\ \text{bits},\ 0)$. Note what
it does **not** tell you: it never names the clusters, never says *why* two atoms
suffice, and would be equally happy with any 264-bit code achieving the same damage.
Compression certifies simplicity; it does not exhibit structure.

*(This is also exactly the shape of the real result: the real frontier point
$+0.0023$ nats at 493 megabits is this toy scaled up 500{,}000-fold, with rotary-aware
training objectives and exact anchor rows doing the work that "use the cluster mean"
does here.)*

## 3. Toy B — the mechanism ledger in 12 tokens

Twelve tokens in $\mathbb{R}^6$, planted as two **classes**: six tokens share direction
$u_1 = e_1$, six share $u_2 = e_2$, each with noise $0.08$; uniform $p_t = 1/12$. Build
the third moment $\mathcal{M} = \sum_t p_t\, y_t^{\otimes 3}$ (here just a
$6\times6\times6$ tensor) and fit rank-2 symmetric nonnegative CP with the same
power-iteration-plus-deflation fitter used on the real cores. Result:

$$
\lambda = (0.528,\ 0.469), \qquad
|\cos(u_{\text{recovered}}, u_{\text{planted}})| = (0.9999,\ 0.9992), \qquad
\text{relative residual } 0.099.
$$

The two archetypes are recovered essentially exactly, and $\lambda_r \propto \sum_{t\in
\text{class}} p_t \|y_t\|^3$ tells you each class's interaction weight. The control that
makes this a *finding* rather than numerology: permute the coordinates of each row
independently (destroying the shared-direction structure while preserving every row's
norm and sparsity) and the rank-2 residual jumps from $0.099$ to $0.586$ — a 6× gap.
That permutation null is the toy version of spec check 3 on the real heads (where real
archetypes beat their nulls by 2–10×).

Note what *this* ledger does not tell you: nothing here says how many **bits** the
description costs or how much **prediction damage** you'd take by replacing $y_t$ with
its class direction. It certifies *structure*, not *sufficiency for behavior*.

## 4. Toy C — two tokens where the ledgers disagree

This is the pedagogical core, and it reproduces the tick-178 surprise (moment-gate
failures ranked *least* damaging functionally). Two tokens:

| token | frequency $p$ | row $y$ | $\|y\|$ |
|---|---|---|---|
| "the" | 0.99 | $e_1$ | 1 |
| "Kowalski" (rare name) | 0.01 | $8\,e_2$ | 8 |

You may keep **one atom**. The function metric weights squared error by exposure,
$\sum_t p_t \|\hat y_t - y_t\|^2$, so the masses are $p\|y\|^2$: the $= 0.99$ versus
Kowalski $= 0.64$ — **"the" matters more**. The moment metric compares
$\|\hat{\mathcal{M}} - \mathcal{M}\|$, and each token enters the third moment with mass
$p\|y\|^3$: the $= 0.99$ versus Kowalski $= 5.12$ — **Kowalski matters more**, because
the extra power of $\|y\|$ lets a rare-but-large row dominate. Measured:

| kept atom | function damage | moment residual |
|---|---|---|
| fit "the" | **0.64** ✓ | 0.98 ✗ |
| fit "Kowalski" | 0.99 ✗ | **0.19** ✓ |

Each ledger picks the opposite atom. Neither is wrong — they are weighting the same
approximation error by different powers of row norm ($p\|y\|^2$ versus $p\|y\|^3$), so
whenever a head's geometry has rare-but-large rows, the two orderings must diverge.
This is very plausibly the story of heads 0 and 4 on the real circuit: their rows
resist third-moment summarization at 512 atoms (residual halves per capacity doubling —
a fat tail of distinctive large rows), yet patching their compressed keys costs only
$\sim 0.0003$ nats each, a third of what the gate-passing heads cost — because the
function metric simply never asks about rows that rarely fire.

---

## 5. So what is each one *for*?

**Reach for the function ledger when the claim is about sufficiency or cost:**

- *Upper-bounding circuit complexity.* "Layer-0 QK is at most 493 megabits of
  algorithm" is a theorem-shaped statement; parameter count is not.
- *Honest comparison of decomposition methods.* Any method — SAE, anchors, low-rank,
  the mechanism pipeline itself (tick 178) — can be scored on the same
  (bits, nats) axes. It is the program's binding metric precisely because it cannot be
  gamed by a decomposition that is pretty but not load-bearing.
- *Distillation / editing budgets.* It tells you directly what you may delete and what
  you must keep (the exact-anchor result — 256 tokens carrying half the error — is a
  function-ledger discovery).

**Reach for the mechanism ledger when the claim is about structure or explanation:**

- *Naming what a head does.* "Head 3 has a $\{$the$\}$-archetype whose match writes
  direction $w$" is a mechanism-ledger statement; no point on the function frontier
  says anything of the kind.
- *Comparing heads to each other and across corpora.* Core cosine 0.98–0.99 across
  disjoint data slices is evidence the *mechanism* is corpus-general — a claim about
  $\mathcal{M}$, not about bits.
- *Generating hypotheses the function ledger then tests.* The archetype token-classes
  predicted the anchor sets at 28–42× chance overlap; the bridge (tick 178) then priced
  the mechanism reconstruction in nats. Traffic between ledgers is where the leverage
  is.
- *Catching solver failure.* Its gates (planted recovery, permutation nulls) are
  known-answer tests; the function ledger has no equivalent internal check — a buggy
  compressor just looks like a bad frontier point.

**And the one-line separation, from Toy C:** both ledgers score the same reconstruction
errors, but function weights them by $p_t\|y_t\|^2$ *through the model's predictions*,
while mechanism weights them by $p_t\|y_t\|^3$ *through the head's interaction tensor*.
Same circuit, different exponent, provably different priorities — which is why we keep
two ledgers instead of pretending one number could serve both purposes.
