# Methods: how we interpret this model (current, 2026-07-29)

> **Status after red-team + composed-fold arcs (2026-07-29, see `redteam_findings_2026-07-29.md`,
> `qk_ledger_v4.json`):** the ledger metric is a **substitutable fraction**, not "understood" —
> only claims passing the code-verify meaning gate count as understood (currently: the induction
> predicate, which subsequently **passed a fully held-out refit**: 98.4–110.5% retention across
> fresh-corpus/period cells). Headline = the **measured joint substitutable fraction 82.5%** (all
> credited MLP programs substituted simultaneously against a jointly-measured floor; superadditivity
> 2.59× vs sum-of-singles), with the distribution dual-reported (unweighted mean 47.8%). Attention
> credit is booked separately as reconstructibility margin (0.053 nats). Since these corrections,
> the **composed-fold method (§3a)** has superseded data fitting as the primary program source:
> composition matches or beats data-fit programs at MLP 0 (96.9 vs 97.9) and MLP 1 (99.5/95.1 vs
> 96.1) with all coefficient tensors from weights.

The current methodology in one sentence: **replace each component of the model with an explicit
program whose form matches that component's computational class, verify the replacement causally on
tasks and adversarially on inputs it was never fit on, and account for what remains against a
per-interface importance floor.** This document is the single current statement of the method;
findings live in `paper_atlas_bilin18.md` and `RESULTS_l0_mdl.md` §32; the MLP0/1 function
inventory in `mlp01_functions.md`.

## 1. Foundations: the objects and their gauge (tensor-network layer)

- A **bilinear MLP** `Down(L·x ⊙ R·x)` folds *exactly* into a symmetric third-order tensor
  `T[o,i,j] = Σ_p Down[o,p]·L[p,i]·R[p,j]` (verified to ~1e-7). Neuron permutation and rescaling
  leave `T` unchanged — the neuron basis is **pure gauge**, which is why we never do neuron-level
  interpretation. The canonical object is the tensor; any decomposition of it is a choice of basis.
- **Attention** here is softmax-free: the pattern is the product of two bilinear forms
  `(q1·k1)(q2·k2)/d²`, causal, unnormalized. Layer 0's pattern is an exact closed-form function of
  token identities (the exact fold + archetype dictionaries from the earlier arcs).
- The gauge/fold analysis is what tells us the **correct program family** per component (§3). This
  is the tensor network's load-bearing role: it names the function class.

## 2. The accounting frame: floors and understood-fractions

For every interface (each layer's QK-pattern input; each layer's MLP input — 36 in all):

- **Floor** = ΔCE when that interface's input is replaced by the dataset-mean vector (a constant:
  no token or context information crosses). This is the interface's importance.
- **Understood fraction** = 1 − ΔCE(our explanation substituted)/ΔCE(floor).
- The **ledger** sums these (first-order: single-interface floors are superadditive, so it is an
  accounting device, not an exact partition). Current: 9.95 nats total floor; ~79% explained before
  the mid-stack ladder polish, rising with it.

All audits are held-out (FineWeb subset disjoint from every fitting corpus); ΔCE in nats is the
only headline metric (MSE/FVU is used inside fitting only — it repeatedly mispredicts ΔCE).

## 3a. The primary method: composed folding with gauge scalars (2026-07-29 onward)

Two verified identities make the whole model an **exact tensor network with analytic scalar-gauge
nodes** (both at ~10⁻⁷, the only remaining nonlinearity being the final tanh/softmax readout):

- **MLP gauge:** `MLP(rms_norm(x)) = D · T(x,x)/‖x‖² + bias` — rms folds out of any homogeneous
  quadratic as a scalar (a ratio of two analytic quadratics).
- **Pattern gauge:** `pattern(i,j) = ⟨R_i q_i, R_j k_j⟩⟨R_i q2_i, R_j k2_j⟩ / (‖q1_i‖‖k1_j‖‖q2_i‖‖k2_j‖)`
  — a quartic multilinear numerator over four norm gauges; rope commutes with the rms scalar.

The **per-layer recipe** (verified through layer 1, extending upward): (1) split the layer's pre-rms
input into its analytic *streams* (embedding + each preceding component's output); (2) truncate each
attention stream to a **named basis** (archetype write-directions, ~144 dims); (3) chain MLP streams
through their own composed forms; (4) evaluate the layer's tensor as the **exact restriction** to
the named stream span, carrying the gauge scalar. Compression lives in the *streams*, never in
naked rank-truncation of the cores (measured to fail — see §5). Data's only role is choosing
truncation bases/metrics; all coefficient tensors come from weights. This exposes interaction
structure directly (e.g., MLP 1's blocks show attention-0 feeds it almost entirely *through*
MLP 0, with the MLP0×attn1 cross-term dominant).

## 3b. The fallback method: data-fit explicit-program substitution

Where composed streams are not yet available, write an explicit program in the component's
computational class, fit it on activations, substitute into the frozen model, and audit.

**For bilinear MLPs** (the class-matched family — a rank-R symmetric CP of the folded tensor,
fit in function space):

    MLP_L(x) ≈ TokenTable[token] + PrevTable[previous token] + Σ_{r=1..R} u_r · (a_r · x)²

- Tables are shrunk conditional means (data-estimated; token-keyed lexical memory made explicit).
- The R quadratic features (direction in → square → direction out) are fit by MSE on (input,
  output) activation pairs from a disjoint corpus. R is swept (64–512).
- A **scalar-only CE polish** afterward (per-feature output gains + table blend scalars, trained
  on cross-entropy through the frozen model) recovers the *behaviorally relevant* part of the
  MSE tail — legitimate because it cannot change the program's structure, only its calibration.
- Why this family: linear codes cannot express squaring; the earlier linear generator got 29% of
  MLP0 where this program gets 97.9%. **The lens must match the function class.**

**For attention functions** (exemplar: induction):

    pattern_head ≈ a · 1[token_{j−1} = token_i]  +  b · positional_template  +  c

- The match predicate is the hypothesized *meaning written as code*; the template is the head's
  mean position-based pattern; three scalars per head, read off by least squares.
- This passed completely: 100.5% of induction on shuffled sequences the parameters were never fit
  on — versus a 64–81% ceiling for linear symbol codes, because equality over 50k tokens is the
  worst case for low-rank linear compression. Same lesson as above, other direction.

**For attention layers as layers** (2–17): the compositional-symbol reconstruction (ridge from
preceding-layer codes) shows every layer's pattern is expressible from earlier variables — the
average-behavior lens — with its known limit (sharp matching is not linearly compressible). The
program upgrade replaces those linear codes function-by-function as functions get named.

**Functional localization** (which components matter for what): per-component mean-ablation
knockout across a task battery (the atlas); minimal circuits by backward-elimination with
selectivity nulls (random same-size subsets) and local-minimality checks.

## 4. The verification battery (the gates a claim must pass)

1. **Task preservation + no-breakage:** after substitution, the component's tasks must still work
   AND unrelated verified functions must not degrade (we re-check the induction service, natural
   ΔCE, and the task battery on every substitution).
2. **Hypothesis-driven generalization:** a functional claim must predict behavior on inputs it was
   not discovered on (shuffled sequences, novel periods, OOD corpora). This is where the induction
   claim partially failed at period 32 and where the circuit-vs-full-copy distinction surfaced.
3. **The code-verify loop for meaning claims:** write the claimed meaning as code from
   *independent* knowledge (e.g., grammar lists, not the feature's own top tokens), translate back
   into the model, compare against a delete-control. This falsified the "syntactic-class detector"
   naming of program features (coded = deleted), while the induction predicate passed — the loop
   separates real understanding from plausible labels. Include negative controls.
4. **Adversarial probing:** OOD stress cells scored against their own baselines, plus
   max-divergence hunts that actively search for where program and model disagree — the residual
   must be *named*, not just priced (MLP0's residual = multi-token word reassembly, pair-keyed,
   verified by a pair-table-vs-prev-table control).
5. **Gentle integration:** when a program replaces only part of a component's function, the
   surgical hybrid keeps the model's own computation and swaps only the claimed channel
   (exact model at initialization, bit-identical sanity check), with scalar-only finetuning as the
   non-structural knob. Full-replacement and hybrid ends of the Pareto are both reported.

## 5. Recurring failure modes (learned, now standing checks)

- **FVU/MSE mispredicts ΔCE** — never headline a variance number (MLP1's tail: 20% of variance,
  6% of function; MLP4: best FVU of the ladder, middling CE).
- **Top-token naming is a trap** — passes inspection, fails the code-verify loop.
- **Wrong content is worse than no content** — token tables *harm* mid-stack MLPs (MLP2 tables
  −48% vs bland floor); depth transitions from lexical to contextual.
- **Program-family mismatch masquerades as "not understandable"** — the 29%/64% linear ceilings
  were lens artifacts, not model properties.
- **Sufficiency ≠ the full mechanism** — the minimal induction circuit carries natural-text
  induction but only 38% of pure-copy capacity; redundancy pruned away still did work.

## 6. Workflow

Hourly cron tick (analyze, commit, restock) + the `qkqueue` supervisor daemon, which consumes
`QUEUE.txt` one script at a time whenever the GPU is free — "go by default" is structural: work
launches even if the analyst forgets to launch it. All scripts, JSON results, and this document
are committed with the findings they support.

> **Polish-on-null addendum (finding 6ii, measured):** the CE polish's 514 scalars lift even a
> random-feature program to 69.6% at MLP 4 (trained: 82.5%). Therefore the ledger's conservative
> credit column is **pre-polish (structural)**; post-polish numbers are reported alongside the
> null-polish reference. Interfaces whose structural credit is already dominant (MLP 0, 1, 16, 17)
> are unaffected; mid-stack post-polish figures should be read net of the null baseline.

## 5b. Failure modes added by the composed-fold and control arcs

- **Naked core truncation fails** — rank-truncating a bilinear tensor to shared quadratic features
  (any closed-form projection: Frobenius or Gaussian-under-metric) collapses below the floor; the
  token-axis variety must be carried by tables or by exact restriction to the stream span.
- **Polish capacity masquerades as program quality** at small interfaces (a null program polishes
  to 69.6% at MLP 4); the conservative credit column is pre-polish structural.
- **Black-box weight analysis undersells the weights** — the same tensor scored 69% analyzed alone
  and 96.9% composed with upstream structure; always compose before concluding data is necessary.
