# THREE-HOURLY MATHEMATICAL REVIEW — 2026-09-04 04:04Z

State read from disk before writing: ledger tail SS2802–SS2808, BENCHMARK_BACKLOG tail, board tail (Codex's R593 different-agent
review at 03:35Z; my directive relay at 03:47Z), runlogs/_completed.txt (03:32 canary2, 03:40 read-split), queue (circuit_battery
running since 04:03:46), and the newest strategic review (HOURLY_STRATEGIC_REVIEW_2026-09-04_0327.md). Two user directives dominate
this review and are treated as constraints, not context: 03:21Z "focus back on the circuits and going in more depth… what's
preventing yall from having 20 high quality circuits in the next 12 hours?" and 03:43Z "why do you need fresh data for every unique
circuit? We should do the 20/80 here… build tools that can be built once and reused, and then scale."

## 0. What changed the problem since the last mathematical review

SS2808 measured something that is a MATHEMATICAL fact about this network, not a fact about one prompt family: the numbered-list
successor is not carried by the attention-8 write T (that write is a context-blind copy of the last label); it is computed by a set of
MLP readers of T whose single-reader damages sum to .994 while their joint damage is 1.914. The damage set function over readers is
strongly SUPER-ADDITIVE. Every mathematical move below is chosen because it addresses that object — the write→reader-set map — rather
than because it is fashionable machinery.

Independent literature check run for this review (a subagent with web access; citations verified to the extent the search returned
them, and the three 2026 arXiv ids are cited as reported, not personally fetched):

- Causal abstraction and interchange-intervention accuracy: Geiger et al., *Causal Abstractions of Neural Networks*, NeurIPS 2021,
  arXiv:2106.02997; DAS, *Finding Alignments Between Interpretable Causal Variables and Distributed Neural Representations*, CLeaR
  2024, arXiv:2303.02536. The operational criterion is exactly what our battery's LOCALISE stage computes.
- The critique that matters to us: *The Non-Linear Representation Dilemma: Is Causal Abstraction Enough for Mechanistic
  Interpretability?*, arXiv:2507.08802 (2025) — a sufficiently flexible learned alignment can score high interchange accuracy without
  being the network's mechanism; and *Many Circuits, One Mechanism*, arXiv:2606.06267 (2026) — the discovered circuit is partly an
  artifact of which corrupted-input distribution you chose.
- Self-repair / redundancy, i.e. SS2808's super-additivity as a known phenomenon: Wang et al., IOI backup name-movers, ICLR 2023,
  arXiv:2211.00593; McGrath et al., *The Hydra Effect*, arXiv:2307.15771; *Conditional Co-Ablation*, arXiv:2607.01940 (2026), whose
  point is precisely that single-component ablation systematically under-counts redundant circuitry.
- Benchmarks / MDL: MIB, ICML 2025, arXiv:2504.13151 (circuit-localization and causal-variable tracks — the closest existing analogue
  of our battery); Braun et al., *Interpretability in Parameter Space* (APD), arXiv:2501.14926, which states "is this circuit a real
  compression" as an explicit MDL objective.
- Minimal realization / Hankel rank / bisimulation applied to transformer circuits: the search found NO current work. Adjacent only
  (RNN realization theory; SVD of QK/OV circuits). Treat as an open gap, not a literature to import.

## 1. Ideas pruned (and why), before ranking

- **Learned alignment search (DAS-style rotations) for our battery.** Pruned. It would add fitted parameters to a protocol whose
  entire advantage is that it has none, and arXiv:2507.08802 says the metric it optimizes is exactly the gameable one. Our REC is a
  capacity-free measurement of an unfitted component; that is a feature.
- **More whole-module / rank sweeps of the late tail.** Pruned by user directive and by SS2807 (the "gate frame" is the pooled
  variance frame; cost = lost energy to <= .002 at every rank). Closed.
- **Bisimulation / automata minimization of the residual stream.** Pruned for now despite being a genuine literature gap: the
  quotient it would build is over a state space we cannot enumerate, it optimizes reconstruction rather than prediction or removal,
  and it has no cheap falsifier. Recorded as an open direction, not scheduled.
- **Information-bottleneck compression of the writer's write.** Pruned: it optimizes local MSE-like objectives and does not compose
  across the RMSNorm interface (a bottleneck fitted on x is not a bottleneck on x/||x||).
- **Tensor-rank / simultaneous factorization of the MLP bilinear forms.** Not pruned but deprioritized: SS2118/SS2127/SS2131 closed the
  cheap-interface versions, and the new evidence (redundant reader SETS) says the interesting structure is BETWEEN components, which a
  per-component factorization cannot see.

## 2. The top three genuinely new mathematical moves

### Move 1 (rank 1) — the exact (2,2)-RATIONAL response certificate of a bilinear reader

**Object.** For a writer's final-position write W and a downstream reader (an MLP block) with input x, the path-patched arm computes
`mlp(rms_norm(x - tW))` for the removal fraction t (the battery uses t = 1; SS2808 used t = 1).

**Theorem (elementary, exact for this architecture).** bilin18's MLP is a BILINEAR form: `mlp(u)` is homogeneous of degree 2 in u.
RMSNorm is `u -> sqrt(d) u / ||u||`. Hence
`mlp(rms_norm(u)) = d * mlp(u) / ||u||^2`,
so along the removal ray `u(t) = x - tW` the reader's output is EXACTLY a rational function of t whose numerator is a vector of
quadratics and whose denominator is the scalar quadratic `||x||^2 - 2t<x,W> + t^2||W||^2`. Five scalars per reader per row determine
the whole curve: `mlp(x), mlp(W), the polarization term, <x,W>, ||W||^2`. This is not an approximation and not a fit.

**Measurable consequence beyond reconstruction.** (i) A CERTIFICATE: from three values of t the entire removal curve is predicted
exactly, so any deviation is a bug or a violated assumption, not noise — a real approximation certificate for every reader edge in
every circuit. (ii) A PREDICTION: the damage from a partial removal (t = .5) is not half the damage from a full one, and the exact
curvature is computable in closed form; a linear/attribution-patching approximation (arXiv:2310.10348) is therefore provably wrong
for this architecture by an amount we can state. (iii) It gives the first principled decomposition of a reader's damage into a
"direction" part (numerator) and a "norm/gate" part (denominator) — the second is exactly the RMSNorm gain change that
attribution methods silently ignore.

**Assumptions that may fail.** The block's MLP must be exactly the bilinear form with no elementwise nonlinearity outside it (must be
checked against `jacclust/tt_model.py`, not assumed); the residual x must be held fixed while t varies, which is true only for a
single reader's arm (multi-reader arms change each other's inputs — that is precisely why joint removal is super-additive, and the
certificate makes that statement exact rather than anecdotal).

**Cheapest falsifying experiment.** On the battery's top-3 readers per capable behaviour, sweep t in {0, .25, .5, .75, 1} (5 forwards
per reader per task, ~250 forwards total, < 1 GPU-minute), fit the exact rational form from t in {0, .5, 1} and predict t = .25, .75.
Falsified if the max relative deviation exceeds 1e-4. If it holds, every reader edge in the whole battery gains a closed-form curve.

### Move 2 (rank 2) — the Möbius/Harsanyi interaction transform of the reader damage set function

**Object.** SS2808's damage as a set function `v: 2^R -> R` over reader subsets, with `v(single)` summing to .994 against
`v(all) = 1.914`.

**Operational definition.** The Möbius (Harsanyi) transform `m(S) = sum_{T subset S} (-1)^{|S|-|T|} v(T)`. The order-1 coefficients
are the single-reader damages; order-2 coefficients are computable from pairwise joint removals. A circuit whose readers are a
k-of-n threshold has a characteristic signature (near-zero order-1, large positive order-2/3), which is exactly what SS2808's numbers
hint at and what the self-repair literature (arXiv:2307.15771, arXiv:2607.01940) predicts.

**Measurable consequence.** A per-circuit REDUNDANCY ORDER — the smallest k such that removing any k readers accounts for >= half of
the joint damage — which is a claim about executable cost (how many components a compiled program must keep) rather than about
reconstruction. It also converts "the effect is distributed" (an admission) into "the effect is a 2-of-5 threshold" (a statement).

**Assumptions that may fail.** The transform needs 2^k evaluations; k must be capped at 4-5. Damages can be negative (SS2808's mlp13),
so no ratio is admissible — differences only.

**Cheapest falsifying experiment.** Top-4 readers per capable behaviour: 11 extra arms (4 singles already measured, 6 pairs, 1 quad)
x ~10 behaviours ~ 700 forwards, ~2 GPU-minutes. Null: all order-2 coefficients within noise of zero (readers additive).

### Move 3 (rank 3) — capacity control for the interchange measurement

**Object.** The battery's `REC(c)` for each component c.

**Why.** arXiv:2507.08802 and arXiv:2606.06267 say an interchange score is only interpretable against the score a
same-capacity irrelevant intervention would get. We fit nothing, so our exposure is smaller than DAS's — but not zero, because a
component with a large output norm can move a logit difference without carrying the variable.

**Operational definition.** For each chosen writer, report REC alongside `REC_rand`: the same interchange patch performed with a donor
drawn from a DIFFERENT family (a P-family donor, whose answer is unchanged) and with a norm-matched random direction. A writer whose
REC is not well separated from REC_rand is not a writer.

**Cheapest falsifying experiment.** 2 extra arms per behaviour, ~50 forwards, seconds. This is the cheapest of the three and should
simply be folded into the battery's next version as a permanent column.

## 3. Executed this hour

The highest-priority safe unblocked implementation was NOT one of the three above but the thing that makes all three cheap: the
reusable battery itself (user directive 03:43Z). Built and committed this hour: `ops/circuit_battery_tasks.py` (16-behaviour bank,
mechanical four-family / three-split generator with disjoint pools), `ops/circuit_battery.py` (capability -> localise -> split ->
selectivity -> held-out engine, exact residual path-patching, zero fitted parameters), `ops/test_circuit_battery_tasks.py` (8 unit
tests, all passing), and ONE protocol preregistration
(`CIRCUIT_BATTERY_PROTOCOL_PREREGISTRATION.md`, sha d60b4c0c…) that replaces per-circuit preregistrations. Measured cost of a
behaviour: ~3 GPU-seconds at 6 rows/cell. The registered 16-behaviour run was enqueued at 04:03Z.

Move 3 is folded into the battery's next version; Move 1 is preregistered next (it needs the battery's writer/reader table as input,
which lands with the run in progress); Move 2 follows it.
