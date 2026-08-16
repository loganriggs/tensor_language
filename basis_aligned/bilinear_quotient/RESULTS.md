# Results — bilinear quotient experiments (Part A)

Program spec: [`../bilinear-quotient-experiments.md`](../bilinear-quotient-experiments.md).
Shared machinery: `bq_common.py` (forms, Λ metric, whitening, reader moments, the four
nulls, five block-recovery routines, CP and dictionary fits). `bq_sanity.py` asserts the
machinery against known answers — 12/12 pass; run it before trusting anything below.

Setting. `y = D((Lx) ⊙ (Rx))`, per-output form `Q_i = sym(Σ_k D_ik l_k r_kᵀ)`, `y_i = xᵀQ_ix`.
`L, R, D` are a gauge choice of factorisation; the stack `Q` is the computation. Every
claim below is a claim about `Q`, and every one is checked against the four nulls
(random weights / gauge scramble / task shuffle / reader shuffle).

Status: **A1, A2, A3, A4 done.** A5, A6 and Part B not started.

| experiment | scripts | results |
|---|---|---|
| A1 parity/XOR kernel | `a1_parity.py` | `a1_results.json` |
| A2 modular addition | `a2_modular.py`, `a2_calibrate.py`, `a2_jade.py`, `a2_jade_struct.py`, `a2_extra.py`, `a2_diag.py` | `a2_*.json` |
| A3 CP calibration | `a3_cp.py` | `a3_results.json` |
| A4 planted quotient | `a4_quotient.py` | `a4_results.json` |

---

## A1 — parity / XOR: kernel verification

DGP. `z = [k bits in ±1 coding | n_dist continuous distractors]`, embedded as `x = Wz`.
Targets are pairwise parities, which in ±1 coding are the exact degree-2 monomials
`y_c = z_i z_j`, so the ground-truth forms are known: `Q_c = ½(e_ie_jᵀ + e_je_iᵀ)`.
Two arms differ only in whether the diagonal of the lift is probed: `pure` (bits exactly
±1, so `z_i² ≡ 1` and the magnitude coordinates are constant on-distribution) and
`graded` (bits carry a random magnitude, so the diagonal is probed to be zero).

### FINDING A1-1 — all three predictions hold exactly, and the blindness claim is causal to machine precision

`graded` arm, 3 seeds, `k=8`, 4 planted pairs, `n_dist=4`, `h=32`:

| quantity | prediction | measured (3 seeds) |
|---|---|---|
| effective rank of W | 4 (= number of planted pairs) | 4, 4, 4 |
| recovered `Q_c` vs planted | cos = 1 | 1.0000, 1.0000, 1.0000 |
| mass on the planted `(i,j)` entries | 1.0 | 1.0000 |
| blind subspace `∩_c ker(Q_c)` | = the 4 distractor directions | dim 4/4, subspace overlap 1.0000 |
| perturb along the blind subspace | zero output change | rms Δy / rms Δy(active) = **8.4e-16** |
| two inputs, same product, different magnitudes | identical output | rel change **4.6e-16** |
| same pair, product scaled by t=1.5 | rel change exactly t−1 = 0.5 | 0.5000 |

The last two rows are the crisp causal test the doc asks for: an input pair whose *lift
difference* is a planted kernel direction (pure magnitude) produces a bit-identical
output, while the row-space direction produces exactly the predicted change.

Note on method: with a constant learning rate this experiment silently degrades. Adam
normalises by the gradient RMS, so once the loss reaches machine precision it keeps
taking lr-sized steps on numerical noise and drifts back off the exact solution
(FVU 1e-25 at 2k steps → 2.8e-6 at 12k steps, and the blindness ratio degrades from
1e-13 to 8e-4). Annealing the learning rate to zero fixes it (FVU ~1e-31). `bq_common.train`
takes `cosine=True` for this reason.

### FINDING A1-2 — what the data never probes fills with junk that is causally visible

The `pure` arm solves the task exactly (FVU ~1e-31) but its forms are wrong:

| arm | cos(Q, planted) | mass on planted entries | mass on the diagonal | same-product input pair: rel output change |
|---|---|---|---|---|
| graded (diagonal probed) | 1.0000 | 1.0000 | 0.0000 | 4.6e-16 |
| pure (diagonal unprobed) | 0.73 / 0.83 / 0.93 | 0.56 / 0.57 / 0.66 | **0.44 / 0.43 / 0.34** | **0.51 / 0.39 / 0.36** |

With ±1 bits the lift's diagonal coordinates are constant, so no training signal
constrains them; 34–44% of the learned form ends up there. This is not inert: two inputs
with the same product but different magnitudes — which the task says are equivalent —
give outputs differing by 36–51%. The model is *not* blind to a distinction the task
never made, and only the causal test reveals it. (Same moral as FINDING 2 in the parent
folder's `RESULTS.md`, here with the equivalence class made explicit.)

The quotient therefore has two independent sources: directions the **weights** kill, and
directions the **data** never probes. Only the first is a property of the model.

### FINDING A1-3 — the anisotropy knob is a measurement problem, and whitening solves it

Measured on the *exact* hand-coded solution, so optimisation cannot confound it. The
analysis is done in `x` coordinates with `W` unknown (the realistic position):

| embedding κ | cond(Σ_x) | raw metric: rank / blind dim / overlap | Λ metric whitened by Σ_x |
|---|---|---|---|
| 1 | 1.2 | 4 / 4 / 1.000 | 4 / 4 / 1.000 |
| 10 | 92 | 4 / 4 / 1.000 | 4 / 4 / 1.000 |
| 100 | 9.2e3 | 4 / 4 / 1.000 | 4 / 4 / 1.000 |
| 1000 | 9.2e5 | **3 / 6 / 1.000** | 4 / 4 / 1.000 |

At cond(Σ) ~ 1e6 the raw threshold reports rank 3 and a 6-dimensional blind subspace
(both wrong); the whitened metric is still exact. The whitening story is real but the
threshold is generous — it only bites past cond(Σ) ≈ 1e4.

Separately, *training* under an anisotropic embedding fails long before analysis does
(κ=100: FVU 4.95, i.e. the student never learns the task). Worth keeping distinct: that
is an optimisation failure, not an interpretability one.

### A1 nulls

| null | outcome |
|---|---|
| 1. random weights | mass on planted entries 0.005–0.016 (vs 1.0), max cos(Q, planted) 0.04–0.16, blind dim 0 |
| 2a. gauge scramble of the **exact sparse** model | function-preserving to 3e-29; Q-level findings **bit-identical** (planted mass 1.0000 → 1.0000, blind dim 4 → 4) while the weight reading **dies**: L-row top-2 concentration 1.000 → 0.36 |
| 2b. gauge scramble of the **trained** model | Q identical (4.8e-28); the weight statistic barely moves (top-2 0.396 → 0.356) — because at h=32 the trained model is *already* dense in the privileged basis. There is no basis-reading finding in it for the null to kill. |
| 3. task shuffle | FVU 0.998, cos(Q, planted) 0.11–0.18, planted mass 0.036–0.043, blind dim 0 |
| 4. reader shuffle | **vacuous here** — one reader per planted form, so there is no reuse to destroy. Run in A2 and A5 instead. |

Null 2b is the more interesting half: the gauge null can only kill a finding that exists.
The honest statement is that the trained model has no weight-basis structure to lose,
which is itself the reason to work with `Q`.

### A1 knobs

Distractors 0/4/8/16: exact recovery throughout (cos 1.0000), blind subspace recovered in
full (16/16 at n_dist=16, overlap 0.99999999). Label noise σ=0.1/0.3: the *forms* stay
exact (cos 1.0000, planted mass 0.99/0.98) but the hard-thresholded blind subspace
collapses (4 → 2 → 0 directions detected) because the noise floor rises above the
threshold — a threshold-calibration effect, not a recovery failure, and exactly the
stratification-tolerance issue that A4 makes explicit.

---

## A2 — modular addition: ∗-algebra blocks and grokking

DGP. `x = [onehot(a) | onehot(b)] ∈ R^46`, target `(a+b) mod 23`, cross-entropy, 40% of the
529 pairs for training, AdamW with weight decay 1.0, h=128. Three seeds reach test
accuracy 0.978 / 0.934 / 0.953 by 40k steps.

### FINDING A2-1 — the planted algebra is finer than predicted: 22 blocks, not 11, and the canonical object is the isotypic component

The doc predicts "2×2 rotation blocks, one per frequency". Running exact SBD on the
*exact* planted family `Q_c = [[0,½N_c],[½N_cᵀ,0]]`, `N_c[a,b] = Σ_ω cos(2πω(a+b−c)/p)`:

- 22 blocks of dimension 2 (not 11 of dimension 4), in-block mass 1.000000
- commutant dimension 33 = 11 × 3

The extra factor of 2 is the **a↔b exchange symmetry** of `a+b` — the planted family
commutes with the swap operator exactly (`max|[Q_c, S]| = 0.0`), so each frequency's 4-dim
subspace splits into the symmetric and antisymmetric combinations of the a-side and
b-side Fourier planes. (A2-8 shows the *trained* models break this symmetry, and what that
costs.) And 3 = μ(μ+1)/2 at multiplicity μ=2 says those two blocks carry
*equivalent* modules — so the split into two 2-dim blocks is **not unique** (any basis of
the 2-dim multiplicity space gives a valid one). The canonical object is the 4-dim
isotypic component per frequency. `bq_common.isotypic_groups` merges fine blocks by
looking for intertwiners in the commutant, and recovers exactly 11 components of dim 4,
each matching a planted frequency with overlap 1.000000.

Practical consequence for the block machinery: **report isotypic components, not blocks.**
A pipeline that reports the fine blocks will report a gauge choice as a finding.

### FINDING A2-2 — instrument calibration: the commutant route dies at 1% noise, JADE survives 30%

Certification on known truth before use, per the standing rule. The planted family is
perturbed by controlled noise and both routes are asked to recover the 11 frequencies.
Scoring is strict: a frequency counts only if the union of blocks assigned to it spans
exactly that frequency's planted 4-dim subspace.

| off-block mass | exact SBD (commutant) | approximate commutant (`sbd_robust`) | JADE + tolerance (`sbd_jade`) |
|---|---|---|---|
| 0 | 11/11 | 11/11 | 11/11 (ε=0.001) |
| 8e-6 | **0/11** | 11/11 | — |
| 0.0088 | 0/11 | 11/11 | 11/11 (ε=0.01) |
| 0.0342 | 0/11 | 9/11 (block count supplied; 2/11 unaided) | 11/11 (ε=0.05) |
| 0.0968 | 0/11 | 3/11 | 11/11 (ε=0.15) |
| 0.1769 | 0/11 | 0/11 | 11/11 (ε=0.20) |
| 0.2903 | 0/11 | — | 11/11 (ε=0.40) |
| 0.4409 | 0/11 | — | 1/11 |

Three things this buys:

1. **Exact SBD is unusable on trained weights.** It asks for matrices that commute
   exactly; any noise makes the commutant trivial (span{I}) and it returns one block.
2. **The fix is to stop asking algebraically.** JADE (Cardoso–Souloumiac orthogonal joint
   approximate diagonalisation) minimises off-diagonal mass by Givens rotations with a
   closed-form optimal angle per rotation, then the partition is read off the residual
   coupling graph. Direct gradient descent on O(d) for the same objective does **not**
   work — it stalls in local minima (L1 objective: true basis 40.5, random basis 168.9,
   best of 4 gradient restarts 44.9).
3. **The tolerance is not a free parameter, it is the stratification cut.** `partition_from_coupling`
   returns the finest partition whose blocks still carry (1−ε) of the mass, by binary
   search. The calibration shows the required ε tracks the noise almost exactly
   (ε ≈ off-block mass), which is a usable operating rule.

The same holds for *structured* noise (off-block frequency-pair couplings, the kind
training actually leaves): 11/11 up to off-block 0.33 at ε=0.4.

### FINDING A2-3 — 29–35% of a trained model's form mass is not identifiable from the function at all

One-hot inputs probe only a 529-dimensional slice of the 1081-dimensional Sym²(V). The
minimum-norm representative of the same function on the data manifold (`canonicalise`)
drops 29–35% of the trained model's mass while changing the function by 6e-12:

| seed | identifiable mass fraction | off-block mass raw | off-block canonicalised |
|---|---|---|---|
| 0 | 0.712 | 0.2226 | 0.1777 |
| 1 | 0.646 | 0.2444 | 0.2075 |
| 2 | 0.705 | 0.2281 | 0.1717 |

The planted family is 0.99999999999978 identifiable, i.e. this is entirely a property of
the trained model, not of the parameterisation. Any weight-space geometry claim has to be
made after this projection; the discarded part is a gauge of the data.

### FINDING A2-4 — projecting the weights onto the Fourier blocks is a data-free generalisation surgery, and the circuit it isolates is complete 26× before the model uses it

Projecting the canonicalised forms onto the planted frequency blocks — a purely
weight-space operation, no data, no gradient:

| seed | test acc before → after | test CE before → after | functional residual removed |
|---|---|---|---|
| 0 | 0.9780 → **1.0000** | 0.1400 → 0.0006 | 0.319 |
| 1 | 0.9340 → **1.0000** | 0.2639 → 0.0026 | 0.341 |
| 2 | 0.9528 → **1.0000** | 0.1584 → 0.0011 | 0.314 |

So the ~32% of the function outside the block structure is not harmless junk. Evaluating
the **residual alone** (`Q_canonical − Q_block`) as a model settles what it is:

| seed | residual-only train acc | residual-only test acc | mean correct-class logit boost, train | on test |
|---|---|---|---|---|
| 0 | 0.9526 | **0.0000** | **+4.51** | **−2.99** |
| 1 | 0.9336 | **0.0000** | +4.70 | −3.12 |
| 2 | 0.9526 | 0.0031 | +4.43 | −2.94 |

The residual is a pure lookup table: 95% on the training pairs, 0% on held-out pairs. It
raises the correct logit by +4.5 on examples it has seen and *suppresses* it by −3.0 on
examples it has not. That is why deleting it doesn't merely leave the model intact — it
makes the model better. The block decomposition separates the circuit from the lookup
table cleanly enough that the split can be made in weight space alone.

Tracking the projection through training (seed 0, every 500 steps) gives the sharper
result:

| step | model test acc | Fourier-projected test acc | projected CE | ctrl: random 4-dim blocks | ctrl: relabelled-group Fourier |
|---|---|---|---|---|---|
| 0 | 0.031 | 0.006 | 3.138 | 0.037 | 0.038 |
| 500 | 0.000 | **0.994** | 1.638 | 0.013 | 0.055 |
| 1500 | 0.000 | **1.000** | 1.130 | 0.024 | 0.085 |
| 5000 | 0.053 | 1.000 | 0.468 | 0.040 | 0.122 |
| 13000 | ~0.50 | 1.000 | ~0.05 | ~0.09 | ~0.17 |
| 27000 | 0.881 | 1.000 | 0.001 | 0.152 | 0.199 |
| 39000 | 0.978 | 1.000 | 0.001 | 0.174 | 0.233 |

**The generalising circuit is fully formed at step 1500, when the model's own test
accuracy is 0.000 and it is a pure memoriser.** Grokking, in this model, is not the
formation of the circuit — it is the slow decay of the memorisation term that is drowning
it. The projected CE (1.130 → 0.001) shows what is actually still improving after 1500
steps: the circuit's *margin*, not its correctness.

Both matched controls fail as they should: projecting onto random 4-dim subspaces of the
same total dimension gives 0.01–0.17, and onto the Fourier blocks of a randomly
**relabelled** group (same shapes, same Fourier character, wrong group) gives 0.04–0.23.
The projection is not doing the work; the specific structure is.

![A2 dynamics](figures/a2_dynamics.png)

### FINDING A2-5 — weights-only block recovery on the trained model: partial, and the residual is harder than noise of the same size

Applying the calibrated JADE + tolerance pipeline to the trained models (no ground truth
used in the recovery, only in the scoring):

| model | off-block | best frequencies recovered | at ε | in-block mass |
|---|---|---|---|---|
| trained seed 0 | 0.178 | 4/11 | 0.20 | 0.802 |
| trained seed 1 | 0.208 | 1/11 | 0.25 | 0.756 |
| trained seed 2 | 0.172 | 5/11 | 0.20 | 0.804 |
| *calibration: isotropic noise at 0.177* | 0.177 | *11/11* | *0.20* | — |
| *calibration: structured noise at 0.200* | 0.200 | *11/11* | *0.25* | — |
| null: random weights | 0.91 | 0/11 | — | — |
| null: task shuffle | 0.90 | 0/11 | — | — |
| null: gauge scramble of seed 0 | 0.178 | **4/11 at ε=0.20 — identical to the trained model** | | |

Read carefully. Recovery is real (4–5/11 versus 0/11 for both nulls, and exactly
reproduced under the gauge null), and it beats the commutant route, which gets 0/11 here.
But it falls **well short of what the same off-block mass permits when the residual is
synthetic** — 11/11 for all three surrogates I could construct at the trained level:

| residual type at the trained model's noise level | off-block | frequencies recovered |
|---|---|---|
| isotropic symmetric noise | 0.177 | 11/11 |
| structured off-block frequency coupling | 0.200 | 11/11 |
| memorisation-shaped (train-set outer products) | 0.159 | 11/11 |
| **the trained model's actual residual** | 0.178 | **4/11** |

Scaling the model's *own* residual isolates the threshold. Writing the canonicalised model
as `Q = Q_block + α·Residual` and sweeping α over all three seeds:

| α | 0 | 0.25 | 0.50 | 0.75 | 0.80 | 0.85 | 0.90 | 0.95 | 1.00 |
|---|---|---|---|---|---|---|---|---|---|
| off-block (seed 0) | 0.000 | 0.013 | 0.051 | 0.108 | 0.122 | 0.135 | 0.149 | 0.163 | 0.178 |
| seed 0 | 11 | 11 | 11 | **11** | 8 | 9 | 6 | 5 | **4** |
| seed 1 | — | — | — | 8 | 7 | 6 | 3 | 3 | **1** |
| seed 2 | — | — | — | **11** | 9 | 9 | 8 | 6 | **5** |

So the pipeline handles ~75% of the real residual and then falls off a cliff between
off-block 0.10 and 0.13 — whereas isotropic noise is tolerated to 0.29 and structured
noise to 0.33. The trained residual is roughly 2–3× harder per unit of off-block mass than
any surrogate, and the decay is steep rather than gradual.

![A2 calibration](figures/a2_calibration.png)

The honest verdict is therefore *partial recovery, tool-limited, with the limitation
localised and quantified*: not "the instrument is too weak" but "this particular residual
is harder than noise of its size, by a measured factor of 2–3, and none of the three
obvious models of it reproduces that hardness." A2-8 identifies most of what makes it
hard, and removes it.

### FINDING A2-8 — the trained model breaks the task's input symmetry; restoring it is a second data-free surgery that both completes grokking and makes the blocks recoverable

`a+b = b+a`, so the exact planted family commutes with the a↔b swap operator exactly
(`max|[Q_c, S]| = 0.0`). This is *why* the isotypic multiplicities in A2-1 are 2. The
trained models do not: only 85–88% of their form mass is swap-equivariant.

Symmetrising is a purely weight-space operation using only the task's evident input
symmetry — no data, no gradients, and strictly less prior knowledge than the Fourier
projection of A2-4:  `Q ← ½(Q + S Q S)`.

| seed | swap-equivariant mass | test acc → | off-block → | JADE recovery → |
|---|---|---|---|---|
| 0 | 0.877 | 0.9780 → **1.0000** | 0.178 → 0.106 | 4/11 → **10/11** |
| 1 | 0.848 | 0.9340 → **1.0000** | 0.208 → 0.119 | 1/11 → **9/11** |
| 2 | 0.863 | 0.9528 → **1.0000** | 0.172 → 0.102 | 5/11 → **11/11** |

Nulls: on random weights the equivariant fraction is 0.536 (chance) and symmetrising moves
test accuracy 0.044 → 0.038; on the task-shuffled model 0.528 and 0.035 → 0.031, and JADE
on the symmetrised task-shuffled model still recovers 0/11. So the surgery only helps a
model that already contains a symmetric circuit, and the trained models are far above
chance equivariance.

Is symmetry-breaking *specifically* what defeats block recovery, or merely extra mass?
Splitting the residual into its swap-symmetric and swap-antisymmetric halves (52% / 48% of
the residual's mass) and rescaling each to the full residual's off-block mass:

| residual added to the block model, all at off-block 0.178 | frequencies recovered |
|---|---|
| symmetry-preserving half only | 7/11 |
| symmetry-breaking half only | **3/11** |
| both (the real residual) | 4/11 |
| synthetic isotropic noise | 11/11 |

The symmetry-breaking half is more than twice as damaging as the symmetry-preserving half
at equal mass — but *both* halves are far harder than synthetic noise. So symmetry-breaking
is a large part of the answer, not all of it, and symmetrisation helps twice over: it
deletes the worse half and halves the total residual at the same time. What remains hard
about the symmetry-preserving half is still open.

This gives a fully weights-only pipeline for this model: **symmetrise → JADE at ε ≈ 0.15 →
9–11 of 11 frequency blocks recovered → project onto them → test accuracy 1.000.** The only
task knowledge used is that the two input slots are interchangeable.

### FINDING A2-6 — verification: blocks ablate as predicted, and splice across independently trained models

**Ablation.** Removing one frequency block from the block-projected model drives that
frequency's logit contribution to zero (variance ratio 1.2e-33 to 2.9e-33 across all 11)
while its correlation with the ideal `cos(2πω(a+b−c)/p)` pattern before ablation is
0.90–0.95. Accuracy is a saturated readout here — 11 frequencies are redundant — so the
dose-response is the informative version:

| frequency blocks removed | test acc | test CE | equal-dimension random subspaces: acc | CE |
|---|---|---|---|---|
| 0 | 1.0000 | 0.0005 | 1.0000 | 0.0005 |
| 3 | 1.0000 | 0.0145 | 0.9790 | 0.2143 |
| 5 | 1.0000 | 0.0994 | 0.8145 | 0.8953 |
| 8 | 0.9403 | 0.9870 | 0.3187 | 2.3355 |
| 10 | 0.1981 | 2.3871 | 0.0933 | 2.9811 |

Cutting along the recovered blocks is uniformly the *least* damaging cut of a given
dimension — which is what it means for them to be the seams of the computation. Any 3
frequencies can be deleted with no accuracy loss at all.

**Splice.** Two independently trained models (seeds 0 and 1), each canonicalised into
frequency blocks; the hybrid takes frequencies 1–5 from A and 6–11 from B:

| A alone | B alone | A's half | B's half | **hybrid** | hybrid, phase-scrambled control |
|---|---|---|---|---|---|
| 1.0000 | 1.0000 | 0.8828 | 0.9830 | **1.0000** | 0.6163 |

The hybrid beats both halves, so this is genuine composition of parts from different
training runs, not one half carrying it. Rotating B's blocks within their own subspaces —
same energy, same frequencies, destroyed phase alignment — drops it to 0.6163, so what
transfers is the phase relationship the blocks encode, not merely their span.

### FINDING A2-7 — crystallisation coincides with the transition and then arrests

Prediction (iii) asked whether block crystallisation precedes or coincides with the
generalisation transition. Measured (seed 0, canonicalised off-block mass):

| step | 0 | 2500 | 10000 | 20000 | 30000 | 40000 | 200000 | 260000 |
|---|---|---|---|---|---|---|---|---|
| test acc | 0.031 | 0.006 | 0.406 | 0.566 | 0.940 | 0.978 | 0.987 | 0.991 |
| off-block mass | 0.913 | 0.580 | 0.344 | 0.311 | 0.206 | 0.178 | 0.213 | 0.214 |

Crystallisation is monotone and coincident with the test-accuracy rise — and then it
**stops**, at ~0.21 raw / ~0.18 canonicalised, and does not move between 40k and 260k
steps while test accuracy creeps to 0.991. The grokked bilinear solution is not the pure
Fourier solution and does not converge to it under continued weight decay. Combined with
A2-4 (deleting that residual gives test 1.000), the residue is both permanent and
harmful — training does not remove it, but weight surgery does.

### A2 nulls (summary)

| null | invariant as it should be | destroyed as it should be |
|---|---|---|
| 1. random weights | — | off-block 0.91, block-projected acc 0.02–0.07, phase coherence 0.08–0.36, 0/11 frequencies |
| 2. gauge scramble | ΔQ 6.8e-27; off-block 0.1777 → 0.1777; JADE recovery 4/11 → 4/11 | L-row Fourier concentration 0.53 → 0.28, 0.59 → 0.27 |
| 3. task shuffle | — | train acc 1.000 / test 0.035; off-block 0.90; block-projected test 0.035; 0/11 |
| 4. reader shuffle | set-level structure untouched (off-block 0.1777 → 0.1777) — SBD/JADE depend only on the *set* of forms, so this null is **vacuous for them** | the reader-indexed claim dies: phase coherence (per-class rotation angle linear in c) 0.980 → 0.199, 0.977 → 0.231 |

Null 4 deserves its own note. Permuting which class gets which form leaves the *set*
`{Q_c}` unchanged, so every set-level structural claim (blocks, algebra, spectra) is
automatically invariant and the null tests nothing about them. It only bites on
reader-indexed claims. The one here is that inside frequency ω's block the per-class 2×2
cross matrix is the reflection at angle `θ_c = 2πωc/p` — measured as circular coherence
0.97–0.99 in the trained models, 0.20–0.23 under the shuffle, 0.08–0.36 at random init.
A pipeline whose only claims are set-level should say so rather than report null 4 as
passed.

---

## A3 — planted low-rank teacher: CP calibration

DGP. `y_i = Σ_k γ c_ik (a_kᵀx)(b_kᵀx)`, known `{a_k, b_k}`, `d=16`, `m=8`, Gaussian inputs.
The gauge-invariant object per component is the symmetric form `S_k = sym(a_k b_kᵀ)`
(invariant to `(a,b) → (ca, b/c)`, joint sign flip, and `a↔b`), so recovery is scored on
`{S_k}` matched by Hungarian assignment. Two arms: `solver` fits CP to the **exact**
teacher forms (isolating the fitter) and `learned` fits CP to a trained student.

### FINDING A3-1 — the operating range: components are trustworthy to K ≈ 1.5d, and the fit error does not warn you past it

Arm 1, exact teacher forms, `d=16` (so `dim Sym² = 136`), 2 seeds:

| true rank K | K/d | CP fit error (Λ) | mean matched cos | fraction > 0.9 | fraction > 0.99 |
|---|---|---|---|---|---|
| 2, 4, 8, 12, 16 | ≤ 1.0 | ~2e-32 | **1.000** | 1.00 | 1.00 |
| 24 | 1.5 | 6e-31 | **1.000** | 1.00 | 1.00 |
| 32 | 2.0 | 4e-18 | 0.957 | 0.97 | 0.03 |
| 48 | 3.0 | **1.7e-32** | 0.907 | 0.65 | 0.00 |

Two things to carry into the other experiments:

1. **Components are trustworthy as variables well into superposition** — exact recovery at
   K = 24 with d = 16, i.e. 1.5× more components than input dimensions. Past K ≈ 2d they
   degrade.
2. **The fit error is a one-sided diagnostic.** At K=48 the CP fit is *perfect*
   (1.7e-32, machine precision) while a third of the components are wrong. A low
   reconstruction error is not evidence that the components mean anything. This is the
   single most important caveat the calibration produces, and it is the reason A4's
   component decomposition is scored against planted directions rather than against its
   own residual.

### FINDING A3-2 — correlation between planted directions, not learning, is the binding constraint

Arm 1b (solver) and arm 2c (learned) agree closely, so this is a property of the
decomposition problem, not of the student:

| correlation ρ between planted directions | solver: mean cos / frac > 0.9 | learned: mean cos / frac > 0.9 |
|---|---|---|
| 0.0 | 1.000 / 1.00 | 1.000 / 1.00 |
| 0.3 | 1.000 / 1.00 | 1.000 / 1.00 |
| 0.6 | 0.938 / 0.75 | 0.943 / 0.88 |
| 0.9 | 0.763 / 0.38 | 0.699 / 0.38 |

By ρ = 0.9 only 38% of components are recovered to cos > 0.9 — and again the fit error
does not signal it (3.5e-5). Anywhere the planted directions are near-parallel, CP
components should not be read as variables.

### FINDING A3-3 — label noise does not corrupt the forms; a capacity-limited student does

Arm 2b, K=8, h=32, label noise σ:

| σ | student FVU | form error vs teacher | mean matched cos |
|---|---|---|---|
| 0 | 1.3e-5 | 0.000 | 1.000 |
| 0.05 | 2.5e-3 | 0.000 | 1.000 |
| 0.2 | 3.8e-2 | 0.000 | 1.000 |
| 0.5 | **0.199** | **0.000** | **1.000** |

At σ=0.5 the student's FVU is 0.199 and its interaction forms are still exactly the
teacher's — the noise is averaged away and lands entirely in the irreducible error, not in
the weights. Recovery quality therefore should not be read off FVU.

What *does* hurt is a student without slack: at `h = K` the student stops fitting
(K=16: FVU 1.8e-2, K=24: FVU 5.8e-3) and recovery falls to cos 0.85–0.90 with only 56–62%
of components above 0.9, while the same K at `h = 4K` or `h = 64` recovers perfectly. The
degradation tracks the fit error, i.e. it is an optimisation failure passed downstream, not
a CP failure.

### FINDING A3-4 — over-ranked fits are the dangerous case

Arm 2d, true K=8:

| fit rank | CP fit error | mean matched cos | frac > 0.9 | subspace recovery |
|---|---|---|---|---|
| 4 | 2.8e-1 | 0.951 | 1.00 | 0.500 |
| 6 | 8.0e-2 | 0.953 | 0.83 | 0.750 |
| **8** | 1.2e-5 | **1.000** | 1.00 | 1.000 |
| 10 | 1.2e-5 | 0.915 | 0.75 | 1.000 |
| 16 | 1.0e-5 | 0.987 | 1.00 | 1.000 |

Under-ranking is *safe and self-announcing*: the fit error jumps by four orders of
magnitude, and the components that are found are still correct (cos 0.95) — you simply get
a subset. Over-ranking is neither: the fit error is indistinguishable from the correct rank
(1.2e-5 at r=10 versus 1.2e-5 at r=8) while a quarter of the components have been split or
duplicated. **Choose the rank by the elbow in the fit error, never by "it fits well".**

### A3 nulls

| null | outcome |
|---|---|
| 1. random weights | mean matched cos 0.130 / 0.143 / 0.117 against the planted set, **0% above 0.9** (versus 1.000 and 100% for the trained student) |
| 2. gauge scramble | refactorisation residual 2e-28 / 9e-28, hidden width `h` 32 → 136, and recovery **unchanged**: mean cos 1.000 → 1.000, frac > 0.9 1.00 → 1.00 |
| 3. task shuffle | student trained on permuted targets: mean matched cos 0.164, 0% above 0.9 |
| 4. reader shuffle | **vacuous for component recovery** — permuting the reader index changes only the mixing matrix `c`, not the component set `{S_k}`. It does bear on the mixing claim, which is not tested here. |

Caveat on one metric: `subspace_recovery` is degenerate at K ≥ d/2, because 2K factor
vectors in `d` dimensions span everything — it reads 1.000 even for random weights. Only
the matched-cosine numbers carry information at these ranks.

---

## A4 — planted quotient: the stratification and the linearization band

DGP. `x = μ + z` in `R^48` (8 directions never read). The teacher is
`y_i = Σ_j γ_j c_ij (a_jᵀx)(b_jᵀx)` over 18 components laid out on a **3×3 design in two
independent axes**, because the doc's prediction names only one of them:

- `γ ∈ {1, 0.1, 0.01}` — the gain, i.e. how much of the function the component is. This is
  the axis the doc's "intermediate singular value" refers to.
- `ρ ∈ {0, 2, 10}` — the mean-to-fluctuation ratio along `a_j, b_j`, i.e. how *curved* the
  component is on-distribution.

Why two. Linearising `(aᵀx)(bᵀx)` about the input mean leaves the residual `(aᵀδ)(bᵀδ)`
with `δ` the fluctuation, so linearisability should be set by `ρ`, not by `γ`. The design
lets the hypotheses disagree. Components read disjoint pairs from a random orthonormal
frame, which makes the realised `ρ_j` exactly the planted one (measured: 0.00, 1.99, 9.96,
… against planted 0, 2, 10) and makes the component forms Sym²-orthogonal. Student:
h=192 on the lifted input `[1, x]` so affine terms are representable, FVU 6.8e-6.

### FINDING A4-1 — planted collapse and the planted gain profile both come back

**(i) Hard collapse.** Perturbing along the 8 never-read directions moves the output by
rms 8.9e-3 against 1.9e0 for a matched live perturbation — blindness ratio **4.6e-3**,
which is the square root of the student's FVU, i.e. blindness holds down to the fit's own
noise floor and not further. The sharp version is algebraic: 99.995% of the dead
directions' lifted mass is captured by the kernel of the lift map.

**(ii) Contraction spectrum.** Recovered gains against planted, by cell:

| planted γ | recovered (6 components each) |
|---|---|
| 1.00 | 0.975, 0.988, 0.990, 0.945, 0.983, 0.996 |
| 0.10 | 0.038, 0.082, 0.055, 0.077, 0.092, 0.090 |
| 0.01 | 0.0086, 0.0088, 0.0057, −0.0041, 0.0012, 0.0080 |

The two-decade profile is reproduced and the three levels never overlap, but the
per-component estimates are only good to a factor of ~2 at γ=0.1 and to sign at γ=0.01.
The contraction spectrum is a reliable *ordering* and an unreliable *value* at this fit
quality (decomposition residual 1.0e-2).

### FINDING A4-2 — the linearization band is real: it improves the Pareto frontier at 22 of 25 budgets, by up to 10×

Per component, three surgeries: keep (2d+m parameters), linearise (d+m — one tangent
covector), prune (0). A DP over the additive surrogate selects an assignment for each
budget, and the reported error is then **measured exactly** on that assignment.

| parameter budget | keep/prune only | + linearize | improvement |
|---|---|---|---|
| 229 | 2.24e-2 | 4.62e-3 | 4.9× |
| 459 | 4.43e-3 | 5.27e-4 | 8.4× |
| 688 | 4.40e-4 | 1.74e-4 | 2.5× |
| 918 | 7.73e-5 | 2.15e-5 | 3.6× |
| 1147 | 1.84e-5 | 1.83e-6 | 10.1× |
| 1377 | 5.44e-7 | 2.35e-7 | 2.3× |
| 1606 | 9.28e-8 | 2.02e-8 | 4.6× |
| 1836 (full) | 2.1e-32 | 2.1e-32 | 1.0× |

Linearisation strictly wins at 22/25 budgets and ties at the two endpoints (where
everything is pruned or everything is kept), which is the predicted shape.

![A4 frontier](figures/a4_frontier.png)

Caveat, stated plainly: the surgery errors are **not** additive across components
(max deviation 33% on random assignments; DP surrogate vs measured error deviates by up
to 1.07 relative). The components are Sym²-orthogonal, but the functional error is
measured under an input distribution with a nonzero mean, and the cross terms do not
vanish. So the reported curve is an *achievable* frontier obtained by an identical
procedure on both sides — a fair comparison, not a proven optimum.

### FINDING A4-3 — the band is not where the doc predicts: curvature governs it, gain does not

Correlation of `log(linearize error / prune error)` across the 18 components:

| against | correlation |
|---|---|
| `log γ` (the doc's predicted axis) | **+0.000** |
| `log(1 + measured ρ)` (curvature) | **−0.993** |

Mean linearize/prune error ratio by cell:

| by γ | 0.01 | 0.1 | 1.0 |
|---|---|---|---|
| ratio | 0.3468 | 0.3464 | 0.3467 |

| by ρ | 0 | 2 | 10 |
|---|---|---|---|
| ratio | 1.000 | 0.0398 | 9.7e-5 |

Gain is *exactly* uninformative about linearisability — the three γ levels agree to four
significant figures — while curvature spans four orders of magnitude. At ρ=0 the ratio is
exactly 1.000 because the tangent at the mean is zero, so linearising *is* pruning.

![A4 band axis](figures/a4_band_axis.png)

The doc's prediction that the linearisation band sits at "intermediate singular value" is
therefore wrong as stated, and the correction is clean:

> **ρ decides whether a component can be linearised; γ decides whether it is worth the
> budget.** They are orthogonal axes, and only the first is about linearisation at all.

The budget sweep shows both roles at once — the linearised set is *always* drawn from
ρ ∈ {2, 10} and never from ρ=0, while *which* of those get linearised marches down the γ
ladder as the budget grows:

| budget | keep / linearize / prune | linearized cells (γ, ρ) |
|---|---|---|
| 229 | 0 / 4 / 14 | (1, 10) ×2, (0.1, 10) ×2 |
| 459 | 1 / 6 / 11 | (1, 2) ×2, (1, 10), (0.1, 10) ×2, (0.01, 10) |
| 918 | 6 / 5 / 7 | (0.1, 2), (0.1, 10) ×2, (0.01, 10) ×2 |
| 1377 | 11 / 4 / 3 | (0.01, 2) ×2, (0.01, 10) ×2 |
| 1836 | 18 / 0 / 0 | — |

This is the doc's "the band's location moves with the downstream tolerance ε", confirmed —
but it moves along the γ axis *within* the high-ρ set, rather than the band itself being a
γ interval.

### A4 caveats

- Only one teacher seed and one student were run. The γ and ρ effects are large (four
  orders of magnitude on the axis question) but the per-component gain estimates would
  benefit from seeds.
- The component decomposition uses the planted directions (oracle). A3 bounds what a
  blind CP recovery would cost here; wiring A3's fitter into A4 in place of the oracle is
  the obvious next tick.
- Nulls 1–4 were not run for A4: the claims here are about a planted teacher and the
  surgery arithmetic, not about discovering structure, so the null battery as specified
  does not apply cleanly. This is a gap relative to the doc's "every experiment ships
  with its four nulls" rule and should be closed if A4's claims are used downstream.

---

## Method notes worth carrying forward

1. **Anneal the learning rate whenever the target is exactly realisable.** Adam normalises
   by the gradient RMS and will walk a machine-precision solution back out. This changed
   A1's blindness ratio by 12 orders of magnitude.
2. **Canonicalise before measuring weight-space geometry.** Project the forms onto the
   span of the lifted data. On modular addition 29–35% of the trained mass is otherwise
   unidentifiable noise that changes every mass-fraction statistic.
3. **Never use the exact commutant on trained weights.** Certify the block instrument
   first; the operating ranges differ by 30× between routes.
4. **Report isotypic components, not blocks,** whenever an irrep can repeat.
5. **The gauge null can only kill a finding that exists.** Check that the basis-reading
   statistic is above chance in the trained model before claiming the null passed.
6. **Reader shuffle is vacuous for set-level claims.** Say which claims it tests.

---

## What Part A changes about the plan

Three items in the spec should be updated before A5/A6/Part B are built on top:

1. **A2's block prediction was too coarse and A4's band prediction was on the wrong axis.**
   Blocks: report isotypic components, not blocks (A2-1). Band: curvature, not gain (A4-3).
2. **"Every experiment ships with its four nulls" needs a caveat.** Null 4 (reader shuffle)
   is vacuous for any claim about the *set* of forms — which includes every block, algebra
   and spectrum claim. It tests reader-indexed claims only. Null 2 (gauge scramble) can
   only kill a finding that exists, so it must be paired with a check that the
   basis-reading statistic was above chance to begin with. Both are recorded per experiment
   above rather than reported as blanket passes.
3. **Canonicalisation belongs in section 0's shared machinery.** The spec's whitening step
   handles anisotropy but not unidentifiability. On modular addition 29–35% of the trained
   form mass is invisible to the function (A2-3); on parity the never-probed diagonal takes
   34–44% and is causally harmful (A1-2). Project onto the span of the lifted data first.

## Next

- **A5 / A6** are the natural continuation: `bq_common` already has `reader_moment`,
  `fit_dictionary`, `form_shuffle` and the path machinery they need, and A3 now bounds what
  the dictionary and CP arms can claim.
- **The open question from A2-5/A2-8** — why the symmetry-*preserving* half of a trained
  residual is still 1.5× harder for block recovery than synthetic noise of the same size —
  is a concrete, cheap target, and it is what stands between 9–11/11 and 11/11 on a
  weights-only pipeline.
- **A4 should be re-run with A3's CP fitter in place of the oracle component
  decomposition,** and with its four nulls, before its numbers are used downstream.
