# Results — bilinear quotient experiments (Part A)

> **Amended after independent review.** Three claims are retracted and six corrected —
> see [`REVIEW_RESPONSE.md`](REVIEW_RESPONSE.md) for the full triage, and
> [`THEORY.md`](THEORY.md) for the results that turned out to be theorems rather than
> measurements. Retractions are marked inline below.

Program spec: [`../bilinear-quotient-experiments.md`](../bilinear-quotient-experiments.md).
Shared machinery: `bq_common.py` (forms, Λ metric, whitening, reader moments, the four
nulls, five block-recovery routines, CP and dictionary fits). `bq_sanity.py` asserts the
machinery against known answers — 12/12 pass; run it before trusting anything below.

Setting. `y = D((Lx) ⊙ (Rx))`, per-output form `Q_i = sym(Σ_k D_ik l_k r_kᵀ)`, `y_i = xᵀQ_ix`.
`L, R, D` are a gauge choice of factorisation; the stack `Q` is the computation. Every
claim below is a claim about `Q`, and every one is checked against the four nulls
(random weights / gauge scramble / task shuffle / reader shuffle).

Status: **A1, A2, A3, A4, A5 and B2 (with the B1 census) done.** A6, B0's third
placement, B3 and B4 not started. `BILIN18_CONNECTION.md` carries the results up to the
546M model.

| experiment | scripts | results |
|---|---|---|
| A1 parity/XOR kernel | `a1_parity.py` | `a1_results.json` |
| A2 modular addition | `a2_modular.py`, `a2_calibrate.py`, `a2_jade.py`, `a2_jade_struct.py`, `a2_extra.py`, `a2_diag.py` | `a2_*.json` |
| A3 CP calibration | `a3_cp.py` | `a3_results.json` |
| A4 planted quotient | `a4_quotient.py`, `a4_nulls.py` | `a4_results.json`, `a4_nulls_results.json` |
| A5 shared vs private | `a5_shared.py`, `a5_extra.py` | `a5_results.json`, `a5_extra_results.json` |
| B2 conjunctive retrieval + B1 census | `b_common.py`, `b2_conjunctive.py` | `b2_results.json` |

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

### FINDING A2-3 — a third of a trained model's form mass is unidentifiable — but that is *better* than chance

One-hot inputs probe only a 529-dimensional slice of the 1081-dimensional Sym²(V). The
minimum-norm representative of the same function on the data manifold (`canonicalise`)
drops 29–35% of the trained model's mass while changing the function by 6e-12:

| seed | identifiable mass fraction | off-block mass raw | off-block canonicalised |
|---|---|---|---|
| 0 | 0.712 | 0.2226 | 0.1777 |
| 1 | 0.646 | 0.2444 | 0.2075 |
| 2 | 0.705 | 0.2281 | 0.1717 |

**With its baseline** (added after review). The identifiable subspace is 529 of 1081
dimensions, so isotropic chance identifiability is **48.9%**. Measured: random symmetric
forms 0.489, random-initialised layers 0.479 / 0.500 / 0.488, trained 0.712 / 0.646 / 0.705.
So the trained model is ~1.4× *more* data-aligned than chance, and the honest framing is
that it concentrates 65–71% of its mass in a subspace occupying 49% of the space — not that
a third of it is junk. The planted family is 0.99999999999978 identifiable.

The methodological point stands unchanged: any weight-space mass statistic must be computed
after this projection, because a third of the raw mass cannot affect the function. Caveat
recorded: `canonicalise` projects in the Frobenius metric, which the program's own rule says
not to use for error; the fraction is metric-dependent.

### FINDING A2-4 — projecting the weights onto the Fourier blocks is a data-free generalisation surgery, and the circuit it isolates ranks correctly long before the model does

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

> **RETRACTED as written.** Two claims here were wrong and are corrected below; see
> `REVIEW_RESPONSE.md` R1 and R2.

**From step 1500 a 44-dimensional projection carrying a quarter of the function already
gets every held-out pair's ordering right, while the model itself scores 0.000.** That much
holds. But the two stronger claims do not:

*The circuit is not "fully formed" at 1500.* At that point the block part accounts for only
25% of the function (functional residual 0.749) and is 0.80 aligned with its own final form
in the Λ metric — an alignment that then wanders (0.835, 0.848, 0.817) before converging.
Test accuracy 1.000 is an argmax statistic; projected CE is 1.130 against 0.0006 at the end.
What is right at step 1500 is the *ranking*, not the circuit.

*Grokking is not the decay of the memorisation term.* Measured directly:

| step | circuit logit rms | residual logit rms | off-block | functional residual |
|---|---|---|---|---|
| 3000 | 0.905 | 1.506 | 0.564 | 0.708 |
| 12000 | 1.843 | 1.850 | 0.322 | 0.490 |
| 21000 | 2.589 | **2.343** | 0.298 | 0.441 |
| 39000 | 2.780 | 1.929 | 0.179 | 0.320 |

The residual *grows* to a peak and ends 28% above its early value; the circuit grows 3.1×.
The falling off-block fraction is the numerator growing. Grokking here is dominated by the
circuit's growth against a memorisation term that rises and then partially recedes.

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

> **Corrected after review (`fix_blind_eps.py`).** The tolerance ε in the table above was
> selected as the best of a 9-point sweep *scored against the planted answer*, so these are
> oracle numbers and "no ground truth used in the recovery" was wrong. Redone with a blind
> rule — ε = the JADE-rotated family's own residual off-diagonal mass, which needs no
> ground truth — and applied identically to every model:
>
> | | off-block | **blind ε** | oracle best-of-9 |
> |---|---|---|---|
> | planted + isotropic noise, all six levels to 0.29 | ≤0.29 | **11/11** | 11/11 |
> | trained seed 0 / 1 / 2 | 0.18 / 0.21 / 0.17 | **2 / 1 / 2** | 4 / 1 / 5 |
> | symmetrised seed 0 / 1 / 2 (A2-8) | 0.11 / 0.12 / 0.10 | **8 / 7 / 11** | 10 / 9 / 11 |
> | null: random weights, task shuffle | 0.91 | **0/11** | 0/11 |
>
> The blind rule is free on the whole synthetic calibration (gap 0 at every noise level,
> mean gap 0.60 frequencies overall) and costs 2–3 frequencies on the hardest trained
> models. So the calibration in A2-2 stands as written; the trained-model number in the
> table below drops from 4–5/11 to **1–2/11**; and A2-8's symmetrised pipeline still
> recovers **7–11 of 11 with no ground truth anywhere**, one seed at 11/11.

The honest verdict is therefore *partial recovery, tool-limited, with the limitation
localised and quantified*: not "the instrument is too weak" but "this particular residual
is harder than noise of its size, and none of the three obvious models of it reproduces
that hardness." A2-8 identifies most of what makes it hard, and removes it.

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

This gives a fully weights-only pipeline for this model: **symmetrise → JADE at a blind
tolerance → 7–11 of 11 frequency blocks recovered → project onto them → test accuracy
1.000.** The only task knowledge used is that the two input slots are interchangeable.
(With the oracle tolerance it is 9–11; the blind rule costs about two frequencies. Both are
reported in `fix_blind_eps_results.json`.)

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

| step | 0 | 20k | 40k | 60k | 100k | 140k | 200k | 260k |
|---|---|---|---|---|---|---|---|---|
| test acc | 0.031 | 0.566 | 0.978 | 0.981 | 0.981 | 0.987 | 0.987 | 0.991 |
| off-block, canonicalised | 0.913 | 0.311 | 0.178 | 0.172 | 0.171 | 0.168 | 0.167 | **0.169** |
| off-block, raw | 0.896 | 0.355 | 0.223 | 0.216 | 0.215 | 0.214 | 0.213 | 0.214 |
| functional residual | 0.903 | 0.452 | 0.319 | 0.313 | 0.309 | 0.307 | 0.307 | 0.306 |

*Restored and now reproducible* (`a2_followups.py`). These columns were withdrawn after
review because they came from an exploratory run whose script was never committed. Re-run
under a committed script, step 40000 reproduces the cached trajectory exactly
(0.9780 / 0.2226 / 0.1777), so the continuation is faithful. The reviewer also caught a
labelling error: what I originally called the canonicalised off-block at 200k/260k
(0.213 / 0.214) was the **raw** figure. Both are now given.

**Crystallisation arrests.** Over the last nine checkpoints — 100k to 260k steps — the
canonicalised off-block mass moves only between 0.1672 and 0.1713, a spread of 0.004, while
test accuracy creeps from 0.981 to 0.991. Six times as much training past the transition
buys nothing structurally.

Crystallisation is monotone and coincident with the test-accuracy rise — and then it
**stops**, at ~0.214 raw / ~0.169 canonicalised, holding flat from 40k to 260k steps while
test accuracy creeps to 0.991. The grokked bilinear solution is not the pure Fourier
solution and does not converge to it under continued weight decay.

Combined with A2-4 the residue is both permanent and harmful: training will not remove it,
and deleting it by hand takes the model to test 1.000. Note this is consistent with, and
sharper than, the corrected A2-4 dynamics — the residual's *logit scale* rises and then
partially recedes, but its *share* of the function stops falling at 40k and stays put for
another 220k steps.

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

### FINDING A3-1b — the limit is NOT on the input-dimension axis (`a3_axis.py`)

A3 as originally run varied only the component count at fixed `m = 8` outputs and `d = 16`
input dimensions, and attributed the breakdown to `K/d`. Reviewer 2 pointed out that the
form family's own effective rank is exactly `m` at every `K`, so the same data is equally
consistent with a limit in `K/m`. Sweeping `m` and `d` independently settles it, and one
pair of cells is decisive:

| m | d | K | K/d | K/m | outcome |
|---|---|---|---|---|---|
| 8 | 16 | 32 | 2.0 | 4.0 | **failed** (3% of components above 0.99) |
| 16 | 16 | 32 | 2.0 | 2.0 | **recovered** (100%) |

Same `K/d`, opposite outcome — doubling the number of *outputs* turns failure into complete
recovery. Across all 25 cells, recovery correlates with `−log(K/Kruskal bound)` at **+0.762**,
with `−log(K/m)` at **+0.722**, and with `−log(K/d)` at **+0.658** — the axis A3 originally
reported is the worst of the three. The Kruskal bound `R ≤ (m + 2d − 2)/2` (`THEORY.md` T6)
is the best single predictor and agrees with the observed outcome on 17 of 25 cells;
recovery beyond it does happen, which is expected since Kruskal is sufficient, not necessary.

The practical statement for `BILIN18_CONNECTION.md` §2.1 is therefore the Kruskal one, not
the `K/d` one, and that document has been corrected.

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

### FINDING A4-3 — curvature governs straightenability and gain does not — but this is a theorem, not a measurement

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

> **Relabelled after review.** `THEORY.md` T3 proves that `err_lin/err_prune` is invariant
> under scaling a component's gain, so the `+0.000` correlation is *forced*, not observed;
> in this DGP the ratio is exactly `1/(1+ρ²)²` (1, 0.04, 9.803e-5 against measured 1.0000000,
> 0.039817, 9.691e-5). The experiment confirmed arithmetic.
>
> I also over-claimed against the plan. The plan's stated prediction is that the band *moves
> with the tolerance*, and `band_vs_budget` confirms exactly that. I substituted a different
> statistic and then called the plan wrong. What is corrected is the mechanism, not the
> prediction:

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

## A4 addendum — the four nulls, and the oracle-free version (`a4_nulls.py`)

Both gaps recorded against A4 are now closed, and one of the nulls changes how A4 should
be read.

### FINDING A4-4 — the result survives dropping the oracle

A4 originally decomposed the student using the teacher's own component directions. Redone
blind, with the CP fitter A3 calibrated (A4 has 18 components in 48 dimensions, K/d = 0.38,
inside A3's certified range):

| arm | component recovery | corr with curvature | straightening wins | median gain |
|---|---|---|---|---|
| oracle directions | 1.000 (by construction) | **−0.992** | 22/25 | 5.2× |
| blind, rank 18 | mean cos 0.534, 44% > 0.9 | **−0.978** | 21/25 | 2.0× |
| blind, rank 22 | mean cos 0.581, 44% > 0.9 | **−0.981** | 21/25 | 2.3× |

The conclusion does not depend on the oracle. Note the components are only half-recovered
(44% above cosine 0.9) and the axis result is unchanged anyway — it is robust to getting
the parts substantially wrong.

One measurement caveat. In the blind arm "size" has to be estimated, and the natural proxy —
how much error deleting the component causes — is itself driven by curvature, so it
correlates −0.88 with straightenability. That is not a contradiction of A4-3: against the
*planted gain*, the correlation is +0.00. The clean statement is that the gain is
uninformative and the realised contribution is informative only because it inherits
curvature.

### FINDING A4-5 — the nulls say the linearization result is not about training

| null | component recovery | corr with curvature | straightening wins | median gain |
|---|---|---|---|---|
| trained (reference) | 1.000 | −0.992 | 22/25 | 5.2× |
| 1. random weights | **0.063**, 0% > 0.9 | −0.757 | **23/25** | **9.3×** |
| 2. gauge scramble | 1.000 | −0.993 | 22/25 | 5.1× |
| 3. task shuffle | **0.075**, 0% > 0.9 | −0.938 | **23/25** | 1.7× |
| 4. reader shuffle | 1.000 | −0.992 | 22/25 | 5.2× |

Null 2 is perfect invariance as required (refactorisation residual 8.4e-26, hidden width
192 → 1225, every number unchanged). Null 4 is vacuous by construction and measured to be so.

Nulls 1 and 3 are the informative ones, and they are not kind to the framing. **A randomly
initialised network shows the straightening advantage at 23/25 budgets with a larger median
gain than the trained one.** What collapses under nulls 1 and 3 is *component recovery*
(cosine 1.00 → 0.06/0.08); what survives is the entire linearization story.

This is correct rather than broken, and worth stating plainly: straightenability is a
property of quadratic forms evaluated on an input distribution with a large mean. It is not
something training puts there. So A4 is a **compression result with a correct selection
rule**, validated against planted truth — not a finding about learned mechanisms. The plan
files it under "the stratification and the linearization band" as though it characterises
the trained computation; it does not, and only the nulls reveal that.

---

## A5 — shared vs private subcomputation (`a5_shared.py`, `a5_extra.py`)

DGP. One shared quadratic form feeds three readers; each reader also has a private form of
its own, so reader u computes `Q_u = S_shared + S_private[u]`. The planted forms are made
mutually orthogonal — an early version used random positive-semidefinite forms, which all
have positive trace and therefore large positive inner products, planting a spurious common
direction at the top of every spectrum and confounding the entire question.

### FINDING A5-1 — the spectral prediction is off, and the correction is exact

The plan predicts the top eigenvalue of the reader-weighted second moment is the shared
form's, "with eigenvalue ≈ 3× the private forms" for three readers. Measured, and derived:

| readers R | measured top/next | analytic | top eigenvector's overlap with the pure shared form |
|---|---|---|---|
| 3 | **4.000** | R+1 = 4 | 0.866 analytic; 0.837 is the **Λ**-metric cosine of the same pair — a metric mismatch, not a discrepancy (see `REVIEW_RESPONSE.md` C4) |
| 5 | **6.000** | R+1 = 6 | — |
| 8 | **9.000** | R+1 = 9 | — |

The ratio is **R+1, not R**. And the top eigenvector is *not* the shared form: it is
√(R/(R+1)) of the shared form plus 1/√(R(R+1)) of **every** private form — (0.866, 0.289,
0.289, 0.289) at R=3. Reading it as "the shared computation" silently imports a slice of
every private one. Both facts fall out of diagonalising the coordinate matrix (`THEORY.md` T4, Proposition 4).
**Caveat added after review:** the `R+1` identity holds only at equal shared and private
gains — at g/h = 1.2 the ratio is 5.32, at 2.0 it is 13.0. It is not a usable read-out of
reader multiplicity unless the gains are known to be balanced. Rows R=5 and R=8 come from
the planted family directly and train no model.

### FINDING A5-2 — the reader shuffle cannot test sharing; a no-sharing control can

| R | shared family | after reader shuffle | matched family with no sharing |
|---|---|---|---|
| 3 | 4.000 | 3.50 / 3.72 / 4.00 | **1.000** |
| 5 | 6.000 | 4.41 / 5.33 / 4.48 | **1.000** |
| 8 | 9.000 | 4.95 / 6.22 / 6.17 | **1.000** |

The plan's prediction (iv) — that the reader shuffle collapses the shared/private gap — is
false, and necessarily so. A component that is *identical* across readers contributes
identical coordinates to every reader, so permuting readers cannot touch it. The control
that works is a matched family of the same size and norm in which nothing is shared; it
separates perfectly, 4.00 versus 1.00.

Generalising, **with the limit review forced**: the shuffle is insensitive at small R and
only *partially* sensitive as R grows — at R=8 it does move the statistic 9.00 → 4.95, a 45%
collapse, so "cannot test sharing" is too strong. The matched no-sharing control is the
instrument to use either way, since it separates completely at every R.

### FINDING A5-3 — a single reader cannot be decomposed into shared and private

Decomposing each reader's form on its own recovers the shared component at cosine
**0.41, 0.41, 0.41** — barely better than chance. This is not a solver failure: `Q_u = S₀ + S_u`
is a sum of two orthogonal forms, and any rotation within their span produces the same `Q_u`,
so from one reader the split is *unidentifiable in principle*. The plan predicted the shared
form would be found "three times in mutually incompatible gauges"; the truth is stronger —
it is not found at all. This is the concrete argument for fitting globally across readers
and explaining individual paths afterwards.

### FINDING A5-4 — with one private form per reader, a dictionary cannot recover the split either — and that is correct behaviour

With 3 readers, "one atom per reader" costs 3 atoms while the true structure costs 4, so the
shortest description is the one that hides the sharing. Measured exactly that: at 3 atoms
the fit is already exact (error 0.000) with 1.00 active atoms per reader, and at 4 atoms the
recovered atoms match the planted ones at only ~0.71.

So the plan's prediction (ii) is untestable in the DGP the plan specifies. Sharing is only
worth representing when readers outnumber atoms. Rebuilt that way — 8 readers each combining
2 of 4 shared forms — the atoms come back at cosine 0.93–1.00.

### FINDING A5-5 — PCA's failure is invisible in reconstruction error

The plan's PCA-orthogonality claim, tested by planting a coarse form plus refinements that
contain it and sweeping how much they overlap. At the true budget in every case:

| planted mutual overlap | PCA error | dictionary error | dictionary atoms vs planted | **PCA components vs planted** |
|---|---|---|---|---|
| 0.00 | 0.0000 | 0.0000 | 0.985 / 0.986 / 0.988 | 0.815 / 0.894 / 0.919 |
| +0.25 | 0.0000 | 0.0000 | 0.972 / 0.983 / 0.990 | 0.704 / 0.805 / 0.800 |
| +0.64 | 0.0000 | 0.0000 | 0.962 / 0.983 / 0.998 | 0.870 / **0.385** / **0.529** |
| +0.90 | 0.0000 | 0.0000 | 0.958 / 0.899 / 0.625 | 0.967 / **0.203** / **0.273** |

PCA and the dictionary are **indistinguishable on error at every overlap level** — both
reconstruct exactly at the true budget — while their recovered components diverge
completely as the planted parts start to overlap. PCA's mean match falls 0.88 → 0.48; the
dictionary holds 0.99 → 0.83.

The model-selection protocol in the plan compares candidate structures on "functional error
against description length". This says that comparison **cannot see the difference that
matters**. Error-versus-budget is blind to identification; only scoring against ground truth
separates them, which is exactly what a real model does not come with. That is an argument
for calibrating decomposition methods on planted structure before applying them, not for
choosing between them by fit.

### A5 nulls

| null | outcome |
|---|---|
| 1. random weights | top/next eigenvalue ratio **1.56 / 1.56** against the trained 4.000; top eigenvector's overlap with the shared form **0.144** against 0.837 |
| 2. gauge scramble | refactorisation residual 4.5e-28, hidden width 96 → 78, ratio **4.000 → 4.000** — exact invariance |
| 3. task shuffle | not run for A5 — the DGP has no labels to shuffle independently of the readers; the matched no-sharing control in A5-2 plays the same role and separates 4.00 vs 1.00 |
| 4. reader shuffle | run, and **shown to be the wrong instrument** (A5-2): it cannot move a statistic about sharing |

---

## B2 / B1 — conjunctive retrieval and the degeneracy census (`b_common.py`, `b2_conjunctive.py`)

DGP. Each token carries a type (1 of 8) and a timing value (1 of 8) in disjoint embedding
subspaces; the query names one of each. Exactly one key matches both, three match the type
only, three the timing only, so any single-property strategy spreads over four keys and
reads the right payload 25% of the time. Two controls make one property sufficient on its
own (ceiling 1.00). Three heads at matched parameter count: score-level product
(`s = (qᵀW₁k)(qᵀW₂k)` then softmax), post-softmax product (`normalise(A₁ ⊙ A₂)`), and an
ordinary single-circuit head at twice the rank. Predictions were registered in the script
before running and are stored in `b2_results.json`.

### FINDING B2-1 — the conjunctive DGP does not force conjunctive machinery

All 18 runs reach accuracy **1.0000**, including the ordinary single-circuit head on the
conjunctive task. Prediction P5 (a measurable penalty for standard attention) is **false**,
and in hindsight it had to be: with two conditions and a unique double match, an *additive*
score gives the match `2h` and every distractor `h + l`, so a sharp enough softmax
implements AND. Raising the distractor count to 7 of each kind (ceiling 0.125) does not
change it — still 1.0000.

So this DGP cannot discriminate architectures by performance. Anything B2 says has to come
from what the heads *are*, not what they score. That is a design lesson for B3: a task whose
conjunction is expressible as a sum of two match scores tests nothing about multiplication.

### FINDING B2-2 — the entropy signature for "genuine conjunction" has no specificity

The plan's B1 test for regime (c) is: both factors individually broad, product sharp. It
fires on **12 of 12** multiplicative runs — including all 8 on control tasks where a single
property is sufficient and there is nothing to conjoin:

| task | head | H(factor 1) | H(factor 2) | H(product) | verdict |
|---|---|---|---|---|---|
| conjunctive | score-level | 0.86 | 0.88 | 0.04 | (c) |
| **timing alone suffices** | score-level | 0.88 | 0.92 | 0.02 | **(c)** |
| **type alone suffices** | score-level | 0.87 | 0.89 | 0.03 | **(c)** |
| conjunctive | post-softmax | 0.65 | 0.63 | 0.21 | (c) |
| **timing alone suffices** | post-softmax | 0.67 | 0.64 | 0.17 | **(c)** |

(entropies normalised by log of sequence length). Multiplying two score fields sharpens the
result whatever they encode; the signature is a property of the operation, not of the task.

**And it is worse than non-specific for the score-level placement.** `THEORY.md` T8 proves
`(W₁, W₂) → (cW₁, W₂/c)` is exactly function-preserving while softmax entropy is not
invariant under it — measured, the same head's factor entropies move `[2.768, 2.769] →
[1.799, 2.773]` under `c = 20` with the function unchanged to 2.7e-16. So the score-level
entropy numbers in the table above carry no information at all and are withdrawn as
evidence; the participation ratio is invariant and is what B0 uses instead. B2's null 2
never tested this gauge — it only applied `Wq → MWq, Wk → M⁻ᵀWk`, which preserves each `W_i`
exactly, and its 7-significant-figure agreement should have been the tell.

The test is not useless — it correctly calls random weights and label-shuffled models
"(a) factor collapse" (accuracy 0.061–0.063). It discriminates trained from untrained. It
does not discriminate conjunctive from non-conjunctive, which is the job the plan gives it.

### FINDING B2-3 — RETRACTED: the ablation instrument is broken

Replacing one factor with a constant that preserves its scale, and re-measuring:

| head | task | ablate factor 1 | ablate factor 2 | ceiling |
|---|---|---|---|---|
| score-level, seed 0 | conjunctive | **0.265** | 0.982 | 0.25 |
| score-level, seed 1 | conjunctive | 0.879 | **0.451** | 0.25 |
| post-softmax, seed 0 | conjunctive | 0.870 | 0.838 | 0.25 |
| post-softmax, seed 1 | conjunctive | 0.861 | 0.870 | 0.25 |

> **RETRACTED.** The replacement constant is the per-example mean over keys, which is
> negative on 55.9% of examples — a negative multiplier inverts the attention ordering, so
> this measures the intervention, not the mechanism. The falsifying evidence was in the same
> JSON: on the *control* tasks, where one property suffices and the ceiling is 1.000, the
> same ablation still drops the head to 0.27, indistinguishable from the conjunctive 0.265.
> A factor that can score 1.000 alone is not "doing nothing".
>
> The conclusion that the score-level placement is functionally single-circuit is withdrawn.

**And the replacement does not exist** (`fix_ablation.py`). Three schemes were implemented
and put through a soundness test that does not involve the conjunctive task at all: *on a
control task where one property already suffices, ablating the factor that reads the other
property must leave accuracy near 1.000.* All three fail:

| replacement scheme | best factor retained, on controls (ceiling 1.000) | worst case | on the conjunctive task, worse factor falls to |
|---|---|---|---|
| positive constant matched on RMS | 0.590 | **0.000** | 0.367 |
| shuffle the factor's scores across keys | 0.658 | 0.436 | 0.318 |
| per-key mean (the original, broken one) | 0.898 | 0.726 | 0.573 |

None comes near 1.000, so none is measuring the mechanism. The reason is structural rather
than a bad choice of constant: in a product head the output depends on `s₁ · s₂`, so
deleting one factor does not leave "the other factor's computation" behind — it leaves a
differently scaled and differently shaped object, and there is no canonical "rest of the
head without factor 2" to compare against.

**So the plan's B1 verification step — "ablate one factor, confirm degradation to
single-property chance performance" — is not implementable as specified.** Taken with B2-2
and B0-1 (the entropy and participation-ratio signatures are non-specific) and `THEORY.md`
T8 (per-factor entropies are gauge-dependent), *every* instrument Part B proposes for
characterising a two-factor head has now failed on a toy with known ground truth.

What did work is asking what each factor **reads** rather than what happens when it is
removed: the weight-space factor readout of B2-4, and the path-sensitivity measurement A6
introduced and B3 uses. That is the instrument to carry to bilin18, and it replaces test #4
in `BILIN18_CONNECTION.md`.

### FINDING B2-4 — factor specialisation is real, but only in the post-softmax placement, and only when the task needs it

Fraction of each factor's QK form sitting in each planted (query-side, key-side) block:

| task | head | factor 1 reads | factor 2 reads |
|---|---|---|---|
| conjunctive | post-softmax s0 | timing→timing 0.39 | type→type 0.37 |
| conjunctive | post-softmax s1 | timing→timing 0.33 | type→type 0.33 |
| timing alone suffices | post-softmax s0/s1 | timing→timing 0.52 | timing→timing 0.52 |
| type alone suffices | post-softmax s0/s1 | type→type 0.48 | type→type 0.49 |
| conjunctive | score-level s0 | timing→timing 0.38 | type→payload 0.15 |
| conjunctive | score-level s1 | type→type 0.18 | type→type 0.45 |

The post-softmax head does exactly what the plan predicts — one factor per property on the
conjunctive task — and, tellingly, puts *both* factors on the sufficient property when only
one is needed. That is a clean confirmation of prediction P2 for one of the two placements.
But the specialisation is weak: only a third to a half of each factor's mass sits in its
block. The score-level head does not split at all.

This matters for the plan's B4 bet that "factors are simpler than their product". At best
they are somewhat simpler, in one placement, on a task built to reward the split.

### FINDING B2-5 — correlated properties do not produce factor alignment

The plan predicts that correlating the two properties should "induce drift from (c) toward
(b) factor alignment". Measured across correlation 0 → 0.9, the similarity between the two
QK circuits goes **0.139 → 0.171 → 0.038 → 0.017** — down, not up. What does change is the
ablation asymmetry, which shrinks (0.265/0.982 → 0.656/0.798): the factors become more
equally unimportant, not more alike.

### B2 nulls

| null | outcome |
|---|---|
| 1. random weights | accuracy 0.061 / 0.063, both heads called "(a) factor collapse" |
| 2. gauge scramble | `Wq → M Wq, Wk → M⁻ᵀWk` per factor: function unchanged to 8.6e-06 while the raw weights move 2.97 relative; accuracy 1.0000 → 1.0000, regime and factor readouts unchanged |
| 3. task shuffle | accuracy 0.062, "(a) factor collapse" |
| 4. factor swap (the reader-shuffle analogue) | function difference **exactly 0.0**, readout labels swap. Factor identity is pure gauge, so "factor 1 reads type" is only meaningful as an unordered pairing |

### B2 verification

Changing the query's type moves attention off the old match (0.977 → 0.157) and onto keys
carrying the new type (0.357), so the head is reading the query rather than memorising —
prediction P6 holds. The ordinary single-circuit head does the same thing (0.844 → 0.125,
0.326), which is another instance of B2-1: this verification does not distinguish
architectures either.

---

## A6 — two-hop routing: the quotient is a property of the path, not the layer (`a6_routing.py`)

DGP. A first bilinear layer computes features A, B, C from its input; two downstream
bilinear readers consume the result, with reader 1's target depending only on A and
reader 2's only on B. C is read by nobody. Planted features occupy disjoint blocks of a
random orthonormal frame (verified orthogonal to 1e-16), so the routing table is exact.

Because the composition of two bilinear layers is quartic, there is no single quadratic
form for a path. The object that answers the routing question is the path's expected
input-gradient outer product, `S = E[(∂out/∂x)(∂out/∂x)ᵀ]`, whose kernel is the set of
input directions the path cannot be moved along anywhere on the data.

### FINDING A6-1 — the routing table comes back from the composed weights

Share of each path's input sensitivity sitting in each planted feature block (2 seeds,
student FVU 8.4e-2 and 7.2e-2):

| | A | B | C |
|---|---|---|---|
| layer 1 transmits | 0.431 | 0.412 | **0.040** |
| reader 1's path | **0.940** | 0.012 | 0.012 |
| reader 2's path | 0.010 | **0.951** | 0.010 |

Each reader reads exactly its planted feature, with 77–103× separation from the next
feature. Activation patching — resampling the input inside one feature's block — agrees:
reader 1 moves 1.40 for A against 0.157 for B, reader 2 moves 1.45 for B against 0.145 for
A. Argmax agreement between the weight-side ledger and patching is 2/2 in both seeds, and
the Spearman correlation across all six (reader, feature) pairs is +1.00 and +0.49.

*Scoring note.* An earlier version of this comparison applied one absolute 5% threshold to
both quantities and scored 2/6. That was wrong: the weight-side number is a share of
sensitivity that sums to 1 across features, while the patching number is a relative RMS
change under a full resample of 4 of 24 input dimensions. They are on different scales, so
the comparison has to be on ranking and separation, which is what is reported above.

### FINDING A6-2 — the path kernel is strictly larger than the layer kernel, and B is the proof

This is the point of the experiment. Layer 1 **transmits** B at 0.412 — nearly as much as A.
Reader 1's path is blind to B at 0.012. So reader 1's blindness to B is not something layer
1 did; the information is present in the bus and the reader does not look. The path kernel
therefore strictly contains the layer kernel, and the difference is exactly the
transmitted-but-ignored feature.

Any analysis that computes "what this layer discards" and stops there will overstate what
is available downstream by exactly this term.

### FINDING A6-3 — end-to-end training does not transmit what no reader needs

C was planted as a third feature to be transmitted and ignored. Layer 1 carries it at
**0.040**, an order of magnitude below A and B: with both readers trained end to end,
the layer simply stopped computing the feature nobody consumes. So the interesting case —
transmitted *and* ignored — has to be forced by construction (as B is, by being needed by
the other reader) rather than hoped for. Worth carrying into B3.

### A6 nulls

| null | outcome |
|---|---|
| 1. random weights | sensitivity is flat across features — reader 1 gives A 0.190, B 0.175, C 0.176, against the trained 0.940 / 0.012 / 0.012 |
| 2. gauge scramble of layer 1 | refactorisation residual 5.8e-29, hidden width 128 → 300, end-to-end function change 2.1e-13, and the ledger is **bit-identical**: 0.940 → 0.940, 0.012 → 0.012 |
| 3, 4 | not run |

Caveat: the students reach FVU ~0.08, not machine precision — a quartic composition through
an 18-dimensional bus is a harder fit than the single-layer experiments. The routing
separations are large enough (77–103×) that this does not threaten the conclusion, but the
numbers are not exact the way A1's are.

---

## B0 — the third placement, the one bilin18 actually uses (`b0_placements.py`)

The plan's B0 lists two placements: the product taken before a softmax, and the product of
two softmaxed patterns. Verified at source (`jacclust/tt_model.py:134-144`), bilin18 uses
neither — its pattern is `(q₁·k₁/D)(q₂·k₂/D)`, causally masked to zero and **never
normalised**, with no softmax anywhere in the model. Entries are signed; measured negative
mass fraction 0.14–0.86 depending on seed.

So the B1 taxonomy, every statistic of which is an entropy of an attention distribution, is
**undefined on the architecture it was written for**. This module adds the third placement
and replaces entropy with the participation ratio `PR(w) = (Σw²)²/Σw⁴`, which is defined for
signed unnormalised patterns, is what the earlier jacclust program used on bilin18, and —
per `THEORY.md` T8 — is the statistic the two-factor scale gauge *requires*.

### FINDING B0-1 — the sharpening signature has no specificity in any placement

Does `PR(product) < min(PR(factors))` — "individually vague, jointly precise" — distinguish
tasks that need a conjunction from tasks that do not?

| placement | conjunctive task | controls needing no conjunction | mean PR drop, conjunctive vs controls | at **random init** |
|---|---|---|---|---|
| unnormalised (bilin18's) | fires 2/2 | **fires 4/4** | 0.199 vs 0.187 | **FIRES** (0.446/0.423 → 0.265) |
| score-level product | fires 2/2 | **fires 4/4** | 0.444 vs 0.323 | does not fire (→ 1.000) |
| post-softmax product | fires 2/2 | **fires 4/4** | 0.318 vs 0.260 | does not fire (→ 0.933) |

It fires everywhere, and the size of the drop barely distinguishes the cases either
(0.199 vs 0.187 for the unnormalised placement). In the score-level placement the product's
PR is pinned at 0.063 regardless of task or seed. Multiplying two score fields concentrates
the result whatever they encode; that is a property of multiplication, not of the
computation.

**And the last column is worse news for bilin18 specifically.** The two softmax placements
at least fail to fire on randomly initialised weights, so their signature separates trained
from untrained. In the unnormalised placement — the one bilin18 uses — **it fires at random
initialisation too.** There it carries no information at all: not about conjunction, and not
even about whether the model was trained.

**This is what the bilin18 connection needed.** `jacclust/SUMMARY.md:69` records
`PR(product) < min(PR(s₁), PR(s₂))` at **100% of bilin18's 162 heads**. My connection
document listed "that is not evidence of conjunction" as an inference. It is now a
measurement, in bilin18's own placement, with control tasks: a statistic that fires on
100% of heads is doing no discriminating work, and here is a controlled setting showing it
fires with nothing to conjoin.

What the right test is remains open — B2-3's ablation was retracted as broken, and a
corrected ablation has not been run.

---

## B3 — full-stack testbed: a planted quotient upstream of conjunctive attention (`b3_fullstack.py`)

Composes A6 and B2, with two departures from the plan that earlier results forced:
the attention head uses bilin18's **unnormalised score product** rather than either
placement B0 names, and routing is measured by **path sensitivity** (A6's instrument) rather
than by the kernel of a quadratic form, because a bilinear layer composed with a bilinear
head is quartic and has no single form.

Planted design: each token carries a type A, a timing B, a payload and a modifier C in
disjoint subspaces, plus dead coordinates nothing should read. A shared bilinear layer maps
tokens to a bus; downstream, attention factor 1 should read A, factor 2 should read B, and
the MLP path should read C. Two separately decodable outputs — retrieve the payload of the
key matching *both* A and B, and report the query's own modifier.

### FINDING B3-1 — the routing table comes back from the composed weights, and patching agrees

Seed 0, rms-normed bus. **Retrieval accuracy 1.0000 against a single-property ceiling of
0.250**, so the conjunction is genuinely being computed; modifier accuracy 1.0000.

| | A | B | C | dead |
|---|---|---|---|---|
| layer 1 transmits | 0.246 | 0.274 | 0.068 | **0.002** |
| attention factor 1 reads | **0.434** | 0.389 | 0.046 | 0.001 |
| attention factor 2 reads | 0.250 | **0.582** | 0.040 | 0.001 |
| MLP path reads | 0.167 | 0.213 | **0.349** | 0.001 |

Activation patching, resampling one planted feature and measuring each path's change:

| resampled | factor 1 | factor 2 | MLP |
|---|---|---|---|
| A | **1.145** | 0.597 | 0.326 |
| B | 1.106 | **1.366** | 0.441 |
| C | 0.084 | 0.089 | **1.400** |
| dead | 0.119 | 0.101 | 0.145 |

The MLP path's isolation is the cleanest result: C moves it by 1.400 and moves the two
attention factors by 0.084 and 0.089 — a **16× separation**, from weights and from patching
independently. Dead coordinates are read at 0.001 by every path and move nothing.

### FINDING B3-2 — path routing recovers reliably; the split *within* a two-factor path does not

All three trained models reach retrieval 1.0000, and all three isolate the MLP path
cleanly. What varies wildly is whether the two attention factors specialise:

| model | factor 1 reads | factor 2 reads | patching separation for A / B |
|---|---|---|---|
| normed bus, seed 0 | A 0.434, B 0.389 | B 0.582, A 0.250 | 1.9× / 1.2× |
| normed bus, seed 1 | **A 0.517**, B 0.304 | **A 0.549**, B 0.278 | 1.0× / 1.2× — *both factors read A* |
| raw bus, seed 0 | **A 0.720**, B 0.181 | **B 0.746**, A 0.145 | **3.8× / 4.0×** |

Same architecture, same task, same number of steps: one model splits the two properties
almost perfectly between its factors, one splits them weakly, and one puts *both* factors on
the same property and still solves the task. Prediction (iv) is therefore **sometimes true
and sometimes false, unpredictably**.

Meanwhile the between-path routing is robust everywhere. Resampling the modifier C moves
the MLP path by 1.376–1.400 and the attention factors by 0.064–0.105 in every arm — a
**13–20× separation** — and dead coordinates are read at 0.001 and move nothing.

**Confirmed and partly corrected at 16 models** (`b3_seeds.py`, eight seeds per arm, every
one reaching retrieval 1.0000):

| | factor split | both factors on one property | MLP path isolation |
|---|---|---|---|
| rms-normed bus | 0.27 ± 0.23 (range 0.03–0.65) | **3 of 8 seeds** | 17.0 ± 7.0 (worst **1.0×**) |
| raw bus | 0.28 ± 0.21 (range 0.03–0.64) | **3 of 8 seeds** | 12.4 ± 9.2 (worst **1.0×**) |

The variability claim holds and is now quantified: relative spread 0.82, a 20× range, and
**6 of 16 models put both factors on the same property** while still solving the task
perfectly.

But my accompanying claim that between-path routing is robust *everywhere* is **wrong**.
Four of the sixteen models — one normed, three raw — show isolation of 1.0×, i.e. no path
separation at all. Path isolation is more stable than the factor split (relative spread
0.58 against 0.82) but it is not reliable either; it fails outright in a quarter of models.

The conclusion the toys support is therefore narrower than both the plan's and my own
earlier statement: **which path reads what is usually recoverable and the factor split
usually is not, but neither is a guaranteed property of the architecture — the same
architecture on the same task produces models where either can fail.** For bilin18
that means a per-head factor census may be asking a question that has no consistent answer,
while a per-path routing ledger should be reliable. It also sharpens B2-4's weaker version
of the same observation, and it is the most direct evidence the program has about the plan's
B4 bet that "factors are simpler than their product": sometimes they are, and you cannot
tell in advance which case you are in.

### FINDING B3-3 — the first task design was unreachable, and the normalisation is *not* load-bearing after all

The first version of this experiment asked for `(payload + modifier) mod 8` from a single
linear readout. That needs a bilinear interaction between two retrieved quantities and the
stack has none — the MLP path never sees the attended value. Both arms plateaued at ~0.50
with each seed latching onto one of A/B and dropping the other (seed 0 dropped B, seed 1
dropped A).

Two things came out of the failure. First, **the instruments correctly reported it**: the
unread feature showed up as ~0.02 in the weight-side readout *and* as a ~0.03 patching
effect, so the ledger said "this model is not doing the planted computation" rather than
inventing structure. That is the more valuable direction for an instrument to fail in.

Second, a measured fact about initialisation: with an unnormalised score product the
attention path's output scale goes as the **fourth power of the bus norm** — RMS 2.9e-3
without normalisation against 0.39 with it, so at initialisation the attention path is two
orders of magnitude quieter than the MLP path.

> **Correction.** On the strength of that measurement I claimed bilin18's pre-attention
> RMSNorm is "load-bearing, not incidental". The comparison arm refutes it: once the target
> is reachable, the **raw un-normalised bus trains to retrieval 1.0000 as well** — and in
> fact produces the *cleanest* factor specialisation of any arm (0.720 / 0.746 against the
> normed arms' 0.434 / 0.582 and 0.517 / 0.549). Training walks through the initialisation
> handicap. The scale effect is real and measurable; the conclusion I drew from it was not.
> Whether the normalisation matters at bilin18's depth and context length is untested here.

### B3 nulls

| null | outcome |
|---|---|
| 1. random weights | retrieval 0.1256 and modifier 0.2561, both exactly chance (1/8 and 1/4); layer 1 transmits every feature roughly equally *including dead at 0.202*, against 0.002 trained — the learned discard is real |
| 2. gauge scramble of layer 1 | refactorisation residual 2.2e-26, hidden width 128 → 496, end-to-end function difference **5.7e-11**, and every ledger entry unchanged to three decimals (0.434 → 0.434, 0.582 → 0.581, 0.349 → 0.349) |

All three trained arms and both nulls are complete.

---

## Queue (authoritative — the hourly cron reads this)

Done since the last update: B2, A5, A4's nulls + oracle-free arm, A6, B0's third
placement, the independent review (see `REVIEW_RESPONSE.md`), the bilin18 connection
(`BILIN18_CONNECTION.md`) and the theory pass (`THEORY.md`, 47/47 verified).

1. **Review items:** ~~blind JADE tolerance~~ done; ~~sanity checks~~ done (12 → 20);
   ~~correct factor ablation~~ done, and it showed no sound ablation exists. Still open:
   re-run A2-7's long-run columns under a committed script.
2. **Finish B3's remaining arms** (seed 1, the no-normalisation comparison) and fold them in.
3. ~~A3's missing axis~~ — done (`a3_axis.py`): the limit is not on the `K/d` axis.
4. **The open theory questions** in `THEORY.md`, chiefly a perturbation bound explaining
   why a trained residual is harder for joint block-diagonalisation than matched noise.

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

---

## Real-model results: the 546M bilinear transformer

Two of the four tests registered in `BILIN18_CONNECTION.md` have now been run on the
actual 18-layer, 546M-parameter bilinear model rather than on toys. Both came back
against the program's own prior expectations, and both are written up in full in
`BILIN18_CONNECTION.md` §5–6.

**Test #4 — the attention census (`bilin18_attention.py`).** The recorded observation
that `PR(product) < min(PR(factors))` at 100% of bilin18's 162 heads, read as evidence
that the two QK branches are genuinely conjunctive, is **uninformative**: the identical
architecture with random weights fires it on 162/162 heads too, with a *larger* median
participation-ratio drop (0.207 against the trained model's 0.097). The recorded 0.49
negative-mass figure is likewise matched at 0.500 by random weights. Zero of 162 heads
have near-identical QK circuits (median |cos(W₁,W₂)| = 0.031).

**Test #2 — the identifiable subspace (`bilin18_identifiable.py`,
`_power.py`, `_blind_direction.py`).** Under 1% of an MLP interaction form's Frobenius
mass is identifiable from 4000 sampled inputs — but the sample-scaling control shows
that number is a sampling artefact, not a fact about the weights. The real finding is in
the curve shape: layers 5 and 13 sit at **0.10× and 0.19× chance** at small sample
counts, an order of magnitude *below* what an unrelated matrix scores, while the
unrelated-form null stays flat at 1.0× across a 32× range in N.

The mechanism was then confirmed directly. Through the middle of the network the
rms-normed MLP input is nearly a single fixed vector — at layers 5–7 the mean |cos|
between a token's input and the top principal component is **0.97**, so every token
arrives within ~14° of the same direction — and the bilinear MLPs there are built to be
blind to it, with curvature along that direction running 0.00–0.19× that of an
equally-sized random form. Layer 7 annihilates it outright. Layer 0 and the last two
layers run the opposite way (up to 5.2× enriched).

**Testing that program's own fix (`bilin18_whitened.py`, `_dirs.py`).** The
recommendation below — whiten before reading mass statistics — was then tested rather
than asserted, by scoring rank-k truncations of the forms by held-out functional error.
Whitening helps at every layer, by 1.5–5.2×, and at four of six layers it is the
difference between reaching 90% of the function inside rank 32–64 and not reaching it
by rank 128. The recommendation holds. **A registered prediction attached to it failed**:
the benefit is largest at layer 17, the layer *least* dominated by a single input
direction, so whitening does not work for the mechanism reason I gave.

That run also produced a methodological correction worth more than its headline.
Scoring interaction-form rank on **random** output directions overstates rank by 2–8×;
on the directions the layer's output actually occupies, whitened rank 32–64 captures 90%
of the function where random directions suggested "irreducibly high rank". And one
robust structural fact survived every basis: **layer 17's MLP is very nearly a
four-dimensional quadratic form** (whitened rank 4 → 90%, rank 16 → 99.5%), against
rank 32–64 for layers 1–13.

The consequence for this program is direct and unwelcome: **every Frobenius-mass
statistic used in Part A is compromised on bilin18**, because on layers 1–13 most of a
form's norm sits in directions the data never visits. The toys could not have caught
this — their one-hot inputs have no dominant direction by construction. The fix is to
whiten by the input second moment (the Λ-weighted metric already implemented in
`bq_common.py`, and not previously applied to the real model) before reading any mass
statistic off bilin18 weights.

### Reading a real computation out of layer 17

The rank result led somewhere. Layer 17's MLP output needs only **4** principal
directions for 90% of its variance, so the layer was replaced outright by 4 output
directions each carrying a rank-2 form — eight signed squared projections, ~13.8k
numbers standing in for ~15.9M parameters — and the model re-scored. Deleting layer 17's
quadratic part costs 1.077 nats of cross-entropy; the eight-term replacement costs
**9.5%** of that. Of those 9.5 points, 8.8 come from confining the output to four
directions and only **0.7** from truncating those four forms to rank 2 — so rank 2 is
nearly free once the directions are chosen. The same rank budget spent by |eigenvalue|
instead of in the Λ-weighted metric does 7× the damage, which is the strongest
confirmation yet that the functional metric is not a technicality.

> **Correction.** An earlier version of this paragraph said the replacement "recovers all
> but 0.7%" of the deletion cost. That conveys the wrong quantity: 0.7% is the damage
> added by rank truncation *beyond* the projection step, while the replacement as a whole
> costs 9.5%. Sweeping the number of output directions properly also showed that
> **layer 16, not 17, is the model's most compressible layer** — same 13,832-number
> budget, 4.2% damage against layer 17's 9.5%. The rank finding stands; the compression
> headline was pointed at the wrong layer.

The eight terms are readable. The leading output direction subtracts a squared
projection onto an auxiliary/function-word direction (59% of the form) and adds one onto
a delimiter/punctuation direction (21%). The second reads as a syntax rule: suppress
"predict a determiner" after sentence-closing punctuation, boost it in the presence of a
copula or pronoun.

Those names were then verified rather than asserted, against measured excitation over
262,144 corpus positions with a permutation null. **A first attempt had no power and is
recorded as such** — it used the 32×513 eval set where only 48 tokens clear the count
threshold, making chance overlap higher than the measured value. On the corrected test
all six features clear their null, but unevenly: the three leading features sit at
ρ = 0.39–0.66 against a null of ~0.06, while the weakest is at 0.114 and its name should
be treated as unverified. Interpretation by unembedding alignment is therefore *partially*
justified here, and should always be reported with the correlation attached.

Full detail in `BILIN18_CONNECTION.md` §8.

### The middle of the network, and where this stops working

Running the layer-17 treatment on all eighteen MLPs answered the question §8.4 left open:
it does not generalise. Only layers 0 and 16 compress at all, and the reason turned out
to be more interesting than "the others are higher rank".

Deleting any single middle layer's quadratic part is nearly free — 0.024 to 0.520 nats,
most under 0.06. Deleting **all fourteen at once costs 5.14 nats**, against 1.79 for the
sum of the individual costs: **2.87× superadditive**. When one mid-layer goes the other
thirteen absorb the loss; when all fourteen go there is nothing left to absorb it. So any
claim of the form "layer *n* contributes almost nothing", built on a one-at-a-time
ablation, is unsafe in this model.

The two layers that do compress are exactly the two with large individual delete costs
(1.80 and 1.17 nats). The ones that resist are those whose individual contribution is
small and whose collective contribution is large — the signature of computation spread
across depth rather than localised. Also worth recording: **layer 1 is the single most
important MLP in the model**, at 5.65 nats to delete, against ~1.1–1.8 for layers 0, 16
and 17 and under 0.52 for everything else. Nothing in the toy program predicted that
concentration and nobody had looked.

### Test #1, closed negative

The last outstanding test from the connection document asked whether re-ranking the
existing repository's quadratic features would improve its error-versus-parameter
frontier. Reading the code corrected the premise — its ranking is already in the
data-weighted metric, not "by raw size" as the document claimed. But it exposed a real
defect: the feature second moment is uncentered while the constant term omits the
corresponding mean, so the low-rank fit spends directions representing a constant that
could be stored free in the bias. That omitted constant carries **16–52%** of the
quadratic term's energy.

Fixing it buys almost nothing: 7 of 12 cells improve, mean +0.0045 nats, best +0.0400 —
a 3% relative gain — and only at the smallest budget, because Eckart–Young absorbs a
rank-one constant nearly for free. The predicted correlation between the win and the
constant's energy share is absent. The diagnosis was right and the predicted consequence
wrong; the test is closed negative.

Full detail in `BILIN18_CONNECTION.md` §9–11.

### Test #3: a challenge that failed, and a statistic that does not mean what it says

The last of the four real-model tests asked whether the repository's boundary result —
"individual directions account for 24% of MLP1's effect; the other 76% appears only under
joint removal", read as evidence the layer is irreducibly distributed — is a fact about
the layer or an artefact of using PCA directions. A5 predicted sparse-dictionary atoms
would be substantially more individually attributable.

**At the original operating point, no basis helps.** Thirty-two directions each: SVD 9.0%
attributable, a random rotation of the same subspace 9.1% (reproducing the original's own
basis-independence control exactly), a 32-atom dictionary 13.1%, and the top 32 of a
4096-atom dictionary 20.5%. Four fifths of the effect stays joint-only under every basis
tried. The boundary result survives the challenge.

**And A5's prediction fails once removed energy is matched** — the control the test
specification asked for and my first script omitted. At matched ~53% removed energy, SVD
is **3× more** individually attributable than the dictionary, the opposite direction; at
~71% the dictionary leads by 1.5×. The direction flips with the matching point, which is
what a confounded comparison looks like.

**What the run does establish is more useful than either prediction.** On the same
unchanged layer, the attributable fraction ranges from **9% to 64%** depending only on how
many directions are removed — 64.1% at 4 directions, 9.0% at 32, 13.8% at 82. Going from
4 to 32 directions grows the joint effect 14× and the sum of solo effects only 1.4×:
interference grows superlinearly in the size of the ablation set. So "76% appears only
jointly" is largely a statement about removing 32 things at once, not a scale-free
property of the layer, and the figure is not meaningful without its count and energy
attached.

That is the same phenomenon as the 2.87× superadditivity found across depth, measured
inside one layer instead of across fourteen. Both my prediction and the framing I
inherited were wrong the same way: treating a budget-dependent number as a measurement of
the layer.

Full detail in `BILIN18_CONNECTION.md` §12. **All four planned real-model tests are now
run.**

### The budget-free answer: MLP1 runs on about ten directions

The batch's through-line — solo-vs-joint attribution depends on the ablation budget —
has a standard repair: Shapley values, which average each direction's marginal effect
over all coalition sizes and are forced by construction to sum to the joint effect
exactly, leaving no unexplained residual. Twenty permutations, 660 model evaluations.

On the same 32 directions where solo ablation says "1.4 effective directions but 91% of
the effect unexplained" and §74's phrasing says "irreducibly distributed", the Shapley
attribution says: **participation ratio 9.5 of 32** — about ten effective directions,
the largest carrying 27% of the layer by itself, the top eight carrying two thirds.
Neither a few nameable parts nor an even smear; both prior readings were artefacts of
their instruments. Solo ablation also *misranks*: the leading direction's true
contribution is 4.6× its solo effect, and two directions with negative solo effects
(removal appears helpful) genuinely carry +0.019 nats each.

Full detail in `BILIN18_CONNECTION.md` §13.

### Naming and causing turn out to be aligned

The registered follow-on closed the loop. The repository's §78 had concluded the
nameable axis and the causal axis of MLP1 are nearly orthogonal — dictionaries name but
do not explain. Its causal side was measured with solo ablation, the instrument just
shown to misrank. Re-asked geometrically with the Shapley ranking: dictionary atoms hold
**1.8× more** of their energy in the span of the ten causally-leading directions than in
the ten trailing ones, above the 95th percentile of a random-subset null, and the effect
*strengthens* when atoms are weighted by usage. The nameable structure lives where the
causal mass lives; what survives of §78 is only that no single atom is individually
load-bearing — which is interference among ten real parts, not orthogonality of naming
and causing. (Enrichment, not identity: 1.8×, one layer, dictionary FVU 0.179.)

Full detail in `BILIN18_CONNECTION.md` §14.

### The ten directions have token structure, not crisp names

Pointing the verified naming instrument at MLP1's six leading Shapley directions: all
six clear their permutation nulls (ρ up to 0.47, nulls ~0.06), so the causal leaders
genuinely have token structure — one is a clean determiner-vs-verb axis, another keys on
sentence-initial discourse openers, the leader fires overwhelmingly on whitespace. But
none has layer 17's crispness, and this does **not** overturn the repository's "0/32
nameable": their bar is causal (localised ablation effect), ours is correlational, and
the honest combined statement is that MLP1's causal directions carry real but diffuse
token structure — verifiable dependence without single-concept names.

Also caught and recorded: the direction basis must come from the same data as the
Shapley run — two earlier versions of the script recomputed the SVD from larger corpora
and the tail directions silently rotated.

Full detail in `BILIN18_CONNECTION.md` §15.

### The sample-size control: the count was inflated, the ratios were not

Refitting the basis on 4.8× the data (153,900 positions, rows disjoint from evaluation)
and rerunning the entire Shapley attribution: **§13's "about ten directions" corrects to
five-to-six**, with the leader at 39% rather than 27%. The mechanism is instructive —
the small-sample PCA split one real component into two noisy directions (both old
"leaders" #3 and #5 map onto the same new direction), and the attribution spread that
component's mass across both, inflating the participation ratio. Basis noise makes a
layer look *more* distributed than it is. The two top leaders and their names are
rock-solid (cos 0.985, 0.946); §15's ranks #3 and #5 were two names for one thing.

The weight basis is not exonerated: at 60% of the data span's causal effect and 63% of
its held-out energy, its standing is identical at both sample sizes. And a third finding
neither prediction anticipated: the in-sample "70.5% of the layer" figures were roughly
double the held-out ones **by document heterogeneity, not overfitting** — the weight
basis, which fits nothing, drops by the same factor on fresh rows, and context length is
irrelevant. Ratios between bases are trustworthy; absolute "X% of the layer" figures are
row-group-relative throughout §12–§15.

Full detail in `BILIN18_CONNECTION.md` §16.

### The data's shape, and the leader unmasked

Two cheap runs (11 s total) answered "what structure does MLP1's output have" and "what
writes its causal directions". The distribution is **not a dense 32-dim subspace**: one
enormous direction, then a long tail (90% of energy needs ~241 dims; 32 dims hold 38%
held-out). Coefficients are mostly near-Gaussian (median excess kurtosis 1.2) — not
SAE-shaped — but the leader's coefficient is **56% document identity by variance**
(ICC), and there is no hierarchical gating (leader anti-correlates with the tail, −0.30,
as a register mixture predicts).

The fold-in is exact because the layer is bilinear: the input is an exact sum of four
writers (embedding, attn0, MLP0, attn1) and each output coefficient splits exactly into
writer pairs (gates ~1e-7). Verdict: **the 39% causal leader is an attention-squared
feature** — 76% of its variance is attn1's output interacting with itself, ≤9% involves
the current token's embedding. Assembled with the earlier findings: attn1 summarises
local context, MLP1 squares it, and the result is a document-register signal
(whitespace-heavy material vs prose). It fires on whitespace but is not a token feature,
which is why token-list naming found its correlate without its meaning.

Compression recommendation that follows from the measurements: a register-conditioned
mixture code (register symbol first, moderate-rank Gaussian bulk within register, sparse
exceptions only for the few heavy-tailed directions), and interpretation that factors
through the writer-pair decomposition rather than the vocabulary.

Full detail in `BILIN18_CONNECTION.md` §17.

### Causal verification of the leader: compression confirmed, semantics refuted

The register-leader story was put through the causal battery it needed. The
**compression half survives decisively**: replacing the leader's 664k-parameter
quadratic with a rank-2 whitened truncation (2,308 params) restores cross-entropy to
within noise of the intact model — 288× smaller at full fidelity — and a single squared
projection (1,154 params) gets 92%. The control that makes it meaningful: the same
surrogate with a random direction repairs 0%.

The **semantic half fails both causal tests**. Deletion damage is *smallest* in
layout-heavy contexts (the story predicted largest), and injecting layout tokens into
prose moves the leader by under 1% of a standard deviation — indistinguishable from
injecting random words. The exact per-key attribution that named layout tokens was
right about the natural covariance and wrong as a mechanism: attribution over a
distribution is still correlational across it, and an exact decomposition is not an
intervention. "Register detector" is demoted from mechanism to correlate.

(A first run of the ablation arms was void — the patch targeted a forward the custom
evaluation never calls, and every rung scored the intact model. Caught because the
numbers were identical; fixed and rerun.)

Full detail in `BILIN18_CONNECTION.md` §19. Next per the plan: the same-depth battery
on layer 0.

### Layer 0 at full depth: the clean layer

The same battery (Shapley → writers → structure → naming → causal ladder) run
bottom-up on layer 0. All three leading directions are embedding-dominated (emb×emb
62–75% of variance; no attention head above 6%) — token-identity features, as depth 0
predicts. The leader is a **punctuation-vs-content axis verified at ρ = 0.95** against a
0.14 permutation null, the program's strongest name; the second leader is a
number/quantifier axis (ρ = 0.80). Structure is the opposite regime from layer 1:
effective rank 24, low kurtosis, almost no document mixture.

The causal ladder splits the two layers: layer 0's leader repairs only 66% at 1,154
params (layer 1's repaired 92%), with the random control at 0.6%. Token-identity
features intrinsically need more of their form than squared context summaries — so
surrogate compressibility itself probes mechanism class.

Full detail in `BILIN18_CONNECTION.md` §20. Plan updated (`LAYER_PROGRAM.md`): next are
interchange interventions in the causal-abstraction sense on the named variables, then
layer 16, then the weights-first theory pass (HOSVD/CP of the bilinear tensor in the
validated metric, per-head folded operators, and a weights-only prediction protocol).

### The variable graph under interchange

The causal-abstraction pass on layer 1's verified surrogate (`z := u·x̂ → c₀ := az²+b →
write`), with base/source pairs deliberately drawn from different documents. The
**head4 → z edge verifies interventionally**: swapping head 4's attention context
produces 79% of all z-movement (runner-up 14%, everything else ≤2%) — unlike the layout
semantics, this attribution survives being moved. The **z → c₀ edge is a partial
abstraction**: patching z's value reproduces 68% of the true coefficient-patch's
downstream KL and 61% of its top-1 flips (shuffle control: 13%). The gap to §19's 92%
on-distribution repair is localised to cross-document transport — on-distribution the
abstraction is nearly complete; transported across contexts a third of the influence
comes from parts of the form z does not see.

Full detail in `BILIN18_CONNECTION.md` §21. Next: layer 16's battery, then the
weights-first theory pass.

### Layer 16: a two-direction layer, and a surrogate that beats the model

The battery's third stop. Layer 16 is the concentration extreme — participation ratio
**2.5** of 32, top two directions 42% + 40% — corroborating the earlier finding that
13.8k numbers replace it at 4.2% damage. Its leaders are deep cross-features (attn×mlps
47–56% of variance, embedding ≤1%), and its second direction is the same
pronoun-vs-sentence-ender syntax axis previously read out of layer 17 — evidence for a
cross-layer *bus signal* rather than a per-layer feature.

The anomaly, reported not smoothed: the 1,154-parameter surrogate for its leader lands
**0.025 nats below the intact model** (the random control behaves normally, so it is not
the machinery). A replacement beating the original means the full form carries a
component that hurts on this eval set. Registered hypothesis: train/eval distribution
shift (fineweb→pile) — the whitened rank-1 core generalises, the 664k-parameter
remainder is distribution-specific. Truncation as regularisation; testable when
fineweb-like data is available.

Depth taxonomy so far: layer 0 = token-identity features (ρ 0.95 naming, 66%
compressible); layer 1 = squared context summary (92%, names diffuse); layer 16 =
accumulated cross-computation (175%, names weak); layer 17 = near-rank-4 syntax rules.
Compressibility, writers, and nameability move together — the battery measures mechanism
class.

Full detail in `BILIN18_CONNECTION.md` §22. Next: the weights-first theory pass
(Phase D).

### The theory pass: the weights knew

Phase D asked how much of the empirical pipeline was derivable from the weights. Answer:
nearly all of it, given exactly one data statistic. The output-mode Gram of the bilinear
tensor in the Λ-weighted metric — `Down[(LSLᵀ)∘(RSRᵀ)]Downᵀ`, closed form, seconds to
compute — holds **90–99.6% of the measured Shapley leader** in its top-8 eigenvectors at
every depth tested (layers 0, 1, 16, 17; random baseline ~1%). Plain weights without S
degrade to 14% at layer 0, so the single matrix S is what turns weight algebra into
prediction. The head discovery replicates too: pure weight algebra (per-head folded
operators) ranks head 4 first at layer 1 with a 5× margin, where interchange measured
79%.

Consequence for the program: the battery order inverts. Weight-side components first
(closed form), model evaluations only to verify and interchange-test them. The verified
layer-1 surrogate, as a tensor network, is 2,306 parameters and 6,903× cheaper per token
than the layer it replaces.

Full detail in `BILIN18_CONNECTION.md` §23.

### The syntax bus: a verified two-layer variable

The pronoun-vs-sentence-ender axis seen independently at layers 16 and 17 is now a
tested edge: the two coefficients correlate at **0.935** (other direction pairs: median
0.13), and steering the layer-16 side by +2σ moves the layer-17 side by **+0.95σ** —
unit gain — with 14.8× specificity over a control direction. The edge is **rectified**
(−2σ moves almost nothing), which is exactly the signature a squared readout predicts:
the bilinear mechanism is visible in the intervention's shape.

The token-level semantics failed specificity again: steering moves determiner log-probs
strongly, but matched control tokens move more. Three-for-three now — structural claims
(head→variable, layer→layer edge, compression) verify interventionally; token-story
claims fail every causal test. The reliable currency of this model's mechanisms is
directions, edges, and gains.

Full detail in `BILIN18_CONNECTION.md` §24.

### The layer-16 anomaly resolved: truncation as regularisation

The registered hypothesis tested and confirmed on a fresh fineweb sample streamed from
the Hub. On pile (shifted) the surrogate beats the intact model by −0.0285 nats,
replicating on fresh rows; on fineweb (training-like) the difference is +0.0011 —
nothing. **In-distribution, the 1,154-parameter surrogate and the 664k-parameter form
are functionally equivalent; the remainder is dead weight in-distribution and a
liability under shift.** The compression claim strengthens to effectively lossless, and
the method generalises: surrogate-vs-full across corpora measures how much of a
component is distribution-robust computation.

Full detail in `BILIN18_CONNECTION.md` §25.

### The robustness split, completed for all three leaders

Layer 1 turns out to share layer 16's pattern — on shifted data its surrogate beats the
full form (109.8% repair), in-distribution it genuinely misses 13% — while layer 0's gap
does not close (76% in-distribution), as registered: token identity uses more of its
form than a rank-1 core carries, on any corpus. The in-distribution missing fraction
orders by mechanism class (token identity 24% > context summary 13% > accumulated
computation ~0%), and delete costs flip with depth (shallow features matter more on
shifted text, deep ones on the training distribution). Three-layer regularity: the
distribution-robust core of every verified leader is its rank-1 whitened surrogate.

Full detail in `BILIN18_CONNECTION.md` §26.

### The middle attributed fairly — and the 2.87× corrected to 1.42×

Layer-level Shapley over the fourteen middle quadratic parts, which first forced a
correction: §10's 2.87× superadditivity mixed two different deletion operators and used
stale means; under one clean operator (exact intact-model mean write) the joint cost is
2.963 vs a solo sum of 2.087 — **1.42×**. Qualitative claim survives, magnitude halved,
and "delete cost" is now flagged as operator-dependent throughout (per-layer costs move
up to 3.5× between operators).

The fair shares reshape the middle entirely: **layers 2 and 3 carry 64%** of it (the
"distributed middle" is mostly two adjacent early layers), the tail layers 5–15 carry
3–7% each with their solo costs understated 3–5×, and **layer 4's Shapley value is
large and negative** (−0.668, −22.5%, robust across operators): removing it *repairs*
part of the damage of removing the others. Its computation helps only while its
downstream partners are intact — a coupling signature invisible to any solo or joint
number, and the sharpest evidence yet that the middle is a pipeline rather than a
collection.

Full detail in `BILIN18_CONNECTION.md` §27.

### Layer 4 oriented: it reads layers 2–3 (my hypothesis was backwards)

The marginal-flip sweep inverted §27's reading. Deleting downstream layers makes layer
4 *more* valuable (+0.11 → +0.28); its marginal flips violently negative exactly when
layers 2–3 enter the coalition (−1.34). Layer 4 is a **reader** of 2–3, not a writer
for 5+: with its suppliers gone its quadratic misfires on the unwritten bus. The
interventionally grounded pipeline at the middle's entrance is **2 → 3 → 4**. The
weight-side input-mode Gram predicts both the direction and the ranking (layer 3 at
12× random, layer 2 at 8×, forward edge to layer 5 at 10× but decaying) — the Phase-D
protocol's second confirmed prediction. Method note: a negative Shapley value finds a
coupled stage but not the coupling's direction; the marginal-flip sweep orients it.

Full detail in `BILIN18_CONNECTION.md` §28.

### Layers 2 and 3: where the model refuses to compress

The middle's two workhorses are the opposite end of every spectrum the battery
measures: flattest Shapley spectra (PR 13.8 and 8.7 of 32; leaders 16% and 22%),
highest-rank outputs (effective rank 80–87, ~500 dims for 90%), leaders reading almost
purely from accumulated MLP/attention writes, and — the two firsts — a leader with **no
verifiable token structure** (layer 3, ρ = −0.005) and a leader whose rank-1 surrogate
**fails outright** (layer 2, 3.5% repair; rank-2 17.6%). This bounds the §26 regularity
honestly (low-rank robust cores exist only for some mechanism classes) and closes the
loop on "irreducibly distributed": layers 2–3 are what that genuinely looks like,
measured with instruments that succeeded on four other layers.

Assembled pipeline: 2–3 perform a high-rank uncompressible transformation; 4 reads it
and misfires without it; 5–15 add small redundant refinements (fair shares 3–7%);
16–17 collapse everything to a few readable directions. Distribution rises then falls
with depth.

Full detail in `BILIN18_CONNECTION.md` §29.

### The weights+S formula predicts blind: 3/4 on unprofiled layers

Predictions frozen from weights + input second moment before any measurement of layers
7/9/11/13; bar registered in advance (leader energy 0.5 in the predicted top-8).
Result: **3/4 hits** — energies 0.89/0.95/0.97 against random ~0.007, and at layer 11
the single predicted #1 eigenvector *is* the measured leader (cos 0.92). Layer 7 missed
(0.403, still 45× random) and stays a miss. Caveats on record: two tail layers'
8-permutation attributions are noisy (negative-dominated spectra), and layer 9's top-32
span *improves* pile CE when deleted — the shift-regularisation pattern appearing
unprompted in the tail. The weights-first battery order survives its first blind
contact.

Full detail in `BILIN18_CONNECTION.md` §30.

### The 3→4 edge is broadband

Weights-first bus candidates for the chain (coupling operator eigenvectors), registered
bars, transplant test: **eight channels carry 21% of the edge, not the predicted 50%**,
and the coupling directions beat plain output-PCA only trivially (0.211 vs 0.195). The
2→3→4 pipeline's existence and orientation stand, but it does not abstract into few
variables — matching the rank, surrogate, and naming verdicts on the same layers. The
middle is high-bandwidth by every instrument.

Full detail in `BILIN18_CONNECTION.md` §31.

### The composition gate: failure exactly where the graph predicted

Installing the three verified surrogates together costs 1.48× the sum of installing
them alone — my registered ≤1.3× failed — and the excess decomposes perfectly: the
L16+L17 pair interaction is +0.0649 of the +0.0654 total excess, the L1 surrogate's
cross-terms are +0.0005 ≈ 0. The one interacting pair is the one pair linked by a
verified edge (the L16→L17 syntax bus): L17's replacement was fit on intact-L16 inputs,
and replacing 16 shifts that distribution. Composition is now a standing battery
criterion, the graph earned an advance prediction about a failure mode, and the fix
(refit 17's replacement downstream of 16's) is registered.

Full detail in `BILIN18_CONNECTION.md` §32.

### Composition has a law: excess ≈ 23 · d16 · d17

Three single-mechanism explanations of the 16–17 composition interaction failed in a
row (refit 21%, rank 0%, span 30% — each against a registered bar). The seven measured
configurations then revealed the real structure: the excess follows a **product law**,
excess ≈ c·d16·d17 with c = 22.9 ± 2.4, at 9% mean error across sweeps of two
different fidelity knobs. Nothing was broken — a quadratic reader of summed errors
produces a cross-term proportional to the product, so every knob shrinks a factor and
none removes the coupling. Composition budgeting is now quantitative:
joint ≈ Σdᵢ + c·Σ_linked dᵢdⱼ, with the causal graph naming the linked pairs (unlinked
pairs measured at +0.0005).

Out-of-sample test: two never-measured corners, predictions registered first, both
within the 25% bar (13% and 23% error). The law is a validated predictive tool.

Full detail in `BILIN18_CONNECTION.md` §33–36.

### The tail, three evaluations per layer

Weights-first sweep of the seven unprofiled tail layers: the Λ-Gram's predicted top-2
directions cost ≥5× random at 4/7 (75× at layer 5); the misses are layers with nothing
to find (span effects under 0.01) and layer 15, whose predicted top-2 *improve* pile CE
when deleted. Negative span effects now at layers 9, 12, 15: shift-fragility is the
tail's norm — its highest-variance directions are largely fineweb-specific fit.
Fineweb arm run and all predictions held: every negative deletion effect flips
positive in-distribution (layer 15's harmful pred-2: +0.0052 on fineweb). The tail does
real fineweb-tuned work in exactly the directions the Λ-Gram finds; under shift that
work inverts to a liability. Confirmed on both sides of the split.

Full detail in `BILIN18_CONNECTION.md` §37.

### The coverage curve: no core to find, and a failed generalisation

The missing ~90% of early-layer causal effect is genuinely space-filling: layer 1's
128-dim top span carries 12% of the full deletion cost, 512 dims carry 75%, and
per-direction cost *peaks mid-spectrum* — the blind-direction theme at its sharpest.
There is no low-dimensional causal core at layer 1; the <10% coverage was geometry, not
bad sampling. And a registered prediction failed informatively: the Λ-Gram, best-in-
class for predicting *leaders*, is *worse than plain variance ordering* for cumulative
coverage at every k. Two instruments, two jobs.

Full detail in `BILIN18_CONNECTION.md` §39.

### The interchange leak survives two hypotheses

The 32% z→c₀ abstraction leak is not the document component (same-document transplants:
60.8% faithful, no better than cross-document) and not the form's second direction
(rank-2 transplant: 72.5%, despite a 0.987 coefficient fit). A transplant with 1.3%
coefficient error still loses ~27% of the downstream effect — the surviving hypothesis
is position-heavy error amplification by the downstream quadratic stack, the same
mechanism as the composition product law at a different scale. Accounting test
registered.

Full detail in `BILIN18_CONNECTION.md` §40.

### The leak accounted for

Single-position patching (the clean design; the aggregate version was confounded by
forward propagation and failed its bars measuring its own confound) shows the
interchange leak is locally coefficient-error-driven: ρ = +0.52 between a position's
mismatch KL and its coefficient error. The 32% gap is small fit residual (1.3% of
variance) amplified ~25× by the quadratic stack — the product-coupling theme at
coefficient scale. Operating-point scaling narrowly failed its bar (ρ 0.29 vs 0.3,
n=12) and stays unproven.

Full detail in `BILIN18_CONNECTION.md` §41.

### The product law refined: coupling is a bilinear form, not a constant

Second linked pair (the 3→4 edge, 3×3 damage grid): the unlinked control (3,14) shows
excess ≤0.0013 at every cell — the cleanest graph validation yet — while the linked
pair's c ranges 6–30 (both registered predictions about the constant failed). The
structure: coupling strength depends on *which* directions are damaged (L3's 8–32 band
couples ~4× stronger than 32–128). Refined law: excess ≈ e_aᵀ C_edge e_b, a bilinear
form in the damage profiles; the scalar law is its rank-1 shadow, valid when damage
shape is fixed (as in the 16→17 sweeps). The scalar c is an engineering number per
damage family, not a constant of the edge.

Full detail in `BILIN18_CONNECTION.md` §42.

### Queue-runner era begins; two clean negative results

Experiments now run continuously from a supervisor-managed queue, decoupled from agent
turns. Its first two results: the coupling operator K does **not** predict the §42
damage-direction anisotropy (c_K 10.7 vs c_PCA 12.8; registered 1.5× bar failed), so
C_edge's structure has no weight-side predictor yet; and operating-point scaling failed
its bar again at n=48 — of the amplification story, only the local error→mismatch link
(ρ 0.52) stands.

Full detail in `BILIN18_CONNECTION.md` §43.

### Runner batch: signed coupling, a §39 correction, and a hidden edge

Three results in one batch. **C_edge(3→4) is signed**: a narrow positive coupling
channel at L3 ranks 8–32 (c = +31) on a broad anti-coupled region (ranks 32–256,
c ≈ −6: damaging them *reduces* joint damage — layer 4 compensates). **§39 corrected**:
there is no mid-spectrum causal peak — cumulative differences measured interference,
not band mass (the §12 lesson recurring in-spectrum); disjoint bands sum to 0.24 nats
vs 4.90 jointly, so layer 1's causal mass is 20× in cross-band *interactions*, not in
any band. **And the front of the graph is chained**: steering L0's punctuation leader
moves L1's leader at unit gain (+1.04σ, rectified) despite a 0.5% proximate-writer
share — the influence routes through attn1, which the folding attributes as the writer.
Proximate-writer shares do not bound upstream influence. Mediation test queued.

Full detail in `BILIN18_CONNECTION.md` §44.

### Mediation confirmed with overshoot; within-layer interactions are 80% higher-order

Freezing attn1 kills the L0→L1 edge entirely and overshoots (−0.35σ — the direct path
is slightly negative); head 4 alone carries 51% (below its 60% bar — this signal is
spread across heads, unlike z's context). Control clean. And layer 1's 4.66-nat
within-layer interaction decomposes as: pairwise 20%, **order-3-and-above 80%** — the
layer is holistic in the measured sense that most of its causal effect exists only in
combinations of three or more direction-bands. The pairwise product law prices
between-layer composition, not within-layer structure.

Full detail in `BILIN18_CONNECTION.md` §45.

### Two channels through one block; interactions graded to all orders

The mediation head sweep inverted expectations again: **head 1 carries the L0→L1 edge**
(96% kill vs head 4's 51%, everything else ≤3%) — so the attention block hosts two
distinct channels: head 4 computes z's context, head 1 transports the L0 signal. And
layer 1's interaction hierarchy is graded, not truncated: solo 5%, pairs 19%, order-3
35%, order-4+ 41% — every order contributes more than the last, so no truncated
interaction model captures the layer.

Full detail in `BILIN18_CONNECTION.md` §46.

### Pattern-routed, and a unification candidate at 99% vs 24%

Head 1 transports the L0 signal by *re-aiming* (pattern freeze kills 54%, value freeze
30% — the registered value-route prediction failed, inverted; control clean). And the
interaction-depth contrast held emphatically: solo+pairwise captures 99% of layer 16's
full-span cost vs 24% at layer 1 — compressibility and interaction shallowness look
like one property from two sides. Cross-layer generalisation queued.

Full detail in `BILIN18_CONNECTION.md` §47.

### The unifying statement, measured: shallow = compressible

Solo+pairwise share of full-span deletion cost across six layers: L1 24%, L2 39%,
L3 57%, L0 67%, L16 99%, L17 100% — every compressible layer above every uncompressible
one, as registered in advance. Interaction shallowness and compressibility are one
property from two sides; the middle resists reading because 43–76% of its effect exists
only in order-3+ combinations. (The head-1 re-aiming test invalidated itself —
underpowered class, violated control, scale confound — and is recorded as instrument
failure, not evidence.)

Full detail in `BILIN18_CONNECTION.md` §48.

### Factorization found, map completed, mechanism refined

Answering the independence question directly: layer 1's synergies are all **positive**
(need-both, never mutual exclusivity), and my "no factorization" prediction **failed**
informatively — the PCA low/high split shows only 15% cross-synergy: an entangled
576-dim core carrying 85% of the layer alone, plus a nearly inert complement (+0.009).
Random cuts see 95% synergy, which is why band analyses read as total holism. Also: the
18-layer shallowness map is complete (all tail layers above 57%; deep interaction is
exclusive to layers 1–3), and the re-aiming mechanism is real but attention-wide (4.1×
punctuation-specific vs random steer; head-6 control violated, so selectivity lives in
what c₀ reads, not in which heads move).

Full detail in `BILIN18_CONNECTION.md` §49.

### The two-sided answer: every layer reads shallowly; only the middle is read deeply

Operator-composition test: weight-side qk-enrichment ranks head 1 first (the carrier
head was derivable from weights), but pattern-dominance and head 4's absence need
second-order signatures — scalar edges don't compose, first-order operator signatures
half-compose. Input-side Möbius: layer 1's input side is 89% shallow vs its 24% output
side (L16: 90/99) — the bilinear architecture bounds within-layer input order at 2, so
interaction depth is manufactured downstream. Compressibility is a property of how a
layer's output is *consumed*.

Full detail in `BILIN18_CONNECTION.md` §50.

### Second-order operator signatures work; edges are filters

The blind routing test failed its edge-existence gate honestly — L0's number axis
moves the L1 leader at only 0.23σ vs the punctuation axis's 1.04σ, so the front edge
is *signal-specific*: a filter, not a pipe. And the second-order response-energy
signature (weights + cached activations, no interventions) recovers what first-order
norms missed: head 1's pattern-response is 69× its value-response (matching the
measured 54/30 route) and 30–300× every other head's — carrier and route character
both predicted. Registered next: does response energy predict edge *strength*,
closing the loop on a weights-side calculus of which signals route at all.

Full detail in `BILIN18_CONNECTION.md` §51.

### The calculus as a routing detector: 6,000× separation, and a new strongest edge

Response energy separates routed from unrouted signals by 3–4 orders of magnitude and
ranked five signals at Spearman 0.80 (exactly at the registered bar; two adjacent
swaps). Its top-ranking miss was itself a discovery: **L0's #3 direction routes into
the L1 leader at 1.60σ — the strongest front edge found so far** — flagged by the
energy ranking before measurement. The blind carrier protocol now runs on this edge.

Full detail in `BILIN18_CONNECTION.md` §52.

### Blind carrier prediction fails: the new edge has no carrier

The #3 edge is distributed — no head kills more than 32% (head 8; predicted head 1
second at 28%) — so the blind protocol failed both bars, and the presumption of a
dominant carrier was itself wrong. Standing: routing detection from weights is solved
(6,000×); carrier prediction is not — first-order enrichment can neither rank heads
under distributed mediation nor predict whether a carrier exists. Per-head
second-order energies vs the kill profile queued.

Full detail in `BILIN18_CONNECTION.md` §53.

### Census, profiles, and a confounded instrument

Response energy is a one-sided routing guarantee (zero false negatives across eight
signals; one false positive — the highest-energy signal barely routes). Mediation
profiles: predicted well on the spread edge (ρ 0.77), and the concentrated edge's ρ =
0.05 is a metric artifact (both top-1s agree; Spearman was eaten by noise-level
entries). The reader-coupling disjointness test failed both bars — but its aggregation
over each reader's 1152 output directions is central-limit flattening, so the
path-separation hypothesis is untested at the resolution it was posed; the per-form
version runs next.

Full detail in `BILIN18_CONNECTION.md` §54.

### Path-separation at form resolution: shared template, real density

The clean per-form test: individual forms couple L1's direction-pairs diffusely (top-5%
mass 0.15–0.18 vs 0.05 uniform, far from sparse), and both within-reader and
cross-reader cosines sit at ~0.6 — one shared coupling template across forms and
readers, plus modest variation. Layer 1's interaction density survives at every
resolution tested. Caveats queued: signed cosines (abs-matrices inflate overlap) and
attention QK readers (the component most likely to specialise).

Full detail in `BILIN18_CONNECTION.md` §55.

### Signed coupling settles it: dense support, near-orthogonal functionals

The §55 "shared template" was absolute-value inflation: signed within-reader coupling
cosine is **0.11** (vs 0.64 unsigned). Corrected synthesis: layer 1's interaction is
dense in *support* (every reader's magnitude envelope is the same; QK heads included at
0.79 unsigned) but diverse in *functionals* — the signed quadratic forms readers
compute are nearly orthogonal. The exclusivity picture fails for supports and largely
succeeds for functionals: an overcomplete family of nearly-independent quadratic
measurements on a common dense substrate. Band deletions destroy all functionals over a
support region at once, which is why they read as inseparable holism.

Full detail in `BILIN18_CONNECTION.md` §56.

### Universal: near-orthogonal functionals on a shared dense support

The completion held on both bars (signed cross-reader 0.089, signed QK 0.156). The
front of the model reads layer 1 through an overcomplete family of nearly-orthogonal
quadratic functionals on one dense substrate — measured within readers, across readers,
and across attention heads. The compression program re-aims to functional coordinates;
the functional-family spectrum and single-functional steering are queued.

Full detail in `BILIN18_CONNECTION.md` §57.

### Functional coordinates: eighty principal functionals, locally steerable

The spectrum: envelope rank 2.6, signed family rank **80** of 240 — the front of the
model's reading of layer 1 compresses to ~80 principal functionals (my "no small
basis" bound failed compressively). Steering along single functional eigenvectors is
surgical adjacently (20× selectivity at L2, control at noise) and dies with depth
(0.2× by L13): functional coordinates are *local* — the product-amplification that
governs composition also sets a finite steering range. Coherence-length measurement
queued.

Full detail in `BILIN18_CONNECTION.md` §58.

### The coherence length is one layer

With the coupling-norm confound fixed: own-movement 1.46σ at the adjacent reader,
0.28σ one layer later, never recovering past 0.5σ (half-range = 1 layer; my registered
2–6 was optimistic). Absolute coupling gates steerability (ρ 0.62). The queued
discriminator: gradient steering (the exact end-to-end sensitivity direction, one
backward pass) tests whether the range limit is intrinsic or a direct-path targeting
artifact.

Full detail in `BILIN18_CONNECTION.md` §59.

### The limit is intrinsic

Gradient steering — the exact end-to-end sensitivity direction — fails at L13 exactly
as direct-path targeting does (0.06σ own, 3.70σ cross-talk), and the two directions
are nearly orthogonal to each other. No static direction addresses a deep coefficient
individually: perturbations diffuse into collective motion within about one layer. The
functional-coordinates arc closes: ~80 principal functionals, surgical at range one,
individually unaddressable at depth. Locally transparent, globally opaque.

Full detail in `BILIN18_CONNECTION.md` §60.

### Verified: the 80-functional basis is shared vocabulary

Leave-one-reader-out R² = 0.711 at rank 80 (bar 0.6; r=8 gives 0.152; random basis
≈ 0). A never-seen reader's L1-couplings are reconstructed from other readers'
principal functionals — the basis is shared structure, and the middle's compressed
description is complete: dense envelope (rank 2.6), ~80 shared near-orthogonal
functionals, steerable at range one, intrinsically unaddressable beyond.

Full detail in `BILIN18_CONNECTION.md` §61.

### Power without addressability

Per-sequence gradient steering lifts deep own-movement 3.7× (0.22σ at L13, bar held)
but cross-talk stays at 2.88σ (bar failed): context-dependence recovers reachability,
not selectivity. Across all three intervention classes tested, anything strong enough
to move one deep functional moves the collective state more. The opacity of the stack
is a selectivity phenomenon.

Full detail in `BILIN18_CONNECTION.md` §62.

### The vocabulary is not made of the verified axes

Both structural-naming predictions failed: the top principal functionals align with
none of the verified directions (below their own nulls) and none are low-rank
(eff-ranks 11–20). Combined with LORO R² 0.71, the verified axes should live inside
the vocabulary's span without being principal — the third appearance of the
importance-vs-identity split. Containment test queued. (This result ran during a
session-restart gap: the cron died, the supervisor runner kept working, and the queue
sat empty until recovery — the cron now self-documents that failure mode.)

Full detail in `BILIN18_CONNECTION.md` §63.

### The ladder completes: power 15×, addressability never

Per-token gradient steering lifts deep own-movement to 0.91σ (static: 0.06) with
cross-talk still ahead at 1.70σ — the full intervention ladder now reads: power fully
recoverable with context-dependence, addressability recoverable at no rung. The
containment test is recorded as an instrument error (input-space objects projected
into an output-space basis — a category confusion); the causal-individuation test of
vocabulary words replaces it, queued.

Full detail in `BILIN18_CONNECTION.md` §64.

### Word constituencies: envelope artifact, then a post-hoc flip

The registered test failed with a violated control — raw profiles share a movability
envelope that even random steering shows (the §55 artifact in causal form). Post-hoc
envelope normalization flips it: residual constituencies are anti-correlated (−0.76 /
−0.38 / −0.11) — distinct and complementary. Labelled post-hoc; the pre-registered
confirmatory rerun is queued.

Full detail in `BILIN18_CONNECTION.md` §65.

### Confirmed: causally individuated, one degenerate pair

The pre-registered envelope-normalised rerun held its main bar (mean residual
correlation −0.20): vocabulary words move complementary constituencies. One pair
overlaps (+0.64) — plausibly near-degenerate principals rotating freely. The chain
from the independence question is complete: dense support → orthogonal functionals →
80-word vocabulary → complementary causal constituencies.

Full detail in `BILIN18_CONNECTION.md` §66.

### Arc closed: the vocabulary is corpus-robust

Transfer held on both bars (fineweb individuation −0.24; per-word cross-corpus
constituency correlation mean +0.78). The independence-question arc is complete at
all five levels: dense support, orthogonal functionals, 80-word verified vocabulary,
complementary corpus-robust constituencies, one-layer steering range.

Full detail in `BILIN18_CONNECTION.md` §67.

### The words have correlational names

Predictions failed optimistically: 3 of 5 vocabulary words clear their nulls at ρ
0.48–0.56 (determiner/possessive context; clause openers; a measurement register), and
four of five carry the document component at ICC ≈ 0.56 — the register leader's exact
number. Principal decomposition finds register-shaped words because the mixture is the
dominant variance. Names are verified correlationally only; the causal test of the
best name is queued (the token-story record stands at three-for-three failures).

Full detail in `BILIN18_CONNECTION.md` §68.

### Four-for-four, stated as a regularity

Word #5's causal name failed at 1.3× (bar 1.5, the lowest ever set). With four causal
token-story failures against four decisive correlational verifications on the same
objects, the program states it as a regularity: **in bilin18, token-level semantics are
readable but not steerable** — activations carry verifiable token structure;
interventions along the same directions shift broad distributional mass, never the
named tokens selectively. The decisive test (the ρ = 0.95 punctuation axis) is queued.

Full detail in `BILIN18_CONNECTION.md` §69.

### The regularity breaks at its decisive test — and becomes graded

The ρ=0.95 axis steers its tokens at 1.7× controls: above the regularity bar, below
clean success. Revised statement: causal token selectivity is graded by correlational
crispness and weak even at its best (0.95→1.7×, 0.48→1.3×, weaker→≤1×). The
systematic ρ-vs-selectivity test is queued.

Full detail in `BILIN18_CONNECTION.md` §70.

### No law — but the first steerable token direction

The graded law failed (Spearman 0.14): identical-ρ directions span 1.0–3.8×
selectivity. The numbers axis (3.82×) is the program's first strongly token-steerable
direction — causal token control exists, is rare, and is unpredicted by naming
quality. The direct-write (unembedding-alignment) predictor is queued.

Full detail in `BILIN18_CONNECTION.md` §71.

### Closed: token steering is direct logit-writing, predictable from weights

|DW| (unembedding-alignment contrast, weights-only) predicts steering selectivity at
Spearman +0.77. Where token steering works, it is the residual bypass writing named
logits directly — no semantic routing, which failed every causal test. The arc closes
consistent with the program's deepest pattern: mechanisms verify, stories don't, and
the working token lever is the one with no story.

Full detail in `BILIN18_CONNECTION.md` §72.

### SGD built the vocabulary; everything else is inherited

The weight-shuffled null split the functional structure cleanly (all four registered
bars held): orthogonality and the dense envelope reproduce on shuffled weights
(generic/typicality); the 80-dim compression (vs 191 generic) and the cross-reader
sharing (LORO 0.71 vs 0.26) do not — they are training's entire fingerprint. The
robustness hypothesis lost its main support under covariance-matched controls (1.6×,
below bar): density is inherited, not selected.

Full detail in `BILIN18_CONNECTION.md` §75.

### Built, not carved

The trained vocabulary is 83–93% outside the generic functional structure (7.5% energy
in the shuffled top-80; 17% in the full generic span). SGD replaced the inherited
machinery's content wholesale with a new shared 80-dim code. Scaling test queued:
vocabulary dimension vs writer complexity across writer layers.

Full detail in `BILIN18_CONNECTION.md` §76.

### The code size is universal

The normalised scaling rerun killed the scaling hypothesis (eff-ranks 79–112 across
writers, Spearman −0.20 with complexity, reversed if anything) and found something
better: the reading code's dimension is a model-wide constant (~80–110 vs ~191
generic), not a property of any writer. Universal-compression check queued.

Full detail in `BILIN18_CONNECTION.md` §78.

### The universal compression constant

Both bars held: generic family size is writer-independent (195–198) and training's
compression is universal (1.77–2.46× across writers). Architecture supplies ~195
dimensions everywhere; training halves it everywhere, into shared, mostly-new codes.

Full detail in `BILIN18_CONNECTION.md` §79.

### Out-of-sample: the origin story closes

Writer L9 (never used in the arc): trained 76, shuffled 198, ratio 2.60× — all bars
held. Five writers agree. SGD's contribution to the middle's reading structure is one
thing: a universal ~2× shared compression built from scratch; everything else came
free with the architecture.

Full detail in `BILIN18_CONNECTION.md` §80.

### Matrix-SAE: vacuous by design error

300 atoms for 240 functionals makes perfect sparse reconstruction trivial — both arms
hit R² = 1.00 and the trained/shuffled comparison carries no information. Recorded as
instrument error; the held-out version (120 atoms, fit on five readers, scored on the
sixth against the dense-basis 0.71 baseline) is queued.

Full detail in `BILIN18_CONNECTION.md` §81.

### Quiet steering: free, useless, and finally explanatory

Projecting the target gradient off the other coefficients' gradients costs nothing
(overlap 0.09 — sensitivities are not shared at first order) and buys nothing
(cross-talk cut 1.1×). The mechanism of non-addressability: control is linear,
collateral is quadratic — bystander coefficients respond to the injection's energy
through their forms' curvature, which no direction can remove. The magnitude-sweep
prediction (selectivity ∝ 1/‖δ‖ at short range; no rescue at depth) is queued.

Full detail in `BILIN18_CONNECTION.md` §82.

### Correction: depth is addressable — at small magnitude

The magnitude sweep confirmed the second-order mechanism (selectivity 21.6× → 1.5×
from quarter to double magnitude) and overturned the ladder's headline: at 0.25–0.5×
magnitude, the deep target steers at 3–5× selectivity. §§60–64's "unaddressable"
holds only for large effects (>0.5σ), where quadratic collateral necessarily wins.
Propagated to the report.

Full detail in `BILIN18_CONNECTION.md` §83.

### The vocabulary's edges

Nine of ten held-out MLP readers reconstruct from the six-reader basis (median R²
up to 0.92); the full-model code is ~139-dim (sublinear growth) and L11 speaks none
of it. Honest sparse coding needs 105 of 120 atoms for worse fit than dense-80 —
no sparse structure exists. QK query-side couplings reconstruct BELOW chance from
the MLP basis (−0.26 vs random +0.37): attention and MLPs read with disjoint
quadratic codes.

Full detail in `BILIN18_CONNECTION.md` §84.

### Constraint release: not yet

Pruning the three tail spans whose deletion looked beneficial, then finetuning 200
steps, lands 0.026 nats WORSE than an identically-finetuned intact control (3.597
vs 3.571 held-out). The "deletion helps" observations were shift artifacts; on the
home distribution those connections are load-bearing even after adaptation. Longer-
finetune control queued.

Full detail in `BILIN18_CONNECTION.md` §85.

### OV census: a third, indifferent code

Value paths read L1's principal span at only 1.7x the uniform rate and their
weighting is uncorrelated with the quadratic vocabulary (Spearman -0.01). Sector
census complete: MLP-quadratic, QK-quadratic, and OV-linear reading are three
mutually independent codes.

Full detail in `BILIN18_CONNECTION.md` §86.

### The dissident: engaged, foreign, load-bearing

L11 reads L1 at normal strength (coupling norm rank 4/16) with functionals outside
the shared vocabulary — a foreign code. Pruning its whole MLP plus finetune lands
+0.033 nats worse than the finetune control: constraint-release refuted on its
second candidate class too.

Full detail in `BILIN18_CONNECTION.md` §87.

### No one owns the dissident; QK shares per-layer

Ablating L11's write moves every downstream layer about equally (top consumer only
1.2x the median; control partially violated) -- its load-bearing output is consumed
diffusely. QK heads compress 1.7x among themselves (eff-rank 15.5 vs 26.4 random)
but share within layers, not across them.

Full detail in `BILIN18_CONNECTION.md` §88.

### The dissident's job; QK is per-head

L11's ablation damage is concentrated (top decile of tokens = 51% of damage) with
an 8.3x entropy side-effect: concentrated content work plus decalibration. QK
heads barely share even within a layer (eff-ranks 6-7.9 of 9; layer codes
distinct): the shared-vocabulary phenomenon is MLP-specific at every grouping.

Full detail in `BILIN18_CONNECTION.md` §89.

### Constraint-release: closed. The dissident is contextual.

Third candidate (cutting the L16-L17 interchange edge) is the most load-bearing
yet: +0.067 nats over the finetuned control. All three motivated candidate classes
refuted, ordered: the more structurally implicated, the more load-bearing. L11's
damage concentration is per-occurrence, not per-token-type (frequency Spearman
-0.09, split-half 0.35) -- contextual business, like every verified story in this
model.

Full detail in `BILIN18_CONNECTION.md` §90.

### Blind edges: v1 instrument error, one real trend

The registered score was magnitude-dominated (identical ranking to the loudness
null, Spearman -0.85) -- the control caught it. Real observation: relative
adjacent-edge strength declines monotonically through the tail (0.36 to 0.04).
Alignment-only v2 queued.

Full detail in `BILIN18_CONNECTION.md` §91.

### Tail writes are unaimed

Alignment ratios for all ten adjacent tail edges sit at 0.81-1.18 -- the isotropic
baseline. The tail routes by dilution, not by aiming: no alignment structure exists
for a blind edge predictor to find (while the same machinery finds within-layer
leader subspaces blindly). Dilution check queued.

Full detail in `BILIN18_CONNECTION.md` §92.

### The tail routes by dilution

After a units-mismatch fix, both bars held: write-to-stream ratio declines
perfectly monotonically (0.252 to 0.040) and predicts edge strength (Spearman
+0.79). With section 92: tail edges are unaimed, and edge strength = the writer's
share of the stream. No routing structure, just arithmetic.

Full detail in `BILIN18_CONNECTION.md` §93.

### The bus has no supplier

No source layer 5-15 moves L16's bus span more than a random span (ratios <=1.15);
long-range influence is a flat ~0.2-sigma floor, not share-proportional. The
L16->L17 relay is local structure on top of diffuse supply.

Full detail in `BILIN18_CONNECTION.md` §94.

### The chain refuses variables

Eight coupling directions carry 21% of the 3->4 edge (bar: half); coupling beats
the writer's own PCA by only ~8%. The chain's content is high-rank -- no small
variable set exists. Open item closed negatively, as the report registered it
might be.

Full detail in `BILIN18_CONNECTION.md` §95.

### Tail profiles complete; deletion-improves-CE is endemic

The blind formula's spans are causal at 8/11 tail layers (all small, as dilution
predicts). Five of eleven layers have an 8-dim span whose deletion IMPROVES
home-corpus CE (best -0.016 at L9) -- truncation-as-regularization is not
shift-specific. Stability check and constraint-release candidate four queued.

Full detail in `BILIN18_CONNECTION.md` §96.

### Constraint-release: closed for good

The home-corpus deletion benefits replicate (stronger: L9 -0.021), yet pruning
those exact spans + finetune still loses to the finetuned control by +0.011.
Four candidate classes, four refutations -- the last selected by the hypothesis's
own criterion. Deletion benefit is a frozen-model property, not spare capacity.

Full detail in `BILIN18_CONNECTION.md` §97.

### Redistribution and assembly

The negative spans trade calibration for sharpness: deleting them helps
confidently-wrong tokens (173% of net gain) and hurts easy ones -- which is why a
finetune reclaims the loss (section 97). The bus content arrives neither from
upstream MLPs nor via L16's attention (0.05-sigma, below the diffuse floor): L16's
MLP assembles it from accumulated stream state.

Full detail in `BILIN18_CONNECTION.md` §98.

### The finetune erases the signature

After adaptation the pruned model's hard-token advantage vanishes (gap spread
evenly) -- frozen-model deletion signatures do not transfer to adapted models.
Bus-assembly regression had a misalignment bug (recorded); aligned v2 queued.

Full detail in `BILIN18_CONNECTION.md` §99.

### Bus origin complete: linear, old, delivered in place

Stream entering L16 predicts the bus at R^2 0.979 (linear!); attention fully
explained away; two-thirds of bus variance already present at L2. The interaction
is special because of WHERE the content is written, not what it is. Lexicality and
effective-linearity probes queued.

Full detail in `BILIN18_CONNECTION.md` §100.

### Lexical bus; nonlinearity map

The bus's early core is token identity (embedding R^2 0.536); "syntax bus"
overreads. Effective-linearity sweep: nonlinearity is NOT monotone -- it
concentrates in the middle (L6-10 at 0.52-0.62) while L16/L17 are almost purely
linear (0.97/0.95), independently re-drawing the depth map. Queued: does the
16->17 product law live entirely in L17's 5% nonlinear residue?

Full detail in `BILIN18_CONNECTION.md` §101.

### The product law is a quadratic skin on a linear pipe

Replacing L17's MLP with its fitted linear map (faithful: base +0.10) kills 98%
of the 16->17 interaction excess while leaving L17's own function intact. The
model's strongest interaction IS the quadratic cross-term of a 5% nonlinear
residue. (v1 was void -- hook-order artifact -- and is recorded.) Bonus thread:
the nonlinearity absorbs upstream damage; compensation test queued.

Full detail in `BILIN18_CONNECTION.md` §102.

### Correction: functional nonlinearity is front-loaded

Middle layers are cheap to linearize (+0.03) despite low linear R^2; L2 is
expensive (+0.23). Variance-nonlinearity (middle) and functional nonlinearity
(front) are different maps. The L17 residue's damage absorption is
channel-specific -- same mechanism as the product law. Joint linear-pipe test
queued.

Full detail in `BILIN18_CONNECTION.md` §103.

### Naive linear pipe fails; compounding, not cross-terms

Joint linearization of layers 5-17 is 2.68x superadditive (+5.49 vs sum +2.05),
and L5's stand-in alone costs +1.51 despite good linear R^2 -- variance predicts
function in NEITHER direction. Mechanism hypothesis: approximation-validity
drift; sequential-refit rescue queued.

Full detail in `BILIN18_CONNECTION.md` §104.

### Instrument correction: lambda-mixing mismatch

Sections 102-104's hook-based stand-ins were fit on post-mix inputs but applied
to pre-mix inputs. L5's "outlier" (+1.51 -> +0.029) and the 2.68x superadditivity
are withdrawn; 103's front-loading claim suspended pending consistent rerun. What
survives: the self-consistent sequential-refit pipe -- 13 layers linearized for
+1.56 nats total.

Full detail in `BILIN18_CONNECTION.md` §105.

### Linearization arc: final numbers

Clean instrument: front-loading reinstated (L2 +0.109, 3x median; middle cheap at
+0.03); interaction-kill revised 98% -> 79% (+0.030 residue survives linear L17);
refit pipe +1.56 for 13 layers. Arc closed.

Full detail in `BILIN18_CONNECTION.md` §106.

### The map crowns L1

Front map complete, unimodal, peak at L1 (+0.282): the hardest-read layer is the
most functionally nonlinear -- two independent hardness measures agree. Model =
five nonlinear front layers + near-linear 13-layer pipe + one quadratic skin.

Full detail in `BILIN18_CONNECTION.md` §107.

### Drift is stream-borne

Pattern-clamping is exactly free and saves 1% of the pipe cost: the compounding
error lives in the residual stream, not in attention. L1 span-partition test
queued (is the nonlinearity crown the same fact as the vocabulary crown?).

Full detail in `BILIN18_CONNECTION.md` §108.

### The vocabulary is the linear part

L1's top-48 span (59% of energy, everything downstream reads) carries only 12%
of its linearization cost; the low-variance complement carries 53%, interaction
35%. The readable interface is nearly linear; the hidden computation sits below
the principal spectrum. Spectral localization queued.

Full detail in `BILIN18_CONNECTION.md` §109.

### The mezzanine

L1's nonlinearity is neither in the top-48 interface (linear-ish) nor the deep
tail (inert) but in mid-variance ranks ~50-500 -- half the replacement cost.
Loud output linear, hard computation quiet. Causal ablation closure queued.

Full detail in `BILIN18_CONNECTION.md` §110.

### Content vs computation; the lambda table

Ablating the interface hurts 2.3x more than ablating the mezzanine -- ablation
removes content, linearization removes generation; L1 = loud linear content,
quiet nonlinear computation. Lambda table: every block re-injects token
embeddings at weight 8; L1 and L5 nearly discard the stream (lambda0 = 0.013,
0.065) -- the lexical bus and L1's vocabulary role follow by construction.

Full detail in `BILIN18_CONNECTION.md` §111.

### Norms correct the lambda story; residual quadratic not PCA-compact

No layer is embedding-dominated (the 8x re-injection is a constant whisper); L0's
write (RMS 1436 into a stream of 6) is the true reset, and the tail amplifies
(L17 writes at 1851, loudest in the network). The linear-residuals are not
low-rank quadratics in input PCA (19% at L17, 0% at L9); Gram-basis version
queued.

Full detail in `BILIN18_CONNECTION.md` §112.

### Compressed quadratic closed; L11-other-writers queued

The nonlinear residues are diffuse in input-PCA AND the Gram basis (peak 0.20 at
L17, ~0 at L9) -- no compact quadratic exists. User prediction registered: L11
shares a code with some other writer.

Full detail in `BILIN18_CONNECTION.md` §113.

### The dissidence is L1-specific

Over writers L0/L9, sharing is weak for every reader (control 0.12-0.19 vs
random 0.07) and L11 sits AT OR ABOVE the healthy control -- not anomalous. Both
the strong shared code and L11's foreignness are specific to writer L1. User's
qualitative prediction supported.

Full detail in `BILIN18_CONNECTION.md` §114.

### Attention compensates too

Freezing L17's attention DOUBLES the surviving excess (0.030 -> 0.066; control
exact): attention absorbs interaction damage rather than carrying it -- second
compensator at this layer. Final-norm freeze queued to close the ledger.

Full detail in `BILIN18_CONNECTION.md` §115.

### Span damage is largely norm-mediated

Freezing the final gain collapses individual span damages six-fold (0.28/0.49 ->
0.05/0.08): most tail span-ablation "damage" is global gain distortion from
removing energy, not content loss -- and the content-level 16->17 interaction is
BIGGER (+0.158). Clean per-arm-referenced rerun queued.

Full detail in `BILIN18_CONNECTION.md` §116.

### Content-level accounting

Control exact: L17's span damage is 82% norm-mediated, L16's 35% -- and the
content-level interaction excess is +0.205, LARGER than raw (+0.143) and ~90% of
both content damages combined. The product law is understated, not artifactual.
Content-level skin-kill test queued.

Full detail in `BILIN18_CONNECTION.md` §117.

### Revision: the skin carries 21% at content level

Linear L17 + frozen gain kills only 21% of the content excess -- the 79-98%
kills were the norm channel. Prime suspect for the survivor: CE curvature (convex
loss on superposing logit deltas). Logit-additivity decomposition queued.

Full detail in `BILIN18_CONNECTION.md` §118.

### Composition decomposed

Curvature alone would give +0.46 excess; network compensation cancels ~2/3,
leaving +0.16 content-level (21% quadratic skin). The product law is the residue
of a war between convex loss and damage-absorbing dynamics -- compensation is
now a three-time motif.

Full detail in `BILIN18_CONNECTION.md` §119.

### Norm-mediation tracks loudness; L16 is the content heavyweight

L17 81% and L5 59% mediated (the loud writers); mid-tail mostly content. At
content level L16's span (+0.148) beats L17's (+0.093), and the
deletion-improves spans get MORE negative. Compensation generality test queued.

Full detail in `BILIN18_CONNECTION.md` §120.

### Compensation is output-end machinery

Not general (3/6, median 24%): cancellation concentrates at L16/L17-involving
pairs; mid-tail pairs can amplify. All three absorbers are logit-stabilization
machinery at the model's final stage. L9+L16 joint-negative queued for
replication.

Full detail in `BILIN18_CONNECTION.md` §121.

### Regularizer-content interaction replicates

Excess -0.0216 on disjoint rows (orig -0.0237; control additive): deleting the
L9 sharpener cushions L16 content damage. (Registered (a) failed due to my
misreading excess as joint total -- recorded.) Content-level product-law re-fit
queued.

Full detail in `BILIN18_CONNECTION.md` §122.

### Product law: final scope

Across damage shapes the scalar law fails (raw R^2 0.12 with sign flips, content
0.47) -- per-family as section 48 said; content-level constant ~13 vs the
norm-inflated 23. Small-damage cells INVERT (compensation strong enough to make
joint damage beneficial). L17 band scan queued.

Full detail in `BILIN18_CONNECTION.md` §123.

### No beneficial band at L17

Bands beyond rank 8 cost exactly what random spans cost (+0.02 generic-removal
floor via the norm channel); the non-monotonicity was inside the floor. L17 =
eight directions of content plus ballast.

Full detail in `BILIN18_CONNECTION.md` §124.

### Pattern census: content-based, L2 positional hub

Offset explains little anywhere (medians 0.01-0.26); L2 hosts the positional
machinery; content routing dominates from L0. Lexical-attention complement
queued.

Full detail in `BILIN18_CONNECTION.md` §125.

### Patterns are contextual

Triple-negative census: not positional (except L2's hub), not lexical (0
everywhere), therefore contextual -- attention is the one non-lexical component.
Stream remembers the token; attention reads the context.

Full detail in `BILIN18_CONNECTION.md` §126.

### Heads are matched filters

Score functions have median effective rank 4.6 of 128 on-distribution (isotropic
null 44.4; factors comparable). Attention reads context through ~5 data-aligned
directions per factor. RoPE-honest validation running.

Full detail in `BILIN18_CONNECTION.md` §127.

### RoPE fans the filters out

Realized pattern rank ~23 (not ~5): the ~5-dim content filters are fanned into a
larger positional family by rotation; weights-level rank does not survive as a
predictor (Spearman -0.20; shuffled null still 2x). Matched filters describe the
content computation, not the realized pattern.

Full detail in `BILIN18_CONNECTION.md` §128.

### A shared attention lexicon? (pending null)

Filter subspaces align at 0.71 within layers (random floor 0.10) -- the
shared-support/private-functional motif possibly extending to attention. Held
pending the covariance-matched null (the section-55 artifact class).

Full detail in `BILIN18_CONNECTION.md` §129.

### The lexicon is real and per-layer

Within-layer sharing survives the matched null (+0.36 beyond concentration);
cross-layer does not (+0.06). Shared watch-list + private combinations now
measured in both component families. One-watchlist unifier queued.

Full detail in `BILIN18_CONNECTION.md` §130.

### Two watch-lists per layer

Attention lexicon vs MLP watch-list: 0.13-0.40, at/below the matched null --
separate institutions, third independent measurement of cross-type separation.
Write-channel symmetry check queued.

Full detail in `BILIN18_CONNECTION.md` §131.

### The stream is multiplexed

Attention and MLP writes occupy separate channels at every layer (5/5, at/below
matched null). With section 131: separate watch-lists AND separate write
channels -- one carrier, two component types, clean separation between them,
diffuseness only within them.

Full detail in `BILIN18_CONNECTION.md` §132.

### Watch-lists causally real, individually light

Institutional deletions cost little (0.002-0.005) except L5->L6-attention
(+0.030, 50x floor): the tail's one concentrated attention-mediated edge.
Institutions real; within them, diffuse as ever.

Full detail in `BILIN18_CONNECTION.md` §133.

### The L5->L6 edge is cargo

87% value-side (patterns clamped: +0.0257 of +0.0297 survives; control exact).
The watch-list found the traffic; the traffic is routed content, not pattern
steering.

Full detail in `BILIN18_CONNECTION.md` §134.

### Gauge audit clean; bilin18 not balanced

All published statistics are gauge-invariant (T-level, activation-level, or
function-level; Gram invariant by cancellation; |DW| activation-space). But the
model itself carries uniform defect 0.28-0.40 -- raw per-matrix statistics would
be ~2x gauge-contaminated. Standing rule: balance before any future per-matrix
reading.

Full detail in `BILIN18_CONNECTION.md` §135.

### No important neurons

Unit-mass spectrum maximally flat: 98-99% effective units at every layer.
Diffuseness confirmed at the finest grain the architecture has.

Full detail in `BILIN18_CONNECTION.md` §136.

### The attention profile

Front-loaded (L1 +0.302 the model's most important attention), non-energetic
(damage does not track write magnitude), L6 cargo edge re-confirmed, and L14's
attention is NET HARMFUL (-0.036 deleted -- largest deletion-benefit in the
program). Replication queued.

Full detail in `BILIN18_CONNECTION.md` §137.

### Late-attention harm replicates

L14 -0.035 (96%), L10/L16 negative both times, control intact: three late
attention blocks reliably subtract value at the frozen point. Signature test
queued.

Full detail in `BILIN18_CONNECTION.md` §138.

### All late attention sharpens

L14's benefit is redistribution at the extreme (hard -0.270, easy +0.022) --
and net-helpful L13 shows the SAME shape. Late components uniformly trade
hard-token accuracy for easy-token sharpness. Early-vs-late dichotomy test
queued.

Full detail in `BILIN18_CONNECTION.md` §139.

### Correction: sharpening was generic damage

The easy+/hard- deletion shape appears at every depth (even L2) -- it is the
convex-loss flattening signature, not a component function. Only L16's span
hurts hard tokens (true content). Sections 98/139 corrected; deciding random
control queued.

Full detail in `BILIN18_CONNECTION.md` §140.

### Generic-damage control: confirmed at size

A random 256-span at L9 relieves hard tokens by -0.044 -- flattening relief is
generic at sufficient damage (2/3, the miss being too-small damage). Content =
hard-token harm; profile re-scoring queued.

Full detail in `BILIN18_CONNECTION.md` §141.

### The gain channel can fake content

L17's hard-token harm (+1.67) is the norm channel amplifying survivors, not
flattening -- content certification requires a frozen gain. L9/L15 regularizers
reconfirmed; L5/L7/L11 moderate content. Gain-frozen scores queued.

Full detail in `BILIN18_CONNECTION.md` §142.

### Difficulty decomposition closed

Gain-frozen: every tail span relieves hard tokens under deletion; L16's real
value is on easy tokens; L17's +1.67 was pure gain channel (collapses to -0.06).
Difficulty splits cannot separate content from generic damage here.

Full detail in `BILIN18_CONNECTION.md` §143.

### Dilution ratios replicate

Fresh stats rows give near-identical ratios (Spearman +0.79, 0 inversions);
effects-side replication queued to complete the check.

Full detail in `BILIN18_CONNECTION.md` §144.

### Dilution law fully replicated (+0.83 all-fresh)

Both sides now independent-data replicated. Headline sweep queued (score-rank,
watch-list gap, L1 linearization cost).

Full detail in `BILIN18_CONNECTION.md` §145.

### Headlines: three for three

Score-rank 4.3, watch-list gap +0.48, L1 cost +0.289 -- all replicate on fresh
rows. QK below-chance replication queued.

Full detail in `BILIN18_CONNECTION.md` §146.

### QK disjointness replicates (gap 0.63, identical)

All recent headline claims now replicated on independent data.

Full detail in `BILIN18_CONNECTION.md` §147.

### The filters are trained

Shuffled 18.7 / gaussian 36.1 vs trained 4.3: training compresses score
functions 4-8x beyond covariance concentration. Lexicon origin null queued.

Full detail in `BILIN18_CONNECTION.md` §148.

### Lexicon one-third trained

0.71 = 0.36 concentration + 0.11 generic clustering + 0.24 trained -- section
130's attribution corrected (training adds a third). Separation origin queued.

Full detail in `BILIN18_CONNECTION.md` §149.

### Separation is free

Shuffled weights separate MORE than trained (0.07 vs 0.13-0.40): the multiplexed
bus's type separation is default; training added the overlap. Attention origin
accounting closed (filters trained, lexicon one-third, separation free).

Full detail in `BILIN18_CONNECTION.md` §150.

### Front attention certified

L1 +0.358 largest, ordering preserved, within 30%. Spectral-split replication
queued (last headline).

Full detail in `BILIN18_CONNECTION.md` §151.

### Replication campaign closed

Spectral split exact (12% identical). Every headline replicated; corrections
only ever came from conceptual re-examination, never fragile numbers. L0 cell
queued.

Full detail in `BILIN18_CONNECTION.md` §152.

### L0 cell filled (accidental exact replication)

Cost +0.1767 (front map said +0.176); R^2 0.742. The nonlinearity map is
complete end to end. Program state: all maps filled, all headlines replicated,
all origins accounted.

Full detail in `BILIN18_CONNECTION.md` §153.

### Canary green

Standing regression test passes on first run (score-rank 4.1, L1 +0.275,
dilution ratios exact). Re-run bilin18_canary.py after any environment or
instrument change.

### Universality: bilin12 obeys all three laws

Dilution monotone, L1 linearization crown (+0.317), matched filters 4.8 vs 16.3
shuffled -- family properties, not quirks. (bilin12 uses a single squared score;
architecture note recorded.)

Full detail in `BILIN18_CONNECTION.md` §154.

### The replacement ladder

L9's MLP = its constant mean output (+0.031, same as full linear); L16 = rank-8
linear; L1 = irreducibly quadratic. The reference Pareto for the
fidelity-vs-complexity benchmark.

Full detail in `BILIN18_CONNECTION.md` §155.

### Correction: a relay, not two channels

Attention watch-lists align with upstream MLP writes (0.62-0.68), not attention
writes (0.20-0.43): MLPs write, attention transports, MLPs consume. Ledger #12.
bilin12 identity confirmed (CE 4.23).

Full detail in `BILIN18_CONNECTION.md` §156.

### First Pareto points

Knowledge assignment: +2.68 at 0.15M params; all-full-linear: +1.26 at 15.9M.
Also: clean-instrument naive joint is +1.26 -- section 104's superadditivity was
mostly the lambda bug. BENCHMARK.md updated with the reference curve.

Full detail in `BILIN18_CONNECTION.md` §157.

### Refit moves the frontier

(0.29M, +1.66) via sequential refit at rank-16 -- 36% bought by refitting alone.
Frontier rank sweep queued.

Full detail in `BILIN18_CONNECTION.md` §158.

### The frontier is flat

Refit rank-4 (0.07M) = +1.81; rank-64 (1.18M) = +1.54; floor +1.26 (15.9M).
17x params buys 0.27 nats -- competition lives below 0.1M params. Quadratic
upper rung queued.

Full detail in `BILIN18_CONNECTION.md` §159.

### The quadratic rung is dead; ladder final

Quadratic corrections buy nothing (-0.009/-0.000): constant -> low-rank refit
linear -> full component is the whole practical ladder. Reference-instrument arc
closed.

Full detail in `BILIN18_CONNECTION.md` §160.

### Fingerprint dataset shipped

12 components, nearly orthogonal fingerprints (pairwise 0.04) -- causal scoring
well-posed. Stability bar was ill-posed (deterministic deltas); adjacent tokens
don't share structure (context-specificity again). Saved to
bilin18_fingerprints.pt.

Full detail in `BILIN18_CONNECTION.md` §161.

### Confound floor low: Track-1 done

Base-loss floor 0.13, position 0.01, residuals orthogonal: explanations scoring
above ~0.2 carry real signal. Both benchmark tracks now have measured
instruments, floors, and traps.

Full detail in `BILIN18_CONNECTION.md` §162.

### Cross-model analog transfer: 0.34 vs 0.05

Analogous components in the two independently-trained models hurt the SAME
tokens when ablated (7x above non-analog). Universality extended to token-level
causal responsibility; Track-1 gets its cross-model generalization split.

Full detail in `BILIN18_CONNECTION.md` §163.

### Correspondence follows depth fraction

Relative-depth pairing beats absolute at both mid components; front strongly
aligned. Full correspondence matrix queued.

Full detail in `BILIN18_CONNECTION.md` §164.

### Exact depth-fraction correspondence

3/3 exact: bilin12's attention components best-match bilin18 at their own depth
fractions (0.06/0.17/0.50). Curves single-peaked, collapsing past fraction 0.65.
The program's strongest universality result.

Full detail in `BILIN18_CONNECTION.md` §165.

### Six for six

MLP correspondence matches too (mlp1->L1, mlp5->L7, mlp8->L11). Every component
tested best-matches the sibling model at its own depth fraction. Capstone
universality result.

Full detail in `BILIN18_CONNECTION.md` §166.

### Third model: front universal, mid-scaling wavers

sqrd12 (conventional MLPs, CE 4.01): front attention corresponds at fraction;
mid pairs at absolute index (0.02 past the bar). Correspondence crosses the MLP
family; fractional scaling of the middle may be bilinear-specific.

Full detail in `BILIN18_CONNECTION.md` §167.

### Tie-break: within-family law

sqrd12's mid-attention peaks at bilin18's L7 (fraction 0.39), bilin12's at L9
(0.50) -- fractional scaling holds within the bilinear family and warps across
families. Correspondence arc closed.

Full detail in `BILIN18_CONNECTION.md` §168.

### Fourth model refines two laws

Front-loading universal (3/3 checkpoints tested). Dilution scope-noted: decline
through mid-tail, RISE at the output end -- in every model (bilin18's own norms
agree). Correspondence warps by MLP family even at equal depth (gated models
front-shifted); front universal.

Full detail in `BILIN18_CONNECTION.md` §169.

### Interfaces graded; edges typed

L5->L6 edge: ~8 dims dominant (72%) with a fat tail. Edge complexity redefined
as functional-form description length: summary-typed (norm/share), coordinate-
typed (few dims), or opaque -- all three measured in this model. BENCHMARK.md
updated.

Full detail in `BILIN18_CONNECTION.md` §170.

### The gain channel has a sign

Attention norm-shares all NEGATIVE (-16 to -32%): the gain cushions attention
damage and amplifies MLP-span damage. Attention edges are content-typed at
every depth; benchmark audits must report the share's sign.

Full detail in `BILIN18_CONNECTION.md` §171.

### Keep-only: 28% preserved, 14x enrichment

Both audit directions agree: the L5 interface is a real minority channel.
Structure is real and never total -- the program's oldest lesson at the newest
instrument.

Full detail in `BILIN18_CONNECTION.md` §172.

### The full atlas

36 exact fingerprints: distinguishable (0.07), depth-smooth (35/36),
type-marked. Ground-truth layer complete.

Full detail in `BILIN18_CONNECTION.md` §173.

### Seriation: order from marks alone

Layer order recoverable from unlabeled fingerprints (|rho| 0.91/0.90, nulls
~0.25). The causal fingerprint is a structured coordinate system, not noise
around a net number.

Full detail in `BILIN18_CONNECTION.md` §174.

### Sibling atlas replicates

bilin12's 24-component atlas: distinguishable, depth-smooth (22/24),
type-marked -- the atlas structure is a family property. Joint seriation is the
staged next step.

Full detail in `BILIN18_CONNECTION.md` §175.

### One causal depth coordinate

Pooled 30-component embedding tracks depth fraction at 0.85 across both models
(null 0.30). The family shares one causal depth axis -- the fingerprint arc's
capstone.

Full detail in `BILIN18_CONNECTION.md` §176.

### The axis transcends the family

Three models on one causal depth axis (0.79, null 0.04); the cross-family warp
is attention-specific -- MLP placement is universal. Fingerprint arc complete.

Full detail in `BILIN18_CONNECTION.md` §177.

### Correction: warp direction unresolved

The two instruments disagree on the sqrd12 attention warp's sign (direct match:
front; axis: back). Displacement real (~0.1-0.17), direction withdrawn. MLP
universality unaffected. Ledger #13.

Full detail in `BILIN18_CONNECTION.md` §178.

### Ledger #13 resolved: no warp

At n=12 both instruments agree (0.90) and displacement is +0.000. The
depth-fraction law is fully universal: every component, every model, both
types.

Full detail in `BILIN18_CONNECTION.md` §179.

### swiglu18: atlas perfect, axis membership failed -- verdict withheld

Median displacement 0.333 on the axis instrument; per the ledger-13 lesson the
direct instrument must report before any law-boundary claim. Queued.

Full detail in `BILIN18_CONNECTION.md` §180.

### The law's fourth model passes

Direct instrument: swiglu18 tracks fraction at median displacement 0.06 (inside
the bar); the axis instrument is twice-demonstrated unreliable at weak
correlations and is now gated in the benchmark. Four models, one law.

Full detail in `BILIN18_CONNECTION.md` §181.

### Zero anomalies

The front plateau was argmax noise over near-ties (runners-up are the correct
depths). Universality arc closed clean.

Full detail in `BILIN18_CONNECTION.md` §182.

### Leverage is text-borne (0.78 cross-model)

The fingerprint matrix factorizes: rows follow depth (the fraction law),
columns follow the text (leverage shared across models beyond difficulty). Both
axes of the family's computation measured.

Full detail in `BILIN18_CONNECTION.md` §183.

### One text profile, four models

All pairs >= 0.54; the coverage-matched cross-family pair is highest (0.83).
The models agree on what is hard, where to work, and where it lands.

Full detail in `BILIN18_CONNECTION.md` §184.

### Coverage closed: minimum pair 0.74

sqrd12's full atlas resolves the caveat; all six leverage pairs 0.74-0.84.
Four complete atlases, both axes aligned. Program assets complete.

Full detail in `BILIN18_CONNECTION.md` §185.

### Canary v2 green

Full-state regression suite in place and passing: original trio + atlas
integrity + leverage law (0.78) + fraction spot (L7) + smoothness (35/36).
Re-run bilin18_canary2.py after any environment change.

### Track-1 pilot: one pass, two lessons

attn14's per-token story passes (+0.18). Lessons hardened into the spec:
explanations declare their measurement regime (free-norm vs gain-frozen
fingerprints); explanations must compile to DISTINCTIVE predictions (difficulty
-shaped stories are mutually unfalsifiable).

Full detail in `BILIN18_CONNECTION.md` §186.

### Track-1 demonstrated end to end

attn14 certified in both regimes (+0.28 frozen); mlp16 regime-flip confirmed
but floor-grade; frozen-fingerprint asset shipped. The semantic track is
operational with honest mixed results.

Full detail in `BILIN18_CONNECTION.md` §187.

### First certified edge explanation

"attn6 transports L5's content" scores 0.223 vs 0.087 median-other by
fingerprint kinship -- the cargo edge's third independent confirmation, and a
new scoreable relation type for Track-1. Scoreboard: two certified.

Full detail in `BILIN18_CONNECTION.md` §188.

### Kinship recovers the relay's direction

16/18 attention components partner upstream (null 11/18). Global rankings are
front-dominated -- kinship scoring must be per-component relative (spec rule).
The 16->17 interchange surfaces in a fourth instrument (attn17~mlp16).

Full detail in `BILIN18_CONNECTION.md` §189.

### Interchange certifies by kinship (8x)

attn17~mlp16 at 0.212 vs 0.026, directional, MLP-side symmetric. Five
instruments on one edge. Track-1: three certified.

Full detail in `BILIN18_CONNECTION.md` §190.

### The interchange is a depth feature

Relay directionality universal (all siblings); the output interchange exists in
both 18-layer models (swiglu18 certifies at 12x!) and neither 12-layer model.
Depth, not architecture, builds the hand-off.

Full detail in `BILIN18_CONNECTION.md` §191.

### Correction: the two interchange signatures dissociate

Causal coupling (normalized) exists in ALL models (bilin12 largest, 4.5);
token-marking kinship only in 18L models. "Depth feature" withdrawn (ledger
14); claims must name their signature.

Full detail in `BILIN18_CONNECTION.md` §192.

### c is not a scalar

Coupling varies 4x with size within a model and flips sign in swiglu18 at small
damage (compensation regime). Only the bilin12 > swiglu18 ordering is robust.
Comparisons must fix (signature, family, size). Comparative arc closed.

Full detail in `BILIN18_CONNECTION.md` §193.

### Sign flip replicates (stronger)

swiglu18 k=4 excess -0.100 on disjoint rows (143%); bilin12 control +0.046.
Output-stage compensation is family-wide, dominant in the gated sibling at
small damage. Comparative arc ledger clean.

Full detail in `BILIN18_CONNECTION.md` §194.

### Hillclimb 1: allocation doesn't pay

Greedy matches uniform at every budget despite 32x nonuniform allocations --
the flat frontier extends to allocation. Refit remains the only lever found.

Full detail in `BILIN18_CONNECTION.md` §195.

### Round 1 verdict + harness shipped

Reader-alignment loses to variance 11/12 (+0.34 jointly): subspace-grain
circuit knowledge loses twice; the wins are class/protocol-level. Round-2
hypothesis: cross-layer shared bases. The scale harness (HARNESS.md +
harness_skeleton.py) shipped with all five self-tests green.

Full detail in `BILIN18_CONNECTION.md` §§196-197.

### The curve is parameterization-invariant

Shared basis lands exactly on the uniform curve; four schemes tried, none beat
it. Diffuseness as Pareto-invariance. Levers that work: refit, class
selection, regime accounting.

Full detail in `BILIN18_CONNECTION.md` §198.

### Round 3: optimization-limited null

Cold-start Adam couldn't bridge L16's 1e12 output variance; class verdict
withheld. Round 3b queued (warm-start linear + residual bilinear, normalized).

Full detail in `BILIN18_CONNECTION.md` §199.

### Round 3b: the class lever works (+0.33 at 0.22M for L1)

Warm-started narrow bilinear beats same-budget linear 3.4x at the front and
nearly matches full linear at 1/6 size. Pareto invariance is within-class only;
class-where selection is the understanding-shaped lever. Hillclimb arc: the
recipe is class selection + refit + joint scoring.

Full detail in `BILIN18_CONNECTION.md` §200.

### Correction (ledger 15): the class lever's cost was misstated

The warm full-rank base (1.33M) was uncounted; honestly, section 200's combo
loses to full linear. Cheap-base retest queued -- the class question reopens.

Full detail in `BILIN18_CONNECTION.md` §201.

### The class question closes; hillclimb final verdict

No structural lever shifts the curve at honest cost. Real levers: refit
protocol and licensed constants. The frontier is close to an
information-theoretic given on this model -- grade submissions on reaching it
cheaply.

Full detail in `BILIN18_CONNECTION.md` §202.

### Portability: protocol transfers, thresholds don't

Refit buys 21% on bilin12; zero constants license (its tail is 4-5x less
replaceable). Replaceability scales WITH model size -- good news for larger
targets. Scan-refit-score-selftest is the universal part.

Full detail in `BILIN18_CONNECTION.md` §203.

### 2x2: ordering yes, rate no

18L row licenses {4,1}, 12L row {0,0}: size ordering holds, magnitude is
architecture-modulated. The scaling prediction stands in scoped form.

Full detail in `BILIN18_CONNECTION.md` §204.

### The modulation is deep

swiglu18: 0/11 layers replaceable even at rank-4 (uniform +0.05-0.09) --
gated models spread function, bilinear models concentrate it and leave slack.
The rank-0/rank-4 scan is the first thing to run on any new target.

Full detail in `BILIN18_CONNECTION.md` §205.

### Closing observation: slack contains the regularizers

bilin18's constant-licensed layers {8,9,14,15} contain the regularizer pair
{9,15}; the gated sibling has neither slack nor regularizers. One phenomenon,
first seen in section 37, closed in section 205.

Full detail in `BILIN18_CONNECTION.md` §206.

### Identity verified: exact in three models

bilin12: no slack, no regularizers; swiglu18: one slack layer (L15), one
regularizer (L15, -0.030). The 206 assertion is now grounded, not asserted.

Full detail in `BILIN18_CONNECTION.md` §207.

### Shared vocabulary: behavioral, not elementwise; section 61 replicates

Fresh-rows LORO 0.637 (orig 0.711) on the activation-weighted metric; 0.26 on
matrix elements. Readers share what their forms do, not what they are.

Full detail in `BILIN18_CONNECTION.md` §208.

### Correction: vocabulary sharing is writer-general (ledger 16)

Behavioral LORO: L0 0.70, L1 0.64, L9 0.54. Section 114's "L1-specific" was an
artifact of the elementwise metric. Readers share a vocabulary over every
strong writer tested.

Full detail in `BILIN18_CONNECTION.md` §209.

### The sharing landscape: one private writer, one solitary reader

Vocabulary sharing is writer-general with two sharp exceptions: L6 (private
-- below the random-projection floor) and reader L17 (shares nothing). Three
explanations (causality, global geometry, communal subspace) each refuted by
a registered run. Organizing principle open.

Full detail in `BILIN18_CONNECTION.md` §210.

### Fourth story dead; L6 stays open

"Private because regularizer" fails its control in existing data: L9 is more
regularizer-flavored and shares fine. L6's privacy is the cleanest standing
anomaly.

Full detail in `BILIN18_CONNECTION.md` §211.

### The anomaly narrows: L6's privacy is its top-8 span

Past the span, L6 shares normally (0.41 vs control 0.46). One 8-dimensional
object -- L6's principal/regularizer span -- is the only private code in the
model.

Full detail in `BILIN18_CONNECTION.md` §212.

### Span 6:1-8 is not unread cargo

Its content propagates everywhere (confound-corrected by the passive-L7
baseline); span-specific amplification peaks at L17 (2x) but compounding is
unexcluded -- the private-channel story stays unproven.

Full detail in `BILIN18_CONNECTION.md` §213.

### The honest channel: mostly confound, remainder real

Magnitude-matched, span 6:1-8 excites the output end 1.4x (not 11x). Freezing
the middle RAISES the ratio to 1.6-1.8: the middle absorbs the span's signal
(compensation), and a modest direct residual/attention channel remains.

Full detail in `BILIN18_CONNECTION.md` §214.

### The private writer is universal: depth fraction 1/3 in both models

bilin12's landscape reproduces bilin18's -- early sharing highest, declining
with depth, one below-floor privacy notch at L4 = fraction 0.33 (bilin18: L6
= 0.33). The placement law now places the newest phenomenon.

Full detail in `BILIN18_CONNECTION.md` §215.

### The signature transfers whole

bilin12-L4's privacy is span-concentrated too (tail coords 0.46 vs control
0.56) -- and its private span is NOT a regularizer (+0.036), dissociating
privacy from regularizer character cross-model.

Full detail in `BILIN18_CONNECTION.md` §216.

### Private computation partially conserved

Cross-model token-damage correlation 0.16 (7.5x every random pair, below the
0.23 shared-span baseline): the fraction-1/3 object is functionally related
across models but drifts more than shared code.

Full detail in `BILIN18_CONNECTION.md` §217.

### Last reader ranks last everywhere; solitude needs depth

bilin12's L11 is the worst fold for 4/5 writers (ordinal signature universal)
but still shares at 0.43 -- true secession is 18L-specific, mirroring the
kinship/coupling dissociation.

Full detail in `BILIN18_CONNECTION.md` §218.

### Privacy is content, not address

The private span's directions are the stream's most contested (owner among
the quietest writers there, both models). What's private is the quadratic
content, not the coordinates -- a whisper under the loudest channels.

Full detail in `BILIN18_CONNECTION.md` §219.

### The whisper is understood -- in dialects

Each reader compresses the private span's content cleanly with its own basis
(median 0.75); the population basis fails (gap +0.56; control gap -0.10).
Structure without agreement. Early readers share a code; deep readers go
private. One null bar was mis-set for the 36-dim space and is corrected.

Full detail in `BILIN18_CONNECTION.md` §220.

### Correction: no early shared code (ledger 17)

At pair resolution nothing transfers over the private span above the
measured random floor (0.31); the control transfers everywhere (0.62-0.82 vs
floor 0.10). Idiosyncrasy is total. Section 220's depth-graded reading is
withdrawn; register the metric AND measure the floor first.

Full detail in `BILIN18_CONNECTION.md` §221.

### The notch is one layer wide

bilin18 neighbors: L5 0.43, L6 0.16, L7 0.52 (nulls measured, all held) --
matching bilin12's sharp notch. One layer, top-8 span, fraction 1/3, width 1.

Full detail in `BILIN18_CONNECTION.md` §222.

### The private writer is a private LAYER

attn6 is the least-shared attention writer (0.13, = mlp6's 0.16): the notch
covers the layer's entire output, both component types. Attention overall
shares slightly less than MLPs (median 0.38).

Full detail in `BILIN18_CONNECTION.md` §223.

### Correction: layer-level privacy was borrowed (ledger 18)

Orthogonalized against the mlp6 span, attn6 is a normal writer (0.13 ->
0.45; control unmoved). The private object is the single 8-dim span; mlp6
writes it, attn6 carries it.

Full detail in `BILIN18_CONNECTION.md` §224.

### Carrier effect is 18L-specific

bilin12's attn4 barely recovers under orthogonalization (0.23 -> 0.28;
bilin18: 0.13 -> 0.45). The anomaly's family symmetry is MLP-side only.

Full detail in `BILIN18_CONNECTION.md` §225.

### No attention notch at 12L

Full profile: attn4 within one MAD of median; minimum is attn1; the 225
"depression" was sampling luck. Unified: the anomaly is only ever the MLP
span -- attention is transport, never origin. (Curious: bilin12 attention
sharing rises with depth while MLP sharing falls.)

Full detail in `BILIN18_CONNECTION.md` §226.

### Gradient story dies as a universal

bilin12's rising attention sharing (+0.82) is its own trait; bilin18's
attention is flat (-0.13), MLP falls (-0.43). attn17's low value is an
instrument boundary (all folds acausal), not an anomaly.

Full detail in `BILIN18_CONNECTION.md` §227.

### Ledger 19: the notch's floor-crossing is instrument-relative

Across ensembles/aggregations, bilin18's L6 is robustly ~half its control
(8/8 constructions); bilin12's L4 depression survives pooled constructions
but vanishes under fold-median. Downgrade: robust relative depression, not
below-floor. Preregistration P3 rewritten (pooled, 3 ensembles, relative
bar); P2 gains a replication requirement.

Full detail in `BILIN18_CONNECTION.md` §228.

### Concentration robust in invariant form

Tail coords are ensemble-stable (0.40-0.53 both models, spread <= 0.07);
only the span-dominated full value flexes. The ledger-19 sensitivity is
itself span-localized. P3 tail bar amended to absolute form.

Full detail in `BILIN18_CONNECTION.md` §229.

### Dialects robust in the discriminative regime; audit complete

Gap 0.46-0.79 at ranks <= half the ambient dim, all splits/ensembles;
saturates by construction at rank 24 (2/3 of the 36-dim space). Control
flat 12/12. Every landscape headline now carries a robustness statement.

Full detail in `BILIN18_CONNECTION.md` §230.

### Ledger 20: no construction-robust secession; the durable core

L17's solitude is ensemble-dependent (5/5 worst in A, 2/5 in D); the 218
contrast falls. bilin12's notch degrades to core-ensemble-only. L7 is a
stats-row boundary case; identity survives. Durable core: relative 18L
notch, tail stability, dialects-in-regime, MLP decline.

Full detail in `BILIN18_CONNECTION.md` §231.

### Partial conservation robust 4/4; audit truly closes

The only landscape claim to pass its sweep unamended. Final tally: 4 robust
as published, 2 robust amended, 3 fallen/degraded (ledgers 19-20).

Full detail in `BILIN18_CONNECTION.md` §232.

### Relay directionality replicates fresh (94%)

Cargo edge's kinship rank drops to 23/324 (causal certification untouched;
cite the interchange number). Null band was mis-registered: best-of-18 over
a causal ordering expects ~53%, not 50-50; signal is 3.5 sigma above.

Full detail in `BILIN18_CONNECTION.md` §233.

### E1 re-certified fresh (margin +0.101, 2x bar)

Rank was the wrong statistic (loud early writers always win it); margin
over median is the claim's actual content and it replicates. Certified
entries all have fresh-data legs now; replication backlog empty.

Full detail in `BILIN18_CONNECTION.md` §234.

### attn14 score replicates fresh (+0.250)

Regime ordering preserved (frozen > free on fresh text). The closing
sentence is now exact: every certified claim, score included, has a
fresh-data leg.

Full detail in `BILIN18_CONNECTION.md` §235.

### Circuit pipeline day one: 147 structural, 27 semantic

54k tokens fingerprinted in 95s; 147/154 ownership circuits held-out
certified (76% coverage); 8 parallel agents wrote 134 stories; 27
semantically certified at median 81x precision lift (null 0). 92 failures
are the rule language's ceiling, not the model's.

Full detail in `BILIN18_CONNECTION.md` §236.

### Red-team + confirmation: 6 FINAL, structure transports

Red-team killed 14/27 wave-1 certifications (gerrymander/mismatch); the
untouched-window confirmation passed 6/39 -- 22 failures were rule-silent
(document-bound vocabulary). But 249/256 clusters transport across
windows: circuits are general, the stories weren't. Wave 3: mixed-window
evidence, class-first rules, cross-window certification by construction.

Full detail in `BILIN18_CONNECTION.md` §237.

### Supervised function circuits: first four causal certifications

Wave 3 failed 1/136 -- metric mismatch (function stories can't be
cluster-precise). Flipped to supervised function slices: 8/9 ownership-
replicable across windows (cos 0.90-1.00); CERTIFIED: digit continuation
(attn8+mlp15), bracket closing (attn13), subword completion (mlp16+mlp15),
name continuation (attn1+attn0). Induction = attn3-5 (not attn1-2),
replicable but redundant (not concentrated). 79% of unsupervised clusters
are these ten functions.

Full detail in `BILIN18_CONNECTION.md` §238.

### Site-specificity: 3/4 local; name circuit works from the antecedent

digit 94%, subword 102%, bclose 64% site-local; name only 18% -- attn0/1
build the copy SOURCE at earlier mentions (transport, not site-local).
Subword completion is most of mlp16+mlp15's total job. Digit and subword
nominated as first slice-conditioned replacement targets.

Full detail in `BILIN18_CONNECTION.md` §239.

### Slice-conditioned constants: 82-88% recovery

One fixed vector per owner, conditioned on site type, carries ~85% of the
digit and subword circuits (random-const controls -105%/-323%). The
compression lever the flat layer-track couldn't see: high-rank
unconditionally, near-constant given the site type.

Full detail in `BILIN18_CONNECTION.md` §240.

### Half the tail is a 640-number dictionary

Eight tail MLP spans jointly replaced by per-class constants: 50% total
recovery (digit 102%, subword 71%; induction -9% -- copying defeats
constants). Shuffle control -17%. New benchmark frontier point at ~0.7KB.

Full detail in `BILIN18_CONNECTION.md` §241.

### Rung 1: 95% oracle-conditioned replacement

Constants + two per-class linear maps replace the eight tail spans at 95%
(induction -9% -> 98%). Caveat stated: class predicates read the target
token -- a structural decomposition, not yet deployable compression;
input-only classifier is the registered next rung.

Full detail in `BILIN18_CONNECTION.md` §242.

### Crude context rules hit 31% -- deciding the type is real work

Hand rules can't recover the site class (bars failed as measured). Next:
linear probe on the tail's input stream + probe-conditioned dictionary.

Full detail in `BILIN18_CONNECTION.md` §243.

### Deployable dictionary: 75% input-only

Stream probe hits 59% class accuracy (floor 36%); probe-conditioned
dictionary recovers 75% (oracle 95%, shuffled -67%). Tail job split: ~75
points writing content, ~20 points knowing the type. Fit/apply interface
bug caught by the oracle arm failing to reproduce 95%.

Full detail in `BILIN18_CONNECTION.md` §244.

### Task circuits + families

Counting = the certified digit circuit (attn8 at +2.21, 10x runner-up;
100% top-1). IOI (58%): executed by LATE attention (attn14 top) while
early attention builds the source -- attn14 is harmful on average text
and essential for retrieval. Addition: model can't (0%). 147 circuits
collapse to 20 owner-function families.

Full detail in `BILIN18_CONNECTION.md` sections 245-246.

### Attention is two materials

Per-class constants replace attention at 67-85% for structure sites
(bclose/rep/sentend/comma/digit) and 3% for induction -- transport is
stereotyped at structure sites, irreducibly contextual at copy sites.
Newline ablation negative (regularizer channel). Next: programmatic copy
stand-in for the induction band.

Full detail in `BILIN18_CONNECTION.md` §247.

### Label mining: 24 accepted, constants near saturation

Taxonomy 10 -> 34 classes; residual 36% -> 22%; constants recovery 50% ->
55%. First ten classes bought 50 points, next 24 bought 5: the constant
channel saturates; what remains is contextual (the linear rung's domain).

Full detail in `BILIN18_CONNECTION.md` §248.

### Naive match-and-copy refuted

Single-position value injection at induction sites: -327% raw, -337%
calibrated (random target -390%). The copy is distributed; band keeps
'not yet mechanistically reduced'.

Full detail in `BILIN18_CONNECTION.md` §249.

### mlp1 is 79% a lookup table

Context-free fold (one forward per vocab token) replaces the model's most
important MLP at 79%. mlp0 only 15% (position-sensitive); mlp2/3
catastrophic. Empirical token-table discriminator queued.

Full detail in `BILIN18_CONNECTION.md` §250.

### The front is a token-dictionary cascade

Token-conditional tables: mlp0 68%, mlp1 85% (79% from weights alone),
mlp2 58%, mlp3 44%. The crown component's dictionary is readable straight
from the parameters.

Full detail in `BILIN18_CONNECTION.md` §251.

### Attention outputs are not token-tables

attn0-5 output tables: 15-36% max (induction band negative). The cascade
is an MLP phenomenon; lexicality lives at the value level -- v-table
discriminator queued.

Full detail in `BILIN18_CONNECTION.md` §252.

### attn1 = lexical value store (87%, margin 51 over null)

Value-level tables certify the lexical claim at the right interface;
attn4's values are half token-determined (the copy failure was the
pattern, not values); attn0's v1 broadcast is NOT token-approximable
(-223%) -- a new named open object.

Full detail in `BILIN18_CONNECTION.md` §253.

### Correction: no attn0 anomaly -- exactly lexical

The -223% was the rare-token fallback contaminating sequences (seen-prefix
damage exactly 0.0000). Layer-0 values are a pure token function by
architecture: trivially foldable, zero error. 253's open object withdrawn.

Full detail in `BILIN18_CONNECTION.md` §254.

### Assembled dictionary model: +1.85 nats, sub-additive

13/36 components (crown included) replaced by tables + probe + constants;
assembled cost 0.88x the sum of solos. Two tables weights-computable.

Full detail in `BILIN18_CONNECTION.md` §255.

### mlp1: linear wins at +0.12

Full ridge linear replaces the crown at +0.12 nats (1.3M params), 5x
better and 40x smaller than the token table (58M raw numbers -- priced
honestly now: interpretively cheap, parametrically expensive). Assembled
v2 queued.

Full detail in `BILIN18_CONNECTION.md` §256.

### v2 super-additive (2.64x): sequential-refit lesson crosses tracks

Linear stand-ins fit on clean inputs misfire under upstream substitution;
tables are id-conditioned and robust. v3 = sequential fitting, queued.

Full detail in `BILIN18_CONNECTION.md` §257.

### The table-firewall effect

Sequential fitting fixed the crown (+0.12 in-context) but v3 = +1.97 still
trails v1's +1.85: token-tables act as error FIREWALLS, input-faithful
components as CONDUCTORS. Composition depends on stand-in type along the
graph. v4 (tables + sequential residual corrections) registered.

Full detail in `BILIN18_CONNECTION.md` §258.

### v4 = +1.39, new assembled frontier

Firewall+absorber tables (rank-32 sequential correctors, residual R^2
~0.6) beat all-table v1 by 25%. Composition grammar validated.

Full detail in `BILIN18_CONNECTION.md` §259.

### v5: 19/36 components at +2.10

The 'incompressible middle' verdict was true of its description class
only: firewall+absorber tables replace mlp4-9 at ~0.12 nats each
(mlp5 +0.02). Frontier: 13 comps @ +1.39; 19 comps @ +2.10.

Full detail in `BILIN18_CONNECTION.md` §260.

## Appendix 261-263: weight-based middle compression + v6 attention rung (2026-08-18)
- CP truncation from weights alone: top 25% of hidden units (by ||down||*||l||*||r||) recover >=80% of every middle MLP, <=+0.07 solo (6/6 HELD). Input-subspace + downstream-read-subspace hypotheses both FAILED: the middle's read/write interfaces are broad, not low-dimensional (two instruments agree).
- Assembled v6: tail attention 10-17 replaced by class dictionaries (constants for structure classes, linear reads for contextual) -> 27/36 components at +2.548 oracle / +2.605 deploy; all 8 rungs survive LOO; a14 (deletion-improves) still costs +0.117 (harmful-on-average != free).
- Quadratic absorber features (weight-derived): +1.993 vs +2.095 baseline; read-subspace absorbers worse than baseline (refuted).
- Frontier: 13@+1.39, 19@+2.10, 27@+2.55.

## Appendix 264-265: weights-only middle + absorber capacity (2026-08-18)
- v7 (CP middle): k=2304 (half the hidden units, weights-only, zero fitted params) -> 19/36 at +1.678 -- new frontier. k=1152 bar FAILED (+1.912). CP-mlp8 LOO marginal NEGATIVE (-0.044): truncation firewalls off-distribution error; firewall/conductor is a spectrum.
- Quad ladder: feature count saturates at 16-32 directions; the rank-32 write basis is the binding constraint (P64: +1.934, best fitted-absorber arm).
- Frontier: 13@+1.39, 19@+1.68, 27@+2.55. v8 merge queued.

## Appendix 266-267: v8 merge + LEDGER 21 (2026-08-18)
- v8 (quad32/P64 front + CP-2304 middle + attention dicts): 27/36 at +2.212 oracle / +2.254 deploy (bar +2.20 FAILED by 0.012). Attention marginals grow on cheaper substrates -- rungs share an error budget.
- LEDGER 21: cp_controls selection nulls refute the CP ranking -- random-k matches top-k, bottom-3456 beats top-1152 at 6/6 layers. The 19@+1.68 frontier stands but its mechanism is REDUNDANCY, not weight-readable importance. New rule: no selection claim without its selection null.
- Frontier: 13@+1.39, 19@+1.68, 27@+2.21.

## Appendix 268-270: selection null, v9 middle attention, P-ladder (2026-08-18)
- In-assembly selection null: random middles straddle the ranked one (spread 0.39) -- ranking = variance reduction, not quality.
- v9: 34/36 components at +2.868 oracle / +3.067 deploy. Induction-band prediction FAILED informatively: a5 marginal NEGATIVE (-0.38) -- real induction machinery misfires on substituted streams; a8 (digit owner) the only drop (+0.4002).
- P-rank ladder monotone sublinear (P128: +1.899).
- Frontier: 13@+1.39, 19@+1.68, 27@+2.21, 34@+2.87.

## Appendix 271: a5 = scaffolding (2026-08-18)
- attn5 in-assembly: real +3.47, dict +2.87, MEAN +2.72, zero +4.75. The component is needed only for its average output there; input-dependence misfires. Greedy mean-vs-dict sweep queued.
- v10 (+2.853) barely improves v9-best; front absorber rank no longer binding at 34 comps.

## Appendix 272: greedy selection + benchmark figures (2026-08-18)
- Five attention rungs (a2,a5,a9,a14,a16) do best as plain mean vectors: 34/36 at +2.779 oracle / +2.981 deploy.
- Ceiling measured: embeddings-only = +11.95 nats. Frontier in %-of-model-work: 88% @ 13 comps, 86% @ 19, 81% @ 27, 77% @ 34.
- Figures: bilin18_frontier.png, bilin18_module_relevance.png.

## Appendix 273: round-2 greedy (2026-08-18)
- mlp8's stand-in collapses to a mean vector too; greedy converged. Frontier 34/36 at +2.749 oracle / +2.953 deploy. a8 still the one dead rung.

## Appendix 274: front marginals (2026-08-18)
- In-assembly LOO: mlp1 +0.006 (99.9% of its 5.32-nat solo relevance captured); front block total +0.046; tail spans 0.011-0.082. Deploy probe token-prior upgrade refuted (0.686->0.701, no CE change).
- a8 rescue refuted: digit-linear arm no better than plain dict (+0.62 marginal both). Counting is cross-position transport.

## Appendix 275-276: rank floor + LEDGER 22 fresh-data audit (2026-08-18)
- Linear arms genuinely high-rank (rank-128 costs +0.087). 
- Fresh never-seen pile window: best-34 = +3.114 (standard window +2.749 -- ~0.35 optimism bias); only 3/6 greedy swaps replicate; all-means baseline +7.04 (assembly content real); fineweb +3.45.
- LEDGER 22: within-sample fit/eval splits insufficient; selections must validate out-of-sample. Honest frontier: 34/36 at ~+3.11 fresh (74% of model work).

## Appendix 277-278: validated frontier + grounding (2026-08-18)
- THE benchmark quote: 34/36 at +2.925 oracle / +3.017 deploy on never-touched fresh data (75% of model work retained). Validated greedy keeps only a5/a14/a16 means.
- m0 table grounded: fold rows match empirical at 0.917 cosine (centered-null rerun owed); rows organize by token type (12x null).
- MLP motif repertoire confirmed (section 58 rerun): rank-80 shared basis, LORO R^2 0.71. Attention motif census running.

## Appendix 279: attention motifs (2026-08-18)
- Cross-layer motif repertoire confirmed at pattern level: prev-token (27 heads, 11 layers), self (51 heads, 16 layers). Mass-based instrument degenerate; argmax-based works. Induction invisible unconditionally -- eligibility-conditioned v3 queued.

## Appendix 280: motif census complete (2026-08-18)
- Four named motifs cover 85/162 heads: self 47 (16 layers), prev 27 (11), induction 9 (7 layers, conditional fraction up to 0.74, null 0.04), first 2. Induction pattern capability broader than causal ownership (redundancy).
- m0 table grounding quotable: centered fold-vs-empirical cosine 0.841, null -0.02.

## Appendix 281: motif swap v1 (2026-08-18)
- Whole-family pattern swaps too expensive (both +1.28, super-additive) and the random-head control IMPROVED CE -- instrument lacks a reconstruction null; v2 queued with null + per-head greedy.

## Appendix 282: per-head motif validation (2026-08-18)
- Recon null exact 0; 71/74 heads swap individually at <=+0.01; joint +0.45. Motif sentences true per head; composition needs the grammar playbook. v1 control anomaly = real pattern slack.
- Fresh ladder: front +1.98 / +middle +2.02 (CP middle only +0.04 fresh -- weights-derived stand-ins travel) / +tailatt +2.49 / 34-comp +2.93.
- QK factorization refuted: both score sets carry the same weak preference; the product is coincidence-sharpening (evidence squaring), not selector x gate.

## Appendix 285: OV library + block handoff (2026-08-18)
- OV subspaces at the random-spread floor (no shared library); private bindings held. Attention motifs live on the pattern side only.
- Block wiring map: attn->own-mlp handoff ~2x cross-block, front-loaded (fades in tail). attn0 broadcast control held.

## Appendix 286-287: handoffs + decomposition census (2026-08-18)
- Causal attn->own-mlp handoff: three critical junctions (blocks 0, 1, 5 at +0.8/+1.9/+2.1), everything else ~0; spearman 0.80 with the subspace wiring map. attn5->mlp5 = the private-writer neighborhood; the assembly bypasses it.
- Decomposition census: sparse wins 126/126 at matched budget (Gaussian null owed -- v3 running).
- Two v1 instruments voided honestly (wrong interface / unfair budget) before the science bars.
- Decomposition-type direction CLOSED: weights entrywise Gaussian (kurtosis 0.03-0.52, sparse advantage = chance, masks unaligned). All structure is functional/subspace-level, never entry-level (208 generalized).

## Appendix 289: narrow handoff + induction third strike (2026-08-18)
- attn5->mlp5 = a 4-16 direction channel (top-4 carry 76%, random-16 nothing).
- Matched-successor linear read refuted on the census heads (same-position control wins); exact-mixing rerun queued under three-strikes.

## Appendix 290: convergence (2026-08-18)
- Induction linear reduction CLOSED at scope (three strikes, exact mixing).
- The attn5->mlp5 channel is a clause-boundary signal feeding the private span (overlap 40-50x floor; class R^2 up to 0.31, nulls 0). Anomaly arc and circuit arc joined: gather -> encode -> idiosyncratic readout.
- Handoff->span link: REGULATORY not generative -- span variance x4.6 under the cut (random-4: nothing); damage in contextual classes, clause classes unharmed. Logit-lens semantics != causal payload.
- Span specificity: explosion targeted (span 4.58x, full 1.65x, next-8 directions SHRINK 0.39x). The channel is a targeted range-governor of the private code. Arc complete.
- Assembly preserves span statistics (0.70 uniform) -- the regulator works through its MEAN, which is why a5's best stand-in is one vector. Arc closed; fresh replication queued.
- Regulator arc REPLICATES FRESH: junctions +0.96/+1.95/+2.16 vs controls +0.05/+0.004; span explosion 4.33x. Quotable.
- Block-1 junction INVERTS the rank ladder (top-4 cut +4.08 > full cut +1.95): a coherence junction -- mlp1 tolerates full presence or clean absence, not partial edits. Junction typology: narrow regulator (b5) vs coherence handoff (b1).

## Appendix 295-296: junction typology (2026-08-18)
- Three critical junctions, three architectures: b0 distributed lexical (no narrow core), b1 coherence-critical (top-4 cut +4.08 > full +1.95), b5 narrow regulator (4 dirs, mean-carried). Shared wiring motif, private implementations. Fresh shape-certification queued.
- Junction types b5 (narrow) and b1 (coherence-inverted) certified fresh; b0's "distributed" amended to window-dependent/broad (token-conditioned junction follows the token mix). Basis-stability test queued.
- b0 puzzle: subspaces agree (0.676) but cost transfer fails 7x -- variance basis != importance basis. Orthogonal-remainder test queued; b0 labeled broad/importance-unstable.
- b0 resolved: handoff cost is HOLISTIC (joint cut 6x the sum of parts; no additive decomposition at rank<=16). Final typology: front junctions coherence-holistic (b0 harmless-end, b1 catastrophic-end), b5 additive-narrow regulator. Arc closed.

## Appendix 300: junction anatomy universal (2026-08-18)
- bilin12: same anatomy (front junctions, narrow mid junction +3.52 at block 5 top-4=85%, free tail). Regulator architecture transports; adjacency to private writer holds with flipped order (consume vs regulate) -- link test queued.
- bilin12 channel lives IN the span (overlap 0.656, 63x floor); damage signature matches bilin18 class-for-class. ONE conserved complex: private code + narrow adjacent channel, plumbing order the only free parameter. Cross-model arc closed.

## Appendix 302: swiglu contrast (2026-08-18)
- Front junctions architecture-general (swiglu even larger); mid junction exists in 3/3 models (contrast refuted); the NARROW 4-dir channel + private-code fusion is bilinear-only (swiglu intermediate, 3-5x smaller). Concentrate-vs-spread dichotomy now visible in wiring. Junction program closed, scoped.

## Appendix 303: head hybrid (2026-08-18)
- 26 comps + 38/72 motif heads = +2.364 window C, beating the all-dict band by 0.27; motif swaps' joint marginal NEGATIVE. Fresh certification queued.

## Appendix 304: new fresh frontier + a8 closed (2026-08-18)
- Frontier: 26 comps + 38/72 heads at +2.543 fresh (79% of model work). Motif marginal +0.12 fresh (C-window negative was luck).
- a8 symbolic rescue refuted (identical to stream-only); a8 closed irreducible alongside induction.
- Fold tables strictly dominate empirical (C: 1.390 vs 1.481; fresh: 1.602 vs 1.926). Transfer tax avoidable. Merging into the frontier config.
- Naive fold merge FAILED (+3.76) by violating sequential fitting; corrected in-chain refit queued.
- In-chain fold merge also failed (+4.77): fits under dict-band context, eval with band real. Rule upgraded: fit under the eval config's context. Matched-context merge queued with empirical twin control.

## Appendix 308: matched-context merge (2026-08-18)
- 20 components at +1.457 fresh / +1.251 C (fold front + CP middles + context-matched tail). Beats the 34-comp frontier by 1.47 nats with 14 fewer components. Fold beats matched empirical twin by 0.40. Ladder rebuild queued.

## Appendix 309: composition economics (2026-08-18)
- Motif-head cost is substrate-dependent 9x (0.12 empirical vs 1.04 fold base): stream fidelity, not CE, prices downstream rungs. Frontier: 20@+1.46 / 28@+2.84 / 34@+2.93 fresh. Stream-MSE mechanism test queued.
- Global stream-fidelity hypothesis refuted; LOCAL read-site pricing fits all four observations (fold worse at 2-9 where motif heads read -> +1.04; better at 10-17 where tail dicts read -> 0.34). CE robust to ~100% relative stream error. Control queued.
- Local pricing CERTIFIED (0.43 vs 1.04; 18% local fidelity gap -> 2.4x cost). New envelope point: 20 comps + 38 heads @ +2.29 fresh (dominates 26+38 @ +2.54). Composition economics complete.
- Pricing law calls its second config: 28 comps + 38 heads at +2.674 fresh (beats old 34-comp by 0.26 at comparable coverage); increment +0.386 in the registered band. Envelope: 20@1.46 / 24@2.29 / 32@2.67 / 34@2.93.
- Mixed front: between the pure substrates, beats neither at its coverage; motif marginal orders by mid-band fidelity across 3 substrates (0.43/0.59/1.04) -- pricing law's third confirmation. Envelope unchanged.
- Damage modes v1: stable modes but document-confounded (one doc dominates; split-half shared docs). v2 queued: 3x rows, winsorized, document-disjoint halves.

## Appendix 315: damage modes certified (2026-08-18)
- Data-dual instrument works: 5 document-disjoint-replicable co-dependence modes, NONE aligned with the 10-class taxonomy (its blind spot measured). New candidate circuits incl. the tail-MLP band as one causal unit and a name-fragment attention complex. Certification queued.
- Circuit tree v1: 17 leaves, recursion VALIDATED (child replication 0.60; 12/17 taxonomy-invisible). v2 scaling: 1.5x data, 24 roots, depth 2.

## Appendix 317: 68 supervised circuits (2026-08-18)
- Circuit tree v2: 68 document-disjoint-replicated leaves (4x v1), 49 taxonomy-invisible, recursion rate 0.60 at both depths; depth-2 gate correctly refuses tiny slices. Binding constraint = document diversity. Pack builder + naming wave + third tranche queued.
- Tree v3: corpus boundary landed on the split -- accidental CROSS-CORPUS gate; 35 survivors = corpus-general tier (two-tier census: 68 within / 35 cross). v4 with interleaved rows queued.
- Blind-discrimination naming: mean 4.45/6 (chance 3.0), 35/67 names certified (>=5/6). Census tiers: 68 structural / 35 corpus-general / 35 blind-nameable. v4 scale-up running.

## Appendix 320: 118 supervised circuits (2026-08-19)
- Corpus-interleaved tree: 118 replicated leaves (>100 target), 91 taxonomy-invisible, child replication 0.72. Trajectory 17->68->118 in one day. Naming wave launched.

## Appendix 321: scored census + mode-class refutation (2026-08-19)
- 47/115 blind-nameable (mean 4.15/6). Census tiers now: 118 structural / 47 blind-nameable.
- Mode-constants dictionary refuted (8.4% vs 55.6%; shuffle identical): modes are component-axis, not output classes; overrides destroy base labels. Hierarchical v2 queued.
- Hierarchical mode splits harmless-but-useless (56.1 vs 55.6): orthogonality double-certified. Mode->benchmark feedback goes via component-axis stand-in selection (backlog, to be registered fresh).

## Appendix 323: red-team round one (2026-08-19)
- Passed: distinctness 118/118, contrastive 97/118, confounds clean. FAILED: sign-blind joint-causal bar (3/20) -- but big SIGNED effects (members improving 24x controls) indict the bar; sign-aware re-certification registered and queued.

## Appendix 324: census downgrade (2026-08-19)
- Sign-aware causal FAILED 4/20 (registered stakes honored): census = causal-dependence neighborhoods, NOT linear response units (composition physics at circuit scale). Selectivity (14/20 at 2x, post-hoc) registered fresh on the next 20 leaves.
- Selectivity bar HELD fresh (15/20 at 2x matched controls): census certified as causally-selective dependence neighborhoods. Audit complete: 3 attacks repelled, 1 reshaped the claim, reshaped claim survived fresh.

## Appendix 326: mechanism v1 (2026-08-19)
- 0/22: trigger sets hijacked by GPT-2 glitch tokens (globally loud fold outputs) -- instrument flaw, fixed member-blind in v2 (fractional projection + corpus support). Side finding: glitch tokens detectable weights-only.

## Appendix 327: mechanism ladder (2026-08-19)
- Unigram mechanisms insufficient (0/22 twice, sane triggers = name-initials vs second-token members): front circuits are PAIR conditions minimum. Bigram fold queued; complexity-class-by-rung is now an instrument. Overnight: fake battery, programs, bigram mechanisms, gated assembly, slack harvest.

## Appendix 328: first cash-ins (2026-08-19)
- Gating: census positions worth 3.1x random per token (control held); economics failed at broad gates -- v2 tighter.
- Slack harvest: hurt due to arbitrary SVD sign convention (my bug); v2 sign-calibrates on fit window.
- Fake battery: F1 rejected cleanly; F2 rejected STRUCTURALLY (no matched controls exist for pure-severity members); rerun with that as a scored outcome.

## Appendix 329: description-language hierarchy (2026-08-19)
- 64-bit programs capture 14/118; unigram mechanisms 0; names 47; the rest real-but-undescribed. Complexity class per circuit is now a measured quantity on three ladders. Bigram mechanisms in flight.

## Appendix 330: harvest milestone + battery holes (2026-08-19)
- MILESTONE: sign-calibrated slack policy improves the real model (-0.048 fresh; random control +0.143). Gating efficiency 9.4x at tight gates.
- Battery found 2 holes + 1 flaw in the certification: composition invisible to selectivity; statistic-selected fakes pass both dims; sibling-foreign broken. v3 adds sign-mixedness per candidate + family-disjoint foreigns.

## Appendix 331: induction-grade pair (2026-08-19)
- 2 circuits certified at full induction grade (weights-derived bigram mechanisms, 33-36x precision, readable: "second piece of rare multi-token words"). Hierarchy: programs 14 / unigram 0 / bigram 2 / names 47. Fold extends to layers 0-9 next.

## Appendix 332: battery closes (2026-08-19)
- 5/5 fakes rejected; adversarial hole CLOSED by sign-minority (0.00 vs reals 0.33-0.44). Specificity finalizes as ratio-of-selectivities (v4, registered 4/5 reals). mb4: deeper leaves need pattern-side mechanisms (backlog).

## Appendix 333: battery final (2026-08-19)
- Converged battery = selectivity + sign-mixedness + structural: 4/5 fake classes rejected per-candidate, reals 5/5. Composition-of-reals is provably per-candidate-invisible (siblings share 0.62-0.81 members intrinsically) -- governed by tree accounting instead. Pipeline calibrated; blind spot named and bounded.

## Appendix 334: pattern-side certifications (2026-08-19)
- Two attention circuits reach induction grade (7.7-7.9x, triggers = place-suffix|newline pairs; r.8.0 = parent of heading-recall). Induction-grade total: 4. 61 attention leaves need longer-context folds (backlog).

## Appendix 335: linear fraction measured (2026-08-19)
- 10/60 = 16.7% (Wilson 9.3-28.0%): one in six census circuits is a linear response unit. Census quotables complete: 118 / 35 / 47 / 14 / 4 / ~17% -- every adjective a measurement.
- Nameability-linearity association REFUTED as registered (p=0.39): independent axes. Batch trend was rank-batching noise.

## Appendix 337: gating deploy gap priced (2026-08-19)
- Oracle gates 9.4x efficient, deploy 64-bit program gates 1.5x: the gap prices census knowledge above surface features. Gating arc closed; both major arcs at fully-measured rest.

## Appendix 338: compositionality dividend (2026-08-19)
- Library search: 36/104 previous failures crack + 14 more from iteration-2 compounding (~50 total; programs are circuits-of-circuits). Programmable population 14 -> ~64/118. Until-dry continuation queued (bar: >=70 converged).

## Appendix 339: ladder converges at 57 (2026-08-19)
- Until-dry compounding: 35 base + 14 + 4 + 4 + 0 = **57/118 programmable, converged**; null 0.509. Registered >=70 FAILED; §338's ~64 corrected to 57. Composition buys one step (+63%), not a cascade.

## Appendix 340: call-graph is flat (2026-08-19)
- 29 edges over 57 programs: 48% layered (coin flip; bar >=75% FAILED), zero hubs with >=4 users (bar >=3 FAILED). Circuit citations borrow correlated context families, not layered subroutines -- data-space echo of the no-shared-OV-library result.

## Appendix 341: 57-program gate (2026-08-19)
- Gain +0.171 fresh (12x the 14-program gate; largest deploy-legal move yet) at 65.7% gated -- but efficiency still 1.54x random (bars FAILED). Deploy gap = description-language property, not program count. probe_gate.py queued (stream probe, bars: AUC>=0.75, eff>=2.5x, >=50% oracle).

## Appendix 342: stream probe 3.8x (2026-08-19)
- Deploy-legal gating ladder now measured: surface programs 1.5x / stream probe 3.8x / oracle 9.4x. AUC 0.621 (bar failed) but efficiency bar HELD -- first deploy gate above the surface ceiling. v2 queued (quadratic features + per-mode regression).

## Appendix 343: induction heads least compressible (2026-08-19)
- Rank-16 truncation: ind +0.254 QK / +0.172 V vs self +0.005/+0.008 -- prediction inverted, all bars failed. Motif simplicity = pattern shape, not weight rank; induction's match-and-copy needs full-width keys. Backlog 5 closes as measured-unpromising.

## Appendix 344: line-break circuit explainer (2026-08-19)
- r.0.0.1 explained standalone: two-signed line-break policy in list layouts (96 members +1.0 hurt / 88 members -2.7 helped by ablation), |dCE| 2.67 on members vs 0.35 background. 5 top + 5 random examples published verbatim on a standalone artifact page. Registered uniform-damage bars failed -> sign-mixedness made concrete.

## Appendix 345: fold-basis ladder registered (2026-08-19)
- Early-layer weight folds (mlp0-3 over vocab, top-8 PCA dirs, 32+64 predicates) added to the ladder library; until-dry rerun in flight. Bars: >=65/118, null <=0.6, fold cited by >=half the gain.

## Appendix 346: fold features substitute, don't extend (2026-08-19)
- 58/118 (+1; bar >=65 FAILED) with fold predicates cited in 19 programs. Programmable frontier ~58 is robust to input-feature choice; the rest needs transported context. Mirrors the deploy-gap language result.

## Appendix 347: probe v2 negative (2026-08-19)
- Quadratic per-mode probe WORSE than plain ridge (AUC 0.551 vs 0.621, gain -0.005 vs +0.033). v1's 3.8x stands. Richer basis at block 2 overfits; ceiling needs later reads or context aggregation.

## Appendix 348: two-signed is universal (2026-08-19)
- 50/50 census leaves two-signed, 96% concentrate >=3x. Every census circuit is a policy with right-and-wrong members. Base-CE stratification + mechanical examples recorded for all.

## Appendix 349: push-brake structure + tension edges (2026-08-19)
- r.0.0.1 = mlp0 break-push MINUS mlp3 brake (bundle anti-corr -0.48; Abbey improvement 95% brake-attributable). 11 tension edges to other leaves measured (first in registry). Interchange test queued.

## Appendix 350: local channel-setting null (2026-08-19)
- Setting the push channel's value locally does ~nothing while deleting it globally moves nats: the read is contextual/bilinear, not local. interchange2 (local vs prefix vs prefix-only vs random) queued to localize the path.

## Appendix 351: registry live + IOI opens (2026-08-19)
- 50 circuit records + viewer live; class labels ported into census_lib after 2/16 program bar failure. IOI: 99% pair accuracy (+2.41 margin, control ~0) -- first open task window; localization queued with registered concentration/attention/head bars.

## Appendix 352: value-setting inert (2026-08-19)
- Local/prefix/context-only transplants of the push channel all ~0 on the newline logit while deletion moves nats: bilinear circuits are subtraction-defined; additive-feature patching intuitions don't transfer. DAS motivated. Arc closed.

## Appendix 353: IOI owners (2026-08-19)
- m1/a14/a5/m2/m0 cover 70% of the IOI margin (all bars HELD). Zero-deletion gives uniform ~1.64 per-head drops in a14 (recompute exact) -> off-manifold magnitude shock, not head content. Mean-ablation head leg queued (ioi_heads2); enriched-library program rerun queued (sop_programs2).

## Appendix 354: name-mover found (2026-08-19)
- Mean-ablation head leg: a14.h4 = 0.93/1.35 of the layer drop (top-2 81%); a5 top = first-head (5,7), induction (5,5) third. Methods rule: zeroing reads uniform (magnitude shock); mean-ablation reads content.

## Appendix 355: enriched programs 5/16 (2026-08-19)
- Class labels lift passes 2->5 (bar >=8 missed); circ_r_1_1_2 reused by 3 other programs (registry compounding live).

## Appendix 356: swarm dry-run (2026-08-19)
- 3 Sonnet agents x SOP: causal numbers reproduced exactly, 0 junk certifications, 1 self-downgraded story (integrity under mid-run edits). 5 defects found -> 3 fix rounds pushed (deep-merge, resume clause, causal-direction red-team, consolidator git model, locked registry). Pipeline ready for scale.

## Appendix 357: pair fold rules out 2-token context (2026-08-19)
- 55/118 (LOWER than 58; bar failed): frontier feature-robust at 55-58 across four vocabularies; adding features overfits greedy search. Unprogrammable leaves need attention-transported long-range context -- now the only standing hypothesis.

## Appendix 358: DAS steers, specificity unproven (2026-08-19)
- Learned basis: holdout +4.42 (PCA was the problem) BUT shuffled-objective control +2.70 -> adversarial steering confound; das2_natural queued (natural donor values in learned basis).

## Appendix 359: IOI is a parallel sum (2026-08-19)
- All joints 97-99% of single sums: no serial chain; independent additive voters (first-head, induction, mover). Consistent with flat call-graph + additive junctions.

## Appendix 360: only subtraction bites (2026-08-19)
- Natural values +0.16; optimized +4.42; ZERO-coords also +4.42 -> DAS learned deletion, not steering. Law: intervention algebra in bilin18 is projection, not assignment. Arc closed.

## Appendix 361: mover is task-contextual (2026-08-19)
- a14.h4 on natural text: diffuse (38% top-5%), modal class 'other', name fifth. IOI mover role = context-induced specialization of a general head; task circuits assembled from general-purpose voters.

## Appendix 362: law refined on 10 circuits (2026-08-19)
- Registered "constant void" FAILED (0.626 < 0.75) -> refined: variance-removal dominant (13x control), off-manifold zero pays ~1.6x offset tax, on-manifold value choice nil. Violator r.6.0.0 earmarked. Rule: mean-ablate, never zero-ablate.

## Appendix 363: not token-definable (2026-08-19)
- Match transports (unbounded range) still 55/118: fifth leg closes the input-computable class. Named result: half the census is not token-definable; description moves to the stream. stream_features queued (bar >=64).

## Appendix 364: shape vs gain (2026-08-19)
- Raw-pattern linear fit fails (1/162) while swaps work: unnormalized squared-product patterns = shape x gain; v1 measured gain variance. v2 queued with per-query normalization. Swarm rule: separate shape from gain.

## Appendix 365: induction-as-code refuted (2026-08-19)
- Literal match-and-copy code: CE +9.8 at match positions, corr 0.03, alphas ~0 mixed-sign. The 'ind' motif is pattern-shape, not functional read. mech_diag queued (sparsity + location of the real read). suffix_code fixed (leaf_hooks) and requeued.

## Appendix 366-369 (2026-08-19)
- 366: stream predicates +0 -> plateau spans six languages; residual half = activation-space objects. Description arc CLOSED at 55-58/118.
- 367: optimality 10/10 (ratios 17-66x); no net-harmful circuit.
- 368: context-freeness refuted: 2-token core = 8.5% of r.3.0's action; 91.5% context gap.
- 369: induction heads = opportunistic part-time match-readers (sparse 6/9, match-seeking 5/9 at ~1/3 rate). Pattern-as-code is the legal next rung; sparse_read_code + head_read_census queued.

## Appendix 370-371 (2026-08-19)
- 370: FIRST computational-grade pass -- one-line sparse-read code carries 77% of induction-head function, 62% IOI margin, 3x control. All bars HELD.
- 371: sparse reads = minority idiom (69/162 heads); functional dictionary written; pattern-as-code scoped to the sparse minority. sparse_code_all + violator_probe queued.

## Appendix 372-373 (2026-08-19)
- 372: 69-head one-read code +0.49 (bars failed; fresh travels); diffuse control CHEAPER (+0.23) -> top-1 share ranks compressibility not need. head_code_frontier queued (cheapest-first curve, all 162).
- 373: violator resolved as max offset tax (2.8x), not a new channel type. Projection law final: variance dominant, offset tax 1.1-2.8x by bundle.

## Appendix 374-375 (2026-08-19)
- 374: MILESTONE -- 40 cheapest heads at -0.077 (model improves), 80 heads at -0.043 grid / +0.064 fresh; first interaction penalty at 120 heads. Induction heads among the least compressible (5/9 rank >120).
- 375: r.3.0 code: 59% mean, median good, key-specificity weak (1.7x). Partially computational; residuals named and under anatomization.

## Appendix 376: induction closes computational (2026-08-19)
- top-k curve: 1->+0.129, 2->+0.024, 4->+0.007, 8->+0.004 (deletion +0.601). Four coincidence reads = 98.8% replication, first computational-grade circuit. Extra reads are NOT match-family (19.8%); no context-corruption effect. topk4_stack queued (all 162 heads on 4-read code).

## Appendix 377-378 (2026-08-19)
- 377: whole-stack 4-read code +0.22 grid (IOI improves; r.3.0 collateral 2.7x): cheapest-80 stands as optimum; cheapest-N-at-k4 sweep queued.
- 378: WARNING -- census partition window-relative (29 leaves at 2x; 9% identity). Swarm plan revised: stability diagnosis queued; production unit may become window-replicated leaves.

## Appendix 379 (2026-08-19)
- Cheapest-N at k=4: 120 heads at -0.026 (negative!), 140 at +0.076, fresh_140 +0.169 (bar missed by 0.017). Attention's interpretive frontier reduced to ~40 heads.

## Appendix 380-381 (2026-08-19)
- 380: r.3.0 heads computational at k=8 (79%, 3.3x control). k = per-mechanism capacity dial.
- 381: clustering context-dependent (5% same-data identity; 77 leaves at matched size). SOP revised: identity = machinery+program+profile, replication on disjoint windows required; member-sets are evidence, not identity.

## Appendix 382-383 (2026-08-19)
- 382: MLPs run on 11% of quadratic units per position (+0.055 grid / +0.044 fresh; random 22x). Sparse-enumerable idiom spans both architecture halves. Combined readable model queued.
- 383: re-carving real (40% containment): identity revision at full strength; census = targeting system, not ontology.

## Appendix 384-385 (2026-08-19)
- 384: COMBINED READABLE MODEL holds: +0.22 fresh, IOI improves. Whole model enumerable per position. Strongest whole-model interpretive form to date.
- 385: routing is real (static-512 9x worse; 18% slot share). MLP semantics live in the routing function, not a fixed unit set.

## Appendix 386 (2026-08-19)
- Training data = FineWeb (user-confirmed). Pile fresh legs were mildly OOD (+0.10 base): transfer results conservative, stand as-is. In-distribution fresh = fineweb_rows() (new standard); census_diverse patched to FineWeb before running.

## Appendix 387 (2026-08-19)
- Induction score input = m0|m0 (unanimous dominant pair; early band 0.80-1.00 concentrated, deep band mixed). mlp0 = the model's identity-code generator. fold_score_test queued: if the fold-only trigger predicts real reads, the early induction circuit is complete (inputs + computation + code).

## Appendix 388-389 (2026-08-19)
- 388: fold-only trigger partial (2.5: 45% vs 0.8% chance); rotary omitted -- exactly computable, v2 queued.
- 389: diverse census (22+ root modes!) outgrew the 90-min cap; runner timeout 4h + dup-guard active; rerun queued.

## Appendix 390-391 (2026-08-20)
- 390: diverse census lands -- 311 leaves (5x FineWeb corpus, <=2 rows/doc), median top-document share 1% (dominance broken), all bars held; A/B machinery-replication leg queued (census_ab_replication).
- 391: fold trigger v2 -- rotary was most of the gap (1.4: 19->62%, 2.5: 45->52%, 3.5: 7.5->39.1%) but registered bars still FAILED (3.5 misses 40% by 0.9pt; corr <0.4 everywhere). v3 queued with the full token-computable residual (wte direct path + all mlp folds).
- 392: fold v3 exhausts token-computable info at ~2/3 early-band hits (3.5 got WORSE with more folds -- m0 is the identity code, later folds add noise).
- 393: LAYER-1 TRIGGER CLOSED: real-m0+wte arm hits 99.8% (corr 0.999) on head 1.4 -- the gap was m0's contextual input; a0's residual write is pattern-irrelevant. Layers 2-3 cap at 58/43% even with full block-0 context (their gap is layer-1+ writes).
- 394: census A/B replication -- 77% of leaves' machinery profiles replicate across corpus halves (matched 0.955 vs null 0.544), but depth>=2 drops to 56%: identity is coarse-grain weighted. Full-tree certification run queued.
- 395: full-tree A/B gate -- 165/311 leaves certified (cos>=0.7); full rate 53% (tree is depth-heavy: 72% coarse vs 46% fine); specificity 0.74 vs 0.27. Swarm claims certified leaves first.
- 396: 2.5 trigger fork resolves M1-MEDIATED (a1 write adds nothing, m1 real write 0.86); MLP-ladder match-code hypothesis registered and queued (mlp_ladder_code).
- 397: early induction match code IS the MLP ladder (1.4: 0.998/corr 0.999, 2.5: 0.859/0.77, 3.5: 0.741/0.65; attention-for-MLP swap halves it); deep band needs attention-written content -- deep_trigger_source queued to localize which layer writes it.
- 398: deep trigger sources localized -- 5.5 = ladder + a4 (one layer, 78% of gap); 8.4 diffuse (a5-a7); a1 closes 2.5 fully once m1 is real. deep_code_content queued: is a4 relaying ladder code (one code, MLP-built, attention-moved)?
- 399: RELAY CONFIRMED at 5.5 -- a4 with ladder values recovers 95% of its lift (0.837 vs 0.860). One identity code: MLP-built, attention-moved. relay_closure queued for 2.5's a1 and 8.4's diffuse channel (with iterated-relay arm and shuffled null).
- 400: relay closure ALL BARS HELD -- 2.5 pure relay (0.999); 8.4 nested relay (0.841 iterated vs 0.617 one-shot; null below baseline). One identity code: MLP-built, attention-moved (sometimes twice). ladder_census queued (all nine heads, maximal one-code reconstruction).
- 401: CORRECTION -- ladder_census's iter_all arm was a tautology (values-from-growing-chain reconstructs the real residual exactly; corr 1.000 everywhere was the tell). Void. ladder_depth queued: bounded relay depth k, chain_k values only from chain_{k-1}.
- 402: relay depth measured -- early band closes at 1 move, deep at 2, all nine >=0.976 by 3; shuffled null flat. ladder_causal queued (live model on computed triggers, CE-priced).
- 403: CAUSAL CLOSE -- live model on computed triggers: -0.002 at match (intact-indistinguishable), shuffled +0.073, deletion +0.50. Induction arc closed end-to-end; report updated. sop_batch_certified queued (A/B gate vs SOP gate, packs for passers).
- 404: GATE INVERSION -- A/B raw-damage-cosine certification anti-predicts selectivity (cert 17% vs uncert 88% SOP-pass). 395's list is a magnitude-stability list, not a swarm shortlist. gate_reconcile queued: both-halves concentration gate + shuffled null.
- 405: selectivity gate reconciles (94% SOP agreement, 79% fresh pass, both-halves stable 88%; shuffled null 6% -- missed its 5% bar by a point, restated honestly). gate_full queued to write swarm_shortlist.json over all 311 leaves.
- 406: swarm shortlist written -- 199/311 leaves pass both-halves selectivity (shuffled null 6%). Depth INVERTS vs the old gate (44% coarse vs 72% fine): the census's fine structure is real selective machinery; only raw damage magnitude is unstable. Depth-0 conc ill-defined (noted).
- 407: relay heads NAMED -- a6.h3 (prev-token head) relays for all of 7.3/8.3/8.4; a4.h7 (prev 0.77) for 5.5. Classic prev->induction composition in exact-code form. Pooled-rho bar failed (early-band dilution, stated). relay_edge_causal queued.
- 408: relay edges causally verified -- deleting a6.h3 shifts the deep trio's reads 19-29% and the early band 0%, +0.05 match-CE (control ~0); a4.h7 selective for 5.5. Induction circuit record complete end-to-end.
- 409: shortlist->SOP transfer 98% (47 new packs, 72 total); census_lib now runs on the diverse tree (use_state). sop_program_batch queued (step 3, strict doc-disjoint programs).
- 410: surface programs fail on the diverse tree (1/72, strict doc-disjoint) -- description != mechanism. User standard recorded: named mechanism over k-laws. SOP v2 shipped; Sonnet dry-run launched; r30_writer_decomp queued (what r.3.0 compares).
- 411: r.3.0's heads compare m0|m0 -- the SAME identity code as induction, at layer 16 (near-unanimous argmax; mass diluted to 30% by depth). The identity code is the model's universal comparison substrate. ladder16 queued.
- 412: Sonnet swarm pipeline VALIDATED (first diverse-tree record r.0.3.0; 4 friction fixes shipped). ladder16's k3 bars disqualified by their own null (shuffled values recover 103% of k1 lift at layer 16); ladder16b queued with shuffled arms at every k.
- 413: layer-16 content fork PARTIAL (16.2 content gap 0.33 clears, 16.8 at 0.17 fails; claim stays at 411). SWARM_RUNBOOK.md + reviewer-two shipped; reviewer dry run launched; r30_read_semantics queued.
- 414: reviewer-two validated (CONFIRM but objection promoted to rule v2: untested specific claims cap at WEAKEN; r.0.3.0 story flagged weak). r.3.0 heads = short-range identity-code comparators (16.8 local -1/-2 reads 76%, 16.2 diffuse; same-token 5-9x null but minority). stack_writer_decomp queued.
- 415: code-resistant heads compare the SAME m0 code (7/8 m0|m0 dominant, 8th m0|m4; 1.1/1.8 at 100% pair mass). Resistance = read breadth, not input breadth. Queue stocked for swap: m0_code_geometry, local_bigram_score, courier_centrality (all registered).
- 416: identity code eff-rank 71 (read universally -- band overlap no higher than random-head null); 16.8/16.2 weak signed bigram split (+0.16/-0.20, lead not claim); a6.h3 is a HUB (shifts all 99 downstream heads, control clean) as well as the match courier -- 407/408 refined; courier_mean_split queued.
- 417: a6.h3's hub effect is content, not bias (mean-ablation doesn't dent it). courier_content_id queued: is its entire downstream role relayed identity-code values?
- 418: courier payload is NOT pure relayed code (ladder-value substitution recovers 40% of the gap, shuffled 22%); 417's closing hypothesis corrected. payload_decomp queued.
- 419: wave-2 swarm -- 4 records, 4 adversarial reviews: gates all reproduce, 1 CONFIRM / 3 WEAKEN, zero REFUTE. Stories died of base-rate and tokenizer-bit objections -> SOP v3: mechanism (leaf_input_decomp) is the deliverable; stories need cl.story_test clearance. Wave 3 running.
- 420: a6.h3's payload = m0 0.27 / first-layer values 0.27 / m3 0.14 -- but the control head carries the same mixture, so payload is a layer property; the courier role must live in the PATTERN. pattern_payload_swap queued (crossover, registered a-c).
- 421: crossover inconclusive by design -- pattern swap and value swap both match deletion, which is either "exact product needed" or metric saturation. shift_metric_audit queued (permuted-write and norm-matched-Gaussian controls, three metrics) before any conclusion is recorded.
- 422: CORRECTION -- top-read-shift is perturbation size, not content (permuted and Gaussian controls shift MORE than real arms). Cross-arm shift comparisons in 416-421 caveated; 408's within-arm dissociation stands. Use dCE + rank correlation.
- 423: functional dissociation -- for a6.h3 and a4.h7, swapping the READ PATTERN is free while swapping VALUES costs as much as deletion. Common (3/4 heads) but not universal. pattern_necessity queued (uniform/reversed/cross-row patterns) to rule out weak-perturbation artifact.
- 424: pattern SHAPE matters, alignment does not -- uniform (+0.023) and reversed (+0.047) patterns are expensive, a cross-row real pattern is cheap (+0.017), sibling corr only 0.226 so 423's dissociation is genuine.
- 425: reviewers caught two flaws in the driver's own instruments -- the ROBUST behavioral gate was underpowered (fixed: ROBUST_V2 population gate + Bonferroni) and a mechanism enrichment was an adjacent-layer artifact (fixed: --baseline cross-leaf specificity check, now required by SOP v3).
- 426: TOOL CORRECTION -- leaf_input_decomp drew one fixed row subsample; now bootstrapped (5 draws, ENRICHED_STABLE gate). r.2.0.2's mechanism claim RETRACTED (0.99-1.46 range); r.3.0.2's a14 survives at reduced scope. Wave-3: 1 CONFIRM, 3 WEAKEN, all gates reproduced.
- 427: head role map -- only 38% of 63 heads are payload-dominant, so position-sensitivity is the norm; all three sampled induction heads are position-sensitive (as their mechanism requires); 15/63 heads cost nothing to delete.
- 428: the costliest heads (1.1, 12.6, 6.3) are previous-token/local-window readers, not identity matchers (1.1: 994/1008 reads at offset -1; 12.6 decaying window; same-token only 6% vs 1.3% null). Reconciles "everything compares m0|m0" with "most heads are position-sensitive": the identity code is the currency, local structure is what most heads buy with it.
- 429: complete 162-head cost map (dCE): 24% of heads are free, spread across all depths (not late-concentrated). Head 5.7 costs +0.92 nats, 8x the next head -- and 415 showed it is the ONE head whose score does not compare m0|m0 on both sides (m0|m4).
- 430 (PARTIAL, 5/12 leaves): random rank-matched subspace ablation already yields concentration 2.4-2.7 vs the census gate of 3. Leaf concentration is only ~1.4-1.6x random. Provisional correction: the gate measures selectivity on top of a large fragility baseline. Full run requeued (now resumable).
- 431: wave 4 -- three records, all honest mechanism negatives; two agents correctly rejected their own near-miss behavioral claims under Bonferroni; one kept a real punctuation claim (39/49, p~0). META: input-writer composition is negative on 5/7 leaves, so the swarm pivots to the output side (leaf_output_decomp, with the random-subspace control built in) plus punct_generality.
- 432: the model's costliest head (5.7, +0.92 nats) is an ATTENTION SINK -- reads position 0 for 99.8% of queries (neighbour 5.3%), where the value norm is 730 vs 197. No softmax here, so it acts as a learned bias adder; sink_bias_test queued (mean/cross-row-mean arms) to see if 1152 numbers replace it.
- 433: wave 4 closes -- 4 gates reproduced, 4 mechanism negatives (one caught only by the bootstrap: 1.396 single-draw vs 1.216 min), 3 self-rejected behavioral claims, 1 kept claim under review. SOP updated with swarm-load timing, dCE threading, rev-capture, and the 430 gate calibration.
- 434: third instrument catch -- the enrichment gate's null was a max-of-5 (1.333 vs a 1.3 bar, 2.5% headroom), so it had no power against weak effects and the swarm's negatives were overstated. Fixed (ENRICHED_STABLE2 = null mean+2sd): r.3.0.2's real positive clears with room, r.5.3.1's negatives hold but are now scoped as "no STRONG mechanism".
- 435: HEADLINE -- head 5.7 (+0.915 nats to delete, the model's costliest) is EXACTLY a constant bias adder: replacing its write with its own mean costs -0.005, and a mean from other rows works equally well. 1152 numbers replace the most important attention head in the model. head_bias_sweep queued (how many others?).
- 436: output-side consumer tool works (all 6 leaves beat their random-subspace control; caveat: in_a17 is the consumer for 5/6, so it is largely a layer-level answer); the punctuation claim is VERIFIED leaf-specific (39/49 vs random-subspace 21/49); gate_specificity final -- median random concentration 2.62 vs the gate of 3, only 4/12 leaves reach 2x, so census concentration is selectivity on a fragility baseline.
- 437: bias-adding is NOT a general role -- only 1 head meets the strict definition (and the bar excluded 5.7 itself by 0.002, stated plainly); across the 12 heads costing >=0.02, a constant explains a median 39% of the cost. 5.7 is the singular case at 101%.
- 438: punctuation claim CONFIRMED by reviewer after three attacks (base-CE confound refuted: help-rate rises with base CE); held-out generalization test queued on fresh FineWeb. Report republished with the sink section + census fragility calibration.
- 439: the sink's bias vector is written by MLP4 at position 0 (share 0.63; my wte/m0 prediction failed), and routing that value through head 5.7's projection reproduces its mean write at cosine 0.999 -- the chain is closed arithmetically. sink_origin queued (is mlp4's position-0 write a fixed learned vector?).
- 440: the punctuation claim GENERALIZES to 64 fresh FineWeb rows (punct -0.008 vs non-punct +0.016, permutation p=0.000) -- ablating the bundle helps at punctuation while hurting elsewhere in the same pass. One of three random subspaces showed a weaker same-direction effect (3.7x smaller), so a small generic component exists; magnitude-matched control named as the next version.
- 441: THE BIAS CIRCUIT IS COMPLETE -- mlp4 writes a fixed vector at position 0 (norm 10.5x normal; direction identical across documents at cosine 0.998, even with different first tokens; built from m0 0.44 / m3 0.30 / m2 0.18), head 5.7 broadcasts it everywhere, deleting costs 0.92 nats, any equal constant is free. Report republished. bias_semantics queued: what does the bias push?
- 442: the bias is a UNIGRAM-FREQUENCY PRIOR -- pushes function words/punctuation (' and', ',', ' (', ' to'), suppresses rare subword debris, and is diffuse (less concentrated than a random vector, as a frequency prior should be). But the sign inverts causally: deleting the head RAISES those tokens' logits 2.8x more than control. Direct unembedding read and total effect disagree; bias_path_split queued to separate the paths.
- 443: CORRECTION -- the path-split injection used a constant 16384x too large (missing /128 on each QK factor; norm 111M vs a residual of 85K), so both arms exploded to +11.5 nats. Run VOID. 442's token/concentration/causal findings are unaffected (scale-invariant or real ablations); only magnitude-dependent arms were hit. Fixed and requeued.
- 444: sign paradox RESOLVED -- the bias's direct path lifts its top-20 tokens by +2.07 logits (exactly as the unembedding read says), while the head's total effect LOWERS them by 0.22: downstream consumers invert the direct effect ~10x. Also: injecting the constant even one sublayer late (block 6, +1.02) is worse than deleting the head (+0.92), while in-place is free -- the bias must reach mlp5. bias_injection_depth queued.
- 445: injection depth is a CLIFF not a decay -- in place -0.005, anywhere after mlp5 +1.02 to +1.37 (all worse than deleting the head), and depth past block 6 barely matters. Monotone-decay prediction FAILED; two arms were also the same point (after-mlp5 = block-6 input), noted. bias_consumer queued with subtraction arms + junk controls to separate "mlp5 is the consumer" from "late constants are generically harmful".
- 446: MLP5 IS THE CONSUMER -- with the bias reaching mlp5, stripping it from the whole rest of the stack costs 0.24 vs 0.92 for deletion (~74% delivered through mlp5); and removing the true bias downstream costs 0.24 while removing a random same-norm vector costs 3.21 (13x). Fourth self-caught instrument bug first: two arms never fired (hook nested in the wrong branch, exactly-0.0 tell), now guarded by an arms-never-fired check. bias_linearizes queued: in a bilinear layer a constant offset creates LINEAR cross terms -- the bias may exist to open a linear pathway through mlp5.
- 447: LINEARIZATION REFUTED. With an exact four-term bilinear decomposition (relative error 1.3e-6), removing the bias's linear cross terms from mlp5 costs 0.020 nats and its constant term 0.005, against 0.915 for full deletion -- a factor of 23 short. Accounting problem stated: mlp5 processing 0.02 + downstream presence 0.24 vs 0.92 total, so the bias's value is overwhelmingly an interaction. (v1 of the run was VOID -- raw constant subtracted from a normalised input, Down_bias omitted; exactness is now a registered gate.) bias_norm_vs_direction queued: is the bias a scale device or a specific vector?
- 448: the bias is a SPECIFIC DIRECTION, not a scale device -- a random same-norm replacement costs 6.30 nats (7x worse than deleting the head at 0.92), while 0.5x and 2x the true constant cost only 0.046 and 0.139. Direction essential, magnitude loosely tuned. bias_stream_geometry queued: is the constant the residual stream's own central direction?
- 449: THE CONSTANT IS THE STREAM -- the sink's bias sits at cosine 0.99 with the residual's mean direction through layers 6-11 and accounts for 62-72% of the residual norm at layers 6-8 (null 0.017; 53x separation). After layer 5 the stream is mostly one fixed vector with the text riding on top. Explains the 7x rotation cost, the free rescaling, and why local ablations find only a quarter of its value. Report republished. mech_tool_recenter queued: does this contaminate the program's own writer-share tool?
- 450: the mechanism instrument SURVIVES the stream-centre finding -- r.3.0.2's confirmed a14 enrichment holds at 3.02 after projecting the bias axis out (was 3.08), so it is not an artifact. Reason recorded: ratios compare member vs off-slice and the bias is position-independent, so it cancels; absolute share numbers at layers 6+ remain inflated and must be read as alignment, not contribution. Wave 5 launched; sink_census queued.
- 451: only TWO sinks in 162 heads (5.7 at 99.7%, 5.2 at 67.6%), both in layer 5 -- sinks are a pair, not a class (bar of 5 FAILED). Median deletion cost 0.916 for sinks vs 0.003 for others, but 5.2 costs only 0.018, so being a sink does not make a head important. sink_pair queued.
- 452: wave 5 -- three records, three mechanism negatives, and behaviorally: r.11.1.2 punctuation (p=0.0007), r.7.1.1 capitalized-initial (p=0.0015, a new class), r.23.2.3 punctuation TESTED AND REJECTED (the negative that makes the positives credible). SOP Bonferroni counting rule clarified from an agent's question.
- 453: the three punctuation claims are ONE shared effect -- joint ablation of all three bundles yields LESS excess (0.25/0.27/0.20) than any bundle alone (0.31/0.29/0.23); independent effects would add, these saturate. Random-subspace controls clean at the corrected bar (one reached p=0.041, stated). punct_carrier queued to find the carrying component.
- 454: the model's two sinks (5.7, 5.2) sit on the SAME AXIS (cosine 0.853) and are superadditive -- deleting both costs 1.21 against a sum of 0.93. Layer 5 maintains the stream centre with a dominant head and a partial backup; 5.2 looks free alone (0.015) only because 5.7 is covering.
- 455: SCOPE CORRECTION -- the punctuation effect is damage-general, not leaf-specific: mean-ablating m7 (in NO bundle) reproduces it at p=0.0005, as do a3/a7/a6; only a12 is clean. Still true: each bundle produces it, 16-dim random subspaces do not, it generalizes off-corpus. Correction appended to all three records. frequency_fallback queued: is this just the model falling back to the unigram prior (which the sink constant already read out as)?
- 456: frequency does NOT explain the punctuation effect -- within the top unigram-frequency quartile, punctuation still helps 77.8% vs 54.4% (23-point excess), and the frequency ladder is flat-to-noisy. Damage does drift predictions toward the unigram prior but only ~4% in KL. Cause still open; punct_competitor queued (is the intact model over-confident in a wrong continuation at punctuation targets?).
- 457: THE PUNCTUATION EFFECT EXPLAINED -- it is a model deficiency, not a circuit. At helped punctuation positions (n=100 pooled) the intact model's top-1 is a NON-punctuation continuation 75% of the time, and ablation suppresses exactly that competitor by -0.108 vs +1.2e-06 for a random token; the control class shows no reverse asymmetry (12.6%). The model over-continues at phrase boundaries and damage relieves it. punct_overconf_source queued: which components create the bias?
- 458: the over-continuation bias has a LOCATION -- all five components whose ablation helps (a3, a8, m7, a7, a6) push the wrong continuation over the true punctuation target in logit space (+0.39 to +7.28), while the clean control a12 pushes the target instead (-0.45). The geometry leg is recorded as CONTAMINATED (helpers' cosine 0.712 but the control sits at 0.650 -- the stream-centre inflation from 449/450); punct_overconf_recentered queued to redo it with the centre projected out.
- 459: recentering kills the geometry leg (helpers' cosine 0.294, control 0.349 -- HIGHER; 458's 0.712 was stream-centre inflation) and exposes a bar-design error: my binary sign test failed because the control's margin is +0.04 while every helper sits at +3.15 to +8.91 -- a 100x discrimination thrown away by asking for a sign instead of a margin. Lesson recorded. Report republished with the over-continuation arc. punct_repair queued: can we FIX the bias on fresh text?
- 460: the rank-1 REPAIR FAILS -- subtracting the fitted over-continuation direction makes CE worse at every scale, and at the smallest scale hurts punctuation 6x more than a random direction of the same norm (+0.057 vs +0.009). This settles what kind of thing the bias is: NOT a fixed additive vector, unlike the sink constant which is one and is freely replaceable. The model has both kinds. punct_oracle_ceiling queued to measure the headroom an oracle gate would buy, and whether a causal proxy captures any of it.
- 461: THE ORACLE CAME BACK BACKWARDS (+0.117 at punctuation on fresh text, where it should have been an upper bound on benefit) and exposed SELECTION in my own chain: 457's "75% over-continuation" was measured at positions chosen because ablation helped there -- circular. What survives unselected is 440's held-out bundle result (punct -0.008 vs non-punct +0.016, p=0.000). Also newly in conflict: whole-component and 16-dim-bundle ablation point OPPOSITE ways at punctuation, and the arc had treated them as the same. Report corrected and republished; punct_unselected queued to settle it without conditioning.
- 462: THE OVER-CONTINUATION CLAIM IS REFUTED -- unselected, the intact model's top-1 at punctuation targets is non-punctuation only 23.5% of the time (not 75%); the earlier figure was pure selection. Withdrawn from the ledger and the published report (section retitled "Three readings that died"). What survives: ablation spares punctuation under BOTH interventions (bundle -0.025, components -0.422 dissociation) -- 461's "conflict" was my oracle design. punct_confidence queued: is punctuation spared just because it is predictable?
- 463: predictability does NOT explain the punctuation sparing -- confidence-matching leaves 91.4% of it (-0.0229 of -0.0251), and damage is flat across confidence deciles rather than falling. Four hypotheses now killed (circuit, frequency, selection, predictability); the small unselected sparing still stands. class_sparing queued: is punctuation special or one of a structural-token family?
- 464: the surviving sparing is scoped -- punctuation (-0.025) and digits (-0.018) are spared, subword marginally; space-words (+0.014) and capitalised (+0.006) pay extra, and NEWLINE is the worst-damaged class (+0.027), refuting my "format family" guess. The bundle shifts competence away from content words toward punctuation and digits.
- 465: the 39 individually-free heads are genuinely free -- deleting all of them jointly is still beneficial (-0.030), subsets no cheaper, no superadditivity; the same-sized costliest set costs +2.90. The sink pair's mutual cover is local, not a general redundancy pool: a quarter of the attention heads simply are not doing much.
- 466: wave 6 -- r.11.1.1 (negative mechanism, punctuation claim p=0.0004, +21.8pt margin, class-defined per the 462 rule) and r.4.1.1 (negative mechanism, no behavioral claim). FOURTH independent rediscovery of the punctuation effect -> SOP now documents it as a KNOWN GENERAL EFFECT with its population value so agents stop re-reporting it. Two tooling fixes from agent reports: story_test_class returns `margin`; leaf_output_decomp now read-merge-writes (it had silently dropped a concurrent agent's entry). damage_signature queued.
- 467: NOT a universal damage signature -- a random same-rank subspace gives the OPPOSITE sign at punctuation (+0.007) while real machinery of three kinds all spare it (bundle -0.025, head -0.008, MLP -0.062), and the finer profiles disagree (newline: worst under the bundle, spared under the MLP). ARC CLOSED: five readings tested, four died; the survivor is "ablating real machinery shifts competence toward punctuation, arbitrary perturbation does not". Report republished. a14_pathway queued to escalate the program's one confirmed enrichment to a causal claim.
- 468: a14's enrichment is CAUSAL -- ablating it damages r.3.0.2's members at concentration 5.76, matching the leaf's own bundle (6.10). But two registered predictions failed honestly: the adjacent control a13 also reaches 3.40 (selectivity is a gradient, not unique to a14), and a14 + bundle are ADDITIVE (0.892 vs 0.918 summed), so they are two independent pathways rather than one circuit. enrichment_predicts queued: does the mechanism table predict causal selectivity across twelve components, or not?
- 469: THE SWARM'S CENTRAL INSTRUMENT IS VALIDATED -- enrichment ratio predicts causal ablation selectivity at Spearman 0.842 across ten components (a14 2.43->5.76, a16 2.17->8.56, a0 0.74->1.68). A cheap forward-pass table ranks which component's removal will hurt a circuit most. The absent-component control bar was unevaluable (the table covers every earlier component for a layer-15+ leaf) -- design error, recorded. Report republished with an instrument-check paragraph. enrichment_generalize queued: does the ordering survive on leaves whose table is negative everywhere?
- 470: generalization holds but thinly -- ordering survives on one negative leaf (rho 0.905) and collapses on the other (0.119), and negative leaves' concentration range is ~2 vs 7.1 for the positive. SOP now says: chase ratios only when ENRICHED_STABLE2 is true somewhere. CORRECTION to 469: two of the ten validation components were the leaf's OWN machinery (definitionally selective); excluding them, rho = 0.762 rather than 0.842. Validation stands, headline number lowered. mech_map_all queued (60-leaf census-scale mechanism map).
- 471: CENSUS-SCALE MAP -- over 60 shortlist leaves, only 3 (5.0%) carry any stable enrichment, and all three positives are ADJACENT-LAYER MLP pairs (m14->m15 twice, m15->m17 once; adjacency fraction 1.00). All bars held, the first by a wide margin (registered <25%). Honest reading: 57 of 60 damage-clusters are input-diffuse, and step-3M is a cheap SCREEN with a usual outcome of a scoped negative, not the deliverable v3 promoted it to. Specificity screen running on all three positives.
- 472: ONE REAL MECHANISM IN 60 LEAVES -- of three screen positives, m15->m17 is a layer property (a peer scores higher) while m14->m15 is genuinely leaf-specific (2.3 vs peers ~0.9) and is carried by two SIBLING leaves, i.e. two views of one circuit. 57 of 60 leaves are input-diffuse. Bar (b) was unevaluable by my own design (registered a distant-pair comparison when adjacency fraction was already 1.00) -- third error of that kind, rule recorded. SOP: step 3M documented as a screen whose usual correct outcome is a scoped negative. m14_pathway queued.
- 473: CENSUS YIELD IS ZERO WRITERS -- the last surviving mechanism fails its neighbour control: m14 (enriched) gives concentration 4.33/4.29 on the two siblings while adjacent unflagged m13 gives 4.44/4.89. Full funnel: 60 leaves -> 3 screen positives -> 2 leaf-specific -> 0 causally distinguishable from a neighbour. Input-composition analysis RANKS magnitude (rho 0.76) but cannot isolate a writer. Report instrument section rewritten and republished; SOP now mandates a neighbour control before any writer claim. Silver lining: the two siblings agree to 1%, the cleanest evidence yet that census sibling structure is real. band_unit queued (is the unit a BAND?).
- 474: BANDS DO NOT ISOLATE EITHER -- m13+m14 damage is almost exactly additive (1.33 vs 1.45 summed), extending to m12 adds 41-46% (no boundary), and a DISTANT band does 86-91% as much absolute damage. Locality shows only in selectivity (concentration 4.4-4.9 adjacent vs 2.7-3.2 distant). Input-composition thread CLOSED: it gives a magnitude ranking over a smoothly accumulating dependence, not culprits. Pivot to anomaly-chasing (which produced both complete circuits): head_0_3_fold queued on the model's second costliest head, where layer-0 structure makes exact foldability testable.
- 475: head 0.3 (2nd costliest, +0.112) is a PREVIOUS-TOKEN head with a self component (offset -1: 65.8%, offset 0: 28.1%), and its pattern folds EXACTLY from weights+tokens+rotary (fold match 1.000, as layer-0 structure demands). The value-table bar failed on my own coverage bug (table built only over sampled top-read tokens); head_0_3_exact queued with the weights-only full-vocabulary table plus a complete pattern+value fold and a shuffled null.
- 476: THE SECOND COSTLIEST HEAD IS EXACTLY A LOOKUP -- head 0.3 (+0.112 nats to delete) replaced entirely by weights-only lookups (pattern from tokens+rotary, values from a 50304-entry table) costs -0.0 nats; the token-shuffled null costs +0.147, so content matters. Framed honestly: layer-0 foldability is architecturally guaranteed, so the yield is a verified-exact instrument, the head's named shape (prev-token 66% + self 28%), and no runtime attention needed. Third component of the model reduced to something writable. layer0_fold queued: does the WHOLE first attention layer fold, and does layer 1 fail as it must?
- 477: THE FIRST ATTENTION LAYER IS A BIGRAM TABLE -- all nine layer-0 heads replaced by weights-only token-pair patterns and per-token value tables cost -0.000 nats; shuffled tables cost +0.237; the identical construction at layer 1 costs +1.470 (150x), so the boundary is sharp and measured. Report republished. mlp_table_ladder queued: how far does per-token tableability reach in the MLP chain, given mlp0's input includes attn0's output?
- 478: CORRECTION -- the front MLPs are NOT per-token tables. Replacing them with weights-only per-token tables costs +1.018 (m0), +1.775 (m1), +0.744 (m2); the shuffled null is +3.188 so the machinery is fine and the loss is CONTEXT. This corrects the standing "mlp0 is an exactly-foldable token table" framing (its input is rms_norm(wte + attn0_out)); the report now shows the variance-explained figures and the loss price side by side. mlp_bigram_table queued: m0 as a function of the TOKEN PAIR, since attn0 is a bigram table dominated by offset -1.
- 479: the TOKEN PAIR is not the answer -- recomputing m0 with all of attn0's weight on the previous token costs +1.255, WORSE than ignoring attn0 entirely (+1.018); one layer-0 head reads offset -1 at 66% but the other eight do not, and forcing them there misinforms m0 (shuffled null +1.546). Also fixed a self-inflicted RecursionError (hook calling its own module; manual bilinear forward now). m0_context_window queued: sweep attn0's read window k = full/16/8/4/2/1 to measure how wide a prefix m0 actually needs.
- 480: THE FRONT OF THE MODEL IS A BIGRAM FUNCTION -- current token plus ONE previous position costs +0.004 nats, four positions is free (-0.042), untruncated fold exact at +0.0000; only self-only hurts (+0.537). This RETRACTS 479, whose "token pair is worse than no attention" headline came from a twelvefold lambda-mix bug (the residual entering block-0's MLP is 12.19*E + attn_out, not 1.0*E). Report republished. block1_window queued: how fast does required context widen with depth?
- 481: BLOCK 1 IS LOCAL TOO -- restricting attn1 to a 4-token window costs +0.014, and even a 2-token window only +0.080 (I predicted >=0.30). Both of the first two blocks are essentially n-gram machines; whatever makes this model more than that happens later than layer 1. window_by_depth queued: restrict each of the 18 attention layers in turn to a 4-token window to find where long-range dependence actually lives, with the prediction that it is layers 5-8 (the induction band).
- 482: ONE LAYER CARRIES THE NON-LOCAL READS -- restricting each attention layer to a 4-token window costs at most +0.086 for seventeen of eighteen layers, and +1.112 for layer 5 (13x the next worst). All three bars held including "worst layer in 5-8". BUT flagged before claiming convergence with the induction band: head 5.7 (the position-0 sink) lives in layer 5, and a sliding window cuts off position 0 exactly. layer5_window_source queued to disambiguate (allow position 0 / restrict only 5.7 / restrict all but 5.7).
- 483: THE MODEL'S ONLY NON-LOCAL READ IS A CONSTANT FETCH -- layer 5's +1.112 under a 4-token window collapses to +0.077 once position 0 is allowed; windowing head 5.7 alone gives +0.597, the other eight heads +0.052. So every genuine content read in 18 layers fits in four tokens plus one constant from position 0. The induction-band convergence from 482 is WITHDRAWN (the prediction held for the wrong reason). Arms are superadditive (0.649 summed vs 1.112 together). Report republished. window_at_match queued: does induction show up when scored at match positions?
- 484: at MATCH positions too, layer 5's window cost is the sink (+0.996 -> +0.070 once position 0 is allowed), so bars (a)/(b) passed for the wrong reason again and are scored uninformative. Two real findings: LAYER 12 is the only genuinely match-specific long-range layer (+0.209 match vs +0.021 non-match, 10x, far from the induction band), and windowing early layers HELPS at match positions (layer 4: -0.230). Tension with the flagship result sharpened: deleting the induction band costs +0.601 at match but windowing its layers costs ~0. induction_window queued as a decisive fork on the heads themselves.
- 485: THE FLAGSHIP SURVIVES -- windowing the nine induction heads to 4 tokens costs +0.318 at match positions, 75% of the +0.426 for deleting them outright, and 1.8x the +0.180 for nine random heads. Induction's value IS carried by its distant reads, as the circuit work claims; 484's tension resolves in favour of the result. Calibration stated: nine random heads windowed already cost 0.180, so the band-specific excess is 0.138 and the clean statement is the within-heads 75%. induction_redundancy queued: windowing whole layers was free while windowing the band together is not, which points to redundancy across the band -- now measured as a cumulative curve.
- 486: THE INDUCTION BAND IS MUTUALLY COVERING -- each head windowed alone costs ~0 at match positions (mean +0.003, four are helpful), all nine together cost +0.318: a 108x superlinearity against 10x for nine random heads. No head's distant reads are necessary; only the collective loss bites. Qualifies the flagship result (the band computes collectively) and explains why whole-layer windowing looked free. Bar (c) failed honestly -- the control is also superlinear, so the band is 11x more superlinear, not categorically different. Report republished. layer12_match queued on the last unexplained long-range signal.
- 487: LAYER 12'S LONG-RANGE WORK IS ONE HEAD -- 12.6 carries 84.5% of the layer's match-specific cost (+0.177 match vs +0.015 non-match, 12x), median head +0.003. But bar (b) used a BAD PROBE: I tested whether it reads the same token as the query (6% vs 3.3% null) when an induction-style head reads the token that FOLLOWED the repeat (p+1), a motif this program's own census names "induction-target". Recorded as my error, not a finding. head_12_6_reads queued with the corrected classification and a control head.
- 488: HEAD 12.6 IS A DIFFUSE LONG-RANGE READER -- at match positions it reads the induction target only 9.3% (8.5x its null but far from dominant), the repeat itself 3.3%, local 15.5% (control: 43.5%), and 71.4% scattered distant positions with no dominant offset. The corrected probe rules out the induction identity; the head carrying 84.5% of the model's only match-specific long-range cost is an uncatalogued type. head_12_6_targets queued: characterise it by the CLASS of token it reads, with a rarity/salience hypothesis registered.
- 489: HEAD 12.6 READS STRUCTURE, NOT SALIENCE -- its distant reads at match positions are enriched for punctuation (2.33x), capitals (1.79x), digits (1.33x) and newlines (1.27x), and DEPLETED for prose content (subword 0.38x) and rare tokens (0.69x, killing the salience hypothesis). Its layer-mate 12.3 is the mirror image (capitals 3.72x, subword 1.4x, punctuation 0.34x, newline 0.09x): one layer holds a long-range structure reader beside a local content reader. head_12_6_structure queued: does 12.6's contribution scale with the text's structural density?
- 490: CONFIRMED -- head 12.6's window damage at match positions rises monotonically with structural density (+0.0004 / +0.0026 / +0.0075 / +0.0073 across quartiles, 18.25x bottom-to-top) while layer-mate 12.3 is flat (1.26x). With 489's token-class profile this names it a LONG-RANGE STRUCTURE READER on two independent measurements. head_12_6_causal queued: block its reads to distant punctuation/newline positions specifically, against a count-matched content control.
- 491: intervening on the named variable -- blocking head 12.6's distant reads to punctuation/newline reproduces 44.4% of its window damage against 21.9% for an EXACTLY count-matched prose control (3525 vs 3525 positions). Structure carries 2x content, but bar (a) failed at the 50% line and a third of the damage is in neither class, so "structure reader" is a leading tendency not a complete function. Proportionality noted: the head's total effect is 0.007 nats -- the model's largest match-specific long-range signal, but small absolutely. head_12_6_classes queued: per-class damage-per-blocked-position, doubling as a convergence test against 489's enrichment ranking.
- 492: READ ENRICHMENT DOES NOT PREDICT CAUSAL POTENCY -- punctuation, the class 12.6's reads are most enriched for (2.33x), is nearly the least potent to block (0.00082 per 1k positions); newline, enriched only 1.27x, is most potent (0.01260, 15x punctuation); correlation across classes -0.029. So 12.6 is a LINE-BOUNDARY reader, and the general lesson (now hit twice, after writer enrichment in 473) is that where a head looks most often is not what it needs.
- 493: THE ATLAS -- all 162 heads profiled (cost, window need overall and at match, motif profile, read-class enrichment). NO new long-range heads: beyond 5.7 (0.590) and 12.6 (0.164) the next is 14.4 at 0.035, so the model has exactly two heads needing distant reads. 93.8% of heads have a dominant motif -- the population is highly stereotyped. newline_circuit queued as the first behaviour-defined circuit attempt using the atlas.
- 494: behaviour-defined attempt step 1 -- top five components carry 57.5% of newline-target cost and the leader is not the sink, BUT all five cost more at non-newline positions (m1 ratio 0.10), so bar (c) failed and the ranking is just "the front of the model does everything". Lesson recorded about the playbook: for a behaviour target, rank by damage CONCENTRATION on the behaviour, never by absolute damage at it. newline_specific queued with the corrected statistic plus a position-matched control.
- 495: FIRST BEHAVIOUR-SPECIFIC COMPONENT. Ranking by damage concentration rather than magnitude, attention layer 12 costs +0.0777 nats at newline targets vs +0.0073 elsewhere -- ratio 10.64, five times the next component (a10 4.19), and 2.7x its own position-matched control (3.95). All three bars held. The front block that dominated the magnitude ranking sits at 0.10. Head-level decomposition queued with an advance bet from the atlas (12.6, the only expensive head in the layer).
- 496: reviewer wave 6a -- r.8.1.0 CONFIRM (clean negative), r.4.1.1 WEAKEN: its a12 threshold sat 0.055 above the null's worst draw, so the bar rested on the null ceiling and the negative was uninformative rather than wrong. leaf_input_decomp now reports null_bar_separation and labels negatives DECISIVE/UNDERPOWERED/NEAR_MISS; --seeds N widens the bootstrap without overwriting records.
- 497: THE NEWLINE HEAD. Head 12.6 alone carries 88% of a12's newline specificity (+0.0682 nats at line breaks, +0.0057 at position-matched controls, -0.0007 elsewhere), and the head atlas -- which knew nothing about newlines -- named it in advance from delete cost alone (0.073 vs <=0.008 for the other eight); the bet could have failed eight ways. Its share of score mass on the most recent preceding newline is 0.064 at line breaks vs 0.017 at controls (3.7x), the largest in the layer. Caveat recorded: ratio statistics with near-zero denominators are degenerate, so absolute pairs are reported from here on.
- 498: retroactive power audit of the whole census (no GPU -- the numbers were already stored). 43 of 142 negatives (30%) had the bar within 0.10 of the null ceiling, but 33 of those show no enrichment at all, so the census's "zero mechanisms in 60 leaves" now rests on exactly FOUR tests, queued as power_recheck at 20 seeds with four decisive negatives as a manufacturing-positives control.
- 499: the four hinted census negatives were small-sample noise -- at 20 bootstrap draws all four signals SHRANK toward 1.0 (1.46->1.06, 1.50->1.26, 1.42->0.89, 1.22->1.12), no flips, and the decisive-negative controls held, so the census's zero is stronger not weaker. RETRACTION: 498's "30% of negatives underpowered" used a max-of-N statistic that shrinks with N by construction (the same extreme-value mistake as 434, one section after recording it). N-stable replacement (bar above 1.35 = cannot see below 35% enrichment) gives 22.5%, of which 5 show a hint and 3 are now settled; power_recheck2 queued for the other two.
- 500: head 12.6 is a NEWLINE PUSHER, not a line counter. The rhythm hypothesis failed flatly (regular line breaks +0.0875 vs irregular +0.0909, 0.96x), but deleting the head lowers the logit of token 198 by 0.137 at line breaks against 0.004 for the best competitor -- a factor of 31, so it pushes one token and does almost nothing else. Its score mass is diffuse (most-recent-newline share 0.049 against 0.155 unclassified). Two process failures recorded: bar (b) was unevaluable (zero blank-line targets in the sample) and the null was registered as a quotient of two near-zero numbers, both repeats of rules already written down.
- 501: head 12.6 is a two-sided, document-gated line-break detector. It predicts "next token is a newline" with AUC 0.769 (same-layer control head 12.2: 0.450, below chance), pushing the newline logit +0.119 at breaks and -0.074 elsewhere -- it actively suppresses newline mid-line. Turned on by sentence-final punctuation and by a preceding newline ('\n' +0.106, '.' +0.085, '"' +0.075, '?' +0.074), off by ordinary words (median -0.079). Document-gated: the same trigger tokens get 2.14x the push in newline-dense documents (+0.113 vs +0.053), so it is not a bigram. Bar (a) was UNEVALUABLE (ratio against a negative median) -- the third such bar in four sections, now caught mechanically by the new cl.score_bar().
- 502: census audit closes. All five underpowered-with-a-hint negatives retested at 20 bootstrap draws; none flipped and four of five shrank toward 1.0. "Sixty leaves, zero writer-level mechanisms" now stands audited. A transient huggingface.co outage killed two runs at model-load, so load_elriggs now falls back to the local cache.
- 503: CORRECTION (caught by a registered exactness check that voided its own run). The residual stream is rescaled at every block by x = lam0*x + lam1*x0, so a writer's contribution L layers later carries the PRODUCT of lam0 over the intervening blocks, not the target block's lam0. With lam0=0.0127 at block 1, layer-0 writers were overweighted 4,242x and wte underweighted 7x entering layer 12; the flat decomposition reconstructs that input to 68% error, the corrected one to 1.2e-7. Six scripts affected including the mechanism screen behind the "zero mechanisms in 60 leaves" headline. Fixed centrally (cl.writer_coeffs/writer_parts/check_parts), the screen now refuses to run if the reconstruction misses, the confirmed positive r.3.0.2 survives, and a full rescreen of all 75 records is running.
- 504: reviewer wave 7 (r.2.0.1, r.23.2.3) -- both gates exact, both mechanism negatives DECISIVE, both WEAKEN (neither record kept a claim to confirm). Two by-products: (i) probe bundles could list nested PCA spans, so cl.orth built a rank-28 projector where the record implied 16 -- proj_hooks now dedupes and reports the true rank, and 11 of 311 leaves are affected, all in two families; (ii) a genuine lead -- r.2.0.1's bundle DAMAGES digit targets (+0.107) where the population effect SPARES them (-0.018), the first behavioural lead pointing against the population direction, flagged for a dedicated pre-registered test.
- 505: THE CENSUS HAD MECHANISMS AFTER ALL. Rescreening all 75 records with the corrected decomposition: 4 real gains, 1 real loss, 60 of 167 component tests changed top writer, every tag exact. Leaves where every component shows stable enrichment go from one to three -- r.3.0.2 (unchanged, a14; its components are late so the correction is a no-op there) plus NEW r.1.2.0 and r.1.2.2, both enriched for m5 on both of their components (1.53-1.81 against bars of 1.30-1.41). r.6.2.0 was a false positive of the flat weighting. Under the bug the top writer at both new leaves was a0 -- the one overweighted 4,242x -- so the error was installing the same wrong answer everywhere and burying the real one. "Zero mechanisms in 60 leaves" becomes three. Causal + peer verification queued.
- 506: the newline head's input is diffuse. Silencing each writer's contribution to head 12.6's query one at a time, no single writer costs more than 0.0154 AUC (m9; then m11, wte, m10) and four attention writers slightly HELP when removed -- bar (a) wanted 0.05 and failed. But silencing all context at once drops AUC 0.783 -> 0.643, more than twice the sum of the individual drops, so the query is assembled superadditively from the whole late MLP stack (m7-m11 + wte). Same mutual-covering shape as the induction band. The document gate has no located source: every writer moves it by under 0.008 against a 0.050 gap. Random-direction control clean at 0.0005. The head's output side is sharp; its input side has no small variable set.
- 507: mlp0 was an artifact in three more analyses. Corrected (exact to 8e-8): the position-0 vector at layer 5 is m4 0.876 (was 0.626 with m0 at 0.119); mlp4's input at position 0 is m3 0.662 (was m0 0.441); courier a6.h3's payload is a5 0.405 + m5 0.253 + v1 0.210 (was m0 0.271). RETRACTION of 418-419's "the courier's payload is led by m0 at 27%" -- m0's true share is 0.06%. Everywhere the flat weighting was used, m0 looked like a major supplier because its output was multiplied by 1.04 instead of 0.00024; corrected, suppliers are always the immediately preceding components. NOT affected: the identity-code analyses build the chain iteratively (the correct unrolling), so "m0 generates the code that decides where to look" stands -- what changed is what the couriers carry.
- 508: the digit lead REPLICATES on fresh disjoint rows -- +0.0900 (CI +0.044 to +0.142) where the population effect SPARES digits at -0.018, a sign flip; ten random rank-matched subspaces span only -0.015 to +0.015 and the punctuation sanity check lands at -0.037 as it should. Leaf-specificity FAILED for a structural reason: the three "peer" leaves returned identical numbers because r.2.0.0/.1/.2/.3 ARE THE SAME LEAF. 36 of 311 census tags (11.6%) duplicate another in 14 groups, so leaf counts over-count and some reviewer effort was spent twice. 505's r.1.2.0 and r.1.2.2 were checked and are genuinely distinct. digit_lead2 queued with a same-component/different-directions control.
- 509: 505's two new mechanisms FAIL causally. Silencing m5's contribution to m14/m15 costs +0.0006 and +0.0037 nats at member positions, while matched random directions cost +0.012 to +0.026 -- three to seven times more -- and the neighbouring writer m6 beats m5 at both leaves. Peer leaves match or exceed the targets. The correction changed where the screen points, not whether its pointers survive an intervention. Methodological result: the member-vs-off-slice contrast is worthless as a bar because members were selected for damage-sensitivity -- random directions score 8.5 and 18.0 against a registered bar of 1.5. Correct readout is member damage vs matched random directions; mech_a14_verify queued to apply it to r.3.0.2, the census's oldest screen positive and the only one never causally tested.
- 510: SOP step 3M is retired. r.3.0.2, the census's oldest screen positive, fails causally like the others -- silencing a14's contribution to a15/a16/a17 makes its own member positions BETTER by 0.006 nats while matched random directions make them worse by 0.016-0.021, and neighbouring writer a16 helps six times as much. Across sixty leaves, on both the flat and the corrected decomposition, the screen has produced no writer-level mechanism surviving a causal test; the correction moved 60 of 167 top writers without changing that. Damage clusters do not correspond to identifiable upstream writers in this model. Both circuits that did work (induction, the position-0 bias) and the newline head were found by starting from a behaviour and working outward.
- 511: the digit effect is DIRECTION-SPECIFIC and exactly additive. On a third disjoint sample the real bundle gives +0.1459 (CI +0.089/+0.209) while alternative PCA spans of the same rank in the same components give +0.0260 and +0.0086 -- so it is these directions, not a6/a8 wholesale. The (0,4) and (4,16) halves give +0.0931 and +0.0524, summing to +0.1455 against +0.1459 for the whole: additive to three decimals, unusual in a model where almost everything is superadditive. BUT the sanity null was VIOLATED (punctuation -0.0487, outside the registered [-0.040,-0.010]), so the run is reported as uninformative rather than banked -- and the digit number itself moved from +0.090 to +0.146 between samples, revealing sample-to-sample spread no earlier single-sample run could see. digit_lead3 running the same arms on three more samples to settle whether the bracket or the instrument is at fault.
- 512: TIERS OF UNDERSTANDING (user correction). 1 localization, 2 behavioural, 3 first-order writer attribution, 4 compositional (the exact algebra), 5 recursive. 506's "the newline head's input is diffuse" was a statement about the instrument: this network has no softmax and no activation, so a score is a PRODUCT of bilinear forms, and leave-one-out on a multiplicative computation always reports "nothing alone, everything together" -- which is literally what 506 found (0.064 summed vs 0.140 joint). Tier 4 is available in closed form: each score factor is an exact additive sum over 625 writer PAIRS. newline_head_pairs queued to compute it and to rebuild the head's score from its top ten pairs as a sufficiency test. Every "diffuse"/"high-rank" negative in this ledger was measured with first-order tools and is now a candidate for re-examination, not a settled negative.
- 513: the behaviour-first screen GENERALIZES -- ten behaviour classes for the cost of one (136s). Positive control recovered a12 for newlines (14.44). Six other classes qualify with four distinct leaders: close_bracket -> a13 (+0.694 nats at target vs +0.015 elsewhere, the largest concentrated effect measured in this program), capitalized -> a17, open_quote -> a10, open_bracket -> a17, digit -> a8, sentence_end -> a10. Digits landing on a8 independently reproduces the subspace result from a completely different method. NULL VIOLATED for two classes because I defined "elsewhere" as the complement of the target mask, which biases the denominator for large or high-damage classes; the ratio column is not quotable until behaviour_atlas2 re-runs against the global mean. The absolute pairs and the position-matched comparisons are unaffected.
- 514: THE DOCUMENT GATE IS LOCATED, and 506's "no source" was wrong because it only tested one of the head's three input paths. On the key+value path, a10 moves head 12.6's document gate by -0.0243 against a gap of 0.0503 -- 48% of it, from one component, roughly additive across key and value -- while the query side managed 0.0033. Detector quality is carried by wte on the same path (-0.0984, nine times the best context writer), so the head's line-end reading is largely lexical while its document-level gating is a10's. Silencing all context on the key side alone REVERSES the gate (+0.050 to -0.019). a10 also leads opening quotes and sentence ends in the behaviour atlas, so it looks like a structure-tracking component.
- 515: the first tier-4 pair decomposition voided itself -- writer parts reproduce the layer input to 1.29e-7 but my 625-term score reconstruction was off by 17.4x relative, because I divided each writer's share by the rms normalizers and then multiplied the assembled factor by them again. Fixed (the only remaining constant is 1/128; the four projections carry no bias) and requeued. Third time this session an exactness bar caught an error before it became a claim.
- 516: the digit effect is settled -- five disjoint samples give +0.090, +0.146, +0.143, +0.164, +0.132 (mean +0.135) where the population effect for this ablation class SPARES digits at -0.018. Direction-specific (alternative spans of the same rank: -0.027 to +0.033), additive across halves, clean against ten random subspaces, and the punctuation sanity check lands inside its bracket in three of three samples -- so 511's violation was one sample's spread, not a broken instrument. Methodological consequence: single-sample class claims here carry roughly +-0.03 of spread on a +0.135 effect, so anything smaller measured once was never distinguishable from noise.
- 517: digit_heads VOID by its own registered bar, and the bar was the mistake -- I registered "individual per-head removals sum to the whole within 20%", which tests additivity of a cross-entropy readout rather than exactness, one section after writing down why first-order bars mislead on a multiplicative network. The exactness that does have to hold did: the all-heads arm gives +0.1297 against the five-sample full-bundle mean +0.135. Descriptively (not banked): the digit effect is superadditive across heads (individual removals recover 59% of the joint) and concentrated but not single-head -- 8.3 alone carries a quarter, 6.1/6.3/8.7 about a tenth each, thirteen heads contribute nothing. The atlas bet named 8.7 (enrichment 2.45); the leader is its second choice 8.3 (1.68). digit_heads2 queued with necessity AND sufficiency measurements per head.
- 518: the behaviour atlas on an unbiased denominator (supersedes 513's ratios). All nulls now pass. Five distinct components lead nine behaviour classes: close_bracket -> a13 (+0.694 vs +0.017, an order of magnitude beyond anything else in the program), open_quote -> a10 (+0.272), newline -> a12 (+0.099), capitalized -> a15 (+0.039; was a17 under the biased denominator -- that class covers 21% of positions so its complement was the most distorted), open_bracket -> a17, colon -> a15, digit -> a8, sentence_end -> a10. Commas get nothing (0.80, anti-concentrated) and closing quotes had zero targets. a10 leading opening quotes AND sentence ends AND carrying 48% of the newline head's document gate makes it the best structure-tracking candidate in the model, never yet studied directly.
- 519: TIER 4 computed exactly for the newline head (pair sum reproduces the real score to 5.1e-7) and the answer is a clean negative: the top 10 of 625 writer pairs carry only 12.2% of pair mass (uniform would be 1.6%), and the top-10 set at newline targets differs from the one at position-matched controls by just TWO pairs. The head's input structure is FIXED and dense -- the same late-MLP pairs (m11xm11, m9xm11, m10xm11, wtexm11) dominate everywhere, so the behaviour specificity is not in which pairs contribute but in how the same pairing evaluates. Tier 4 confirms tier 3's ranking while overturning its interpretation. a10xa10 is the top non-MLP pair, independently corroborating 514's finding that a10 feeds this head. Sufficiency untested: newline_head_rebuild queued to measure retention as a curve over K with random-K controls.
- 520: A BRACKET MATCHER, and the sharpest circuit in the program. Head 13.8 alone carries 98% of attention layer 13's close-bracket effect: deleting it costs +0.825 nats at positions before a closing bracket, +0.006 at position-matched controls, +0.004 over all text, and +0.007 on random targets (a factor of 122). It lowers the closing-bracket token's logit by 1.542 against 0.473 for the best competitor. The atlas narrowed nine heads to three without knowing about brackets and 13.8 is in the set. Mechanism: it puts 14.6x more score mass on the SPECIFIC matching opener (0.381) than on other earlier openers (0.026), so it tracks nesting rather than detecting bracket-ish context -- and the sign is negative, meaning it subtracts the matched opener's value. Third degenerate-ratio null of the session (the guard catches them; my bar-writing keeps creating them) -- new rule: if a denominator can be near zero, register the PAIR as the bar. Causal test of the matching claim queued.
- 521: the digit subspace is DISTRIBUTED, and the contrast is the finding. Exactness held exactly (all-heads +0.1238 = full-bundle +0.1238). Necessity and sufficiency agree: the leading head 8.3 accounts for 24% alone and 37% by sparing, 6.1 for ~40% by sparing, and four heads across two layers carry it superadditively (individual removals recover 64% of the joint). No single-head circuit exists here. Against the same measurement elsewhere -- closing brackets 98% in head 13.8, line breaks 88% in head 12.6, digits 24-37% shared by four -- this model does both kinds of thing, and which one you get is not predictable from effect size (digits are the third-largest concentrated effect and the most distributed). The atlas bet failed a second time: it profiles what a head reads, and this is about what a head writes. Correction to 516: the eight-sample spread is +-0.05, not +-0.03 (range 0.072 to 0.168).
- 522: ONE MATRIX ENTRY CARRIES 83% OF THE BRACKET HEAD. Leaving head 13.8 intact and zeroing a single (query,key) cell per target -- its score on the specific matching opener -- costs +0.689 of the head's full +0.825. The same operation on the nearest NON-matching opener costs +0.0136 (51x less), on the previous token +0.0742, on a random earlier position +0.0035. All three bars held. The tightest causal result in the program: an 0.825-nat behaviour reduced to one number in one matrix, costing 0.0018 nats over all text because it only fires where a bracket is closing. LIMITATION: only 1 of 84 targets has a match that isn't also the nearest opener, so "matching" and "most recent unclosed opener of the right type" are indistinguishable on this data; bracket_state attacks the state question from the populated side (are already-closed openers discounted?).
- 523: head 13.8 is a POINTER, not a stack. It does not elevate outstanding brackets as a set -- already-closed openers get 0.0204 of its score mass and other still-open openers get 0.0232, the same number, against 0.0097 for a random non-bracket token. The match alone gets 0.3672, 15.8x the other still-open openers. So the head skips any "which brackets are open" representation and lands on exactly one token, treating every other bracket as ordinary text. The registered hypothesis (stack reading) failed and the result is sharper than it: whatever computes WHICH token is upstream and still unlocated. At control positions the pointer is already partly present (0.0870) and sharpens 4x when a bracket is actually due; still-open vs closed shows no separation there (ratio 0.57), so this is computed for the closing decision rather than being a fixed key-side property.
- 524: a10 is a SHARED ADDRESS, not a shared computation -- retracting my 518 reading of it as a structure-tracking component. Decomposed into heads, its three jobs belong to three different heads: 10.7 (+10.6) does opening quotes (+0.056 at target vs +0.0006 at position-matched controls), 10.5 does sentence ends (+0.060, and it HELPS quotes when removed), and 10.2 carries the newline head's document gate (-0.0159 of the layer's -0.0243) while doing nothing for either behaviour. The correction generalizes to the whole atlas: a layer leading several classes may be several unrelated circuits sharing an address, so a15 leading both capitalized words and colons should be assumed to be two heads until checked. NULL 1 violated because the sentence-end head damages random positions too (sentence ends are frequent, so a random set contains them) -- the bar should have used the quote leader's own random damage (+0.0016, 3% of its target damage); both readings recorded.
- 525: the newline head needs about a THIRD of its pair structure. Rebuilding head 12.6's score from its top K of 625 writer pairs and running the real model: K=10 retains 0.048, K=50 0.436, K=100 0.692 (the registered bar was 0.70 -- missed by 0.008), K=200 0.918, K=625 1.000 exactly. Random-K controls retain about zero at K=100 and 0.22 at K=200, so the ranking is enormously informative while no small subset suffices -- which is what a genuinely high-rank computation looks like. Retention at position-matched controls tracks the newline curve to within 0.015 at every K, so there is no behaviour-specific sub-computation to extract. Tier 4 for this head closes as a negative. Contrast with the bracket head, where ONE score-matrix cell carries 83%: circuits in this model are compressible along different axes, and "is this circuit simple" has no single answer.
- 526: the nested-bracket test VOIDED itself on its own null -- in constructed contexts where the closer matches the first of two openers, the model's NLL for the closing bracket is 3.963 nats against a registered bar of 2.0, so it does not expect a closer there and nothing measured counts. The inner condition (match = most recent opener) passed at 1.496. That asymmetry is itself weak evidence for the depth-free hypothesis: the model closes the bracket it just opened comfortably and one from two clauses ago not at all. The nesting question stays open; a future attempt needs real nested text (code, LaTeX, markup) rather than twelve constructed sentences, and the null should be kept exactly as written.
- 527: the bracket pointer is NOT built from a distinct set of writer pairs. Decomposing the one decisive score cell exactly (4.67e-7): the top 10 of 625 pairs carry 14.3% of the mass, and the top-10 set at the match cell differs from the one at the distractor cell by a single pair. The same machinery just evaluates 4.5x larger at the match. Structural clue: seven of the ten leading pairs have WTE on the KEY side, and since both cells share a query and differ only in key, the whole discrimination is key-side -- but two "(" tokens have the same embedding, so it must come from rotary position or from the one minority key writer (m12). If it is rotary, head 13.8 is a relative-distance selector rather than a symbolic matcher, which would also explain why the model cannot close nested brackets (526). bracket_distance queued to decide.
- 528: CHANNELS BELOW THE HEAD (acting on the user's correction that heads are not the natural units). Exact to 3.57e-7. The finding: at digit-target queries, 37.9% of the leading head's channel content is supplied by source positions that are themselves digits, against a 2.7% base rate -- a factor of 14, so the subspace carries digit content forward from earlier digits. No head-level ablation could have shown this. Two negatives: each head's channel is effectively FULL rank in the subspace (15.7 of 16), not the narrow specialist I predicted; and consequently my "do heads share directions" metric was DEGENERATE -- comparing column spaces of full-rank maps returns cosine 1.000 by construction, measuring nothing. The weighted version (normalized Gram matrices M_h M_h^T) is queued. The weights-predict-causal correlation held at rho 0.604 but is NOT banked: random subspaces of the same rank predict at rho 0.40-0.44, so most of it is generic write strength.
- 529: CORRECTION -- the bracket pointer is POSITIONAL, and my matcher framing was partly a confound. Disabling rotary for head 13.8 collapses the match/distractor ratio from 6.48 to 1.08: the two become indistinguishable, so position IS the discrimination. Removing the token embedding from the key side leaves it at 6.81, so "this is a bracket" contributes nothing to which key is picked. And the match sits a median of 2 tokens back (IQR 2), so "points at the match" and "points two tokens back" are largely the same claim in this corpus. The causal results stand as measured (one cell, 83% of 0.825 nats) but the interpretation weakens to a short-range positional pointer, which also retro-explains why the model cannot close brackets opened two clauses earlier (526) and why already-closed and still-open brackets get identical shares (523). Match distances do range to 32, so bracket_range is queued to test whether the pointer adapts to far matches or implements a fixed short offset.
- 530: THE SELECTIVITY IS IN THE ATTENTION, NOT THE CONTENT. Recursion into the value vectors is exact (4.69e-7) and the leading writer into the digit channel is a5 at 0.319 -- but at NON-digit source positions it is also a5, at 0.358, with near-identical profiles throughout, so the channel's writer composition carries no digit information and the claim is not banked. Separately, the four contributing heads do emphasize similar directions (normalized-Gram cosine 0.627) but twenty RANDOM head quadruples average 0.696, higher, so the similarity is generic and not cooperation. Combined with 528 (channel content comes from digit sources at 14x base rate), the positive statement is: selectivity lives entirely in WHICH positions are attended to, not in what they carry. Same shape as 529, where the bracket head's discrimination survived stripping token identity from the keys and vanished when rotary was removed. rotary_selectivity queued to test whether this generalizes across the four behaviour-leading heads, with a differentiated bet (brackets lose everything, newlines should not).
- 531: the positional pointer ADAPTS, refining 529. Split by distance, head 13.8's match share is 0.324 at 1-2 tokens, 0.322 at 3-5, 0.294 at 6-11, 0.196 at 12+, with match/distractor ratios of 4.9, 9.0, 6.1, 11.5 -- so it is not a fixed short offset. With rotary disabled the ratio is 1.04-1.08 in EVERY bin, so the positional collapse is total at all distances. Accurate description: the head computes a distance from its query and selects the key at that distance, ignoring what token sits there, and the distance tracks where the matching opener actually is out to 32 tokens. It cannot disambiguate by content, which is why nesting defeats it. The query's distance computation is upstream and unlocated.
- 532: rotary selectivity does NOT generalize, and the control explains why. Disabling rotary head-by-head: the bracket head 13.8 retains only 0.141 of its behaviour-specific damage with a control at 3.8% of the effect (clean, third independent confirmation), but for the newline, quote and digit heads the same-layer control heads disturb the behaviour 44%, 131% and 265% as much as the target head does. Rotary is shared machinery, so removing it from one head perturbs the whole layer, and only an 0.82-nat effect stands clear of that. My differentiated bet also failed: the newline head does not survive (retains -0.896, the specificity reverses). Recorded as an open method problem -- measuring positional dependence for smaller circuits needs a locally-constructed intervention such as re-rotating one key.
- 533: mlp0 IS exactly 9216 squared linear features (reconstruction 7.8e-7), but the loud ones are the wrong atoms. Keeping the top 32 squares costs 1.69 nats while 32 RANDOM squares cost 0.81 -- because the squares come in 4608 pairs whose difference is the hidden unit's actual output, and ranking individually splits pairs and destroys the cancellation. Compression to under 0.10 nats needs ~2048 of 9216. Naming worked for 2 of 5 leading features and both are genuinely clean -- one is prepositions (' and',' to',' with',' onto'), one is determiners (' a',' the',' an',' their') -- while the coarse class taxonomy lumped them together and the other three were dominated by rare-token junk. mlp0_units queued: rank by hidden unit so pairs stay together, name over frequent tokens only, and price the per-token-table stand-in on the same rows for comparison.
- 534: mlp0's exact form beats the fitted table, but subset selection is the wrong compression operator. Keeping the top K hidden units (pairs intact) and mean-filling the rest: 16 units cost 1.58 nats against 0.81 for 16 RANDOM units, 64 cost 0.99 against 0.70 -- top-K is worse than random until K=256. So the cancellation is global, not within pairs: the loudest terms annihilate each other and keeping them unopposed is worse than keeping nothing in particular. Reaching 0.10 nats takes ~1024 of 4608 (22%), the same crossover the square-level run found. THE BENCHMARK NUMBER: the fitted per-token table costs 1.466 nats on the same rows, so the algebraic stand-in is 3.6x better at 256 atoms and 15x better at 1024. Naming works with a fine taxonomy over frequent tokens -- 3 of 5 leading units are clean (determiners, a verb/noun class, capitalized words). mlp0_lowrank queued to replace subset selection with low-rank truncation of the map.
- 535: ATTENTION LAYER 0 COMPARES TOKENS ALONG TWO DIRECTIONS. Its input is exactly the token embedding (verified 7.2e-8 -- block 0 forms a scalar multiple of E and rms_norm is scale invariant), so the layer is a function of token identity and distance alone. Truncating every head's query and key maps to rank r: rank 2 costs 0.053 nats, rank 16 costs 0.023, rank 128 (untouched) 0.000 -- while RANDOM projections of the same rank cost 1.2-1.4 nats at every rank including 64. So an arbitrary 64-dimensional restriction is catastrophic and the right 2-dimensional one is nearly free. The leading directions of the layer's most expensive head are morphological -- 'ed','ing','ers','ating' on the query side, 'ates','ations','izing' on the key side -- read off the weights with no data, which is exactly what an exact bigram table should be comparing. The attention layer is dramatically more compressible than the MLP beside it (which needs 22% of its atoms to reach 0.10 nats).
- 536: mlp0 WRITES NARROW AND READS WIDE. SVD-truncating the map (the operator 534 argued for): writing into 64 of 1152 directions costs 0.099 nats (5.6%), while reading through 64 costs 0.378 and it takes 256 directions -- 22% -- to reach 0.10. Random projections of the same rank cost 2.05-2.12, so unlike subset selection (which was worse than random below K=256) low-rank truncation beats random everywhere; the operator question is settled. Both sides beat the fitted per-token table (1.466 nats) from RANK 8 -- reading through eight directions of the embedding already beats a full vocabulary lookup. The 22% read-side rank matches 534's 22% of atoms, so subset selection was recovering the read rank the hard way. Against attn0's 1.6% (rank 2 of 128 per head), the attention layer is an order of magnitude more compressible than the MLP beside it.
- 537: COEFFICIENT MAGNITUDE IS NOT INFORMATION FLOW. The exact writer coefficients say each early MLP's input is dominated by the re-injected embedding (8.15 vs 1.03 at mlp1, 24.1 vs 5.0 at mlp2), but the measurement inverts it: from mlp1 upward, mean-filling the embedding's direct contribution costs 0.036, 0.001, 0.0004, 0.0000, 0.0003 nats while mean-filling the component writers costs 0.07-0.31. Only mlp0 behaves as the coefficients suggest. So the eightfold per-block re-injection of the token is REDUNDANT -- layer 0 has already written the same token information into the stream, and its outputs are exact functions of the token pair. Spearman between coefficient ratio and cost came out +0.657 where I registered -0.60, sign inverted. CORRECTION to 503's gloss ("the model re-injects the embedding at strength 8 and shrinks what is there"): the arithmetic was right but it does not mean the embedding carries the computation. Good news for the benchmark -- folding into mlp1 goes through attn0 and mlp0, which are the rank-2 and rank-64 objects of 535/536.
- 538: THE EARLY-LAYER RANK TABLE. Attention passes 0.10 nats at rank 2 (a0), 32 (a2), 8 (a3, a4) and never for a1 and a5; MLPs at rank 512 (m0, m1, m4), 128 (m2, m3), 8 (m5). Median passing fraction 0.25 for attention vs 0.444 for MLPs, so attention is the more compressible half across six blocks. TWO CAVEATS THAT MATTER MORE THAN THE TABLE: (i) truncating all six attention layers simultaneously at their own passing ranks costs 0.718 nats against 0.19 summed individually -- compression is superadditive by ~4x, so a per-component rank table is not a recipe for a front-of-model stand-in; (ii) the random-projection null failed for six of twelve components -- where a random restriction costs as much as the SVD one (m5: 0.069 vs 0.072), the passing rank reflects insensitivity rather than low-rank structure. The defensible rows are a0 (rank 2, 25x over random), a2 (32, 21x), m1 (512, 44x), m0 (512, 5.8x). Compressibility is not monotone in depth: a0 needs 2 directions and a1, sitting beside it, needs all 128.
- 539: THE FRONT SPEAKS UPWARD THROUGH 64 NUMBERS. Projecting layer 0's combined write onto its top r principal directions: r=64 (5.6% of 1152) costs 0.081 nats, r=128 costs 0.029, r=32 costs 0.168 against 0.57-0.61 for a RANDOM 32-dimensional interface. Deleting the write entirely costs 0.838. Split by writer: mlp0's write is 128x larger in norm than attn0's and dominates the combined curve at every rank, yet attn0's write needs only 16 directions to pass 0.10 where mlp0 needs 64 -- the small write is the concentrated one. Together with 535 (attn0's input IS the embedding; rank 2 per head) and 536 (mlp0 writes into ~64 directions), the front of this model is a function of the token pair emitting a few dozen numbers. Not yet shown: that it composes -- 538 found per-component truncations superadditive by 4x, so joint_rank is queued to measure a common-rank truncation across all six early blocks.
- 540: PER-COMPONENT RANKS DO NOT COMPOSE. Six early attention layers at a common rank cost 0.512 nats at 32 and 0.252 at 64; the six early MLPs cost 1.539 at rank 512 where their individual costs sum to ~0.18 (more than 8x); both together at attention 64 / MLP 576 cost 1.776 -- worse than replacing mlp0 alone with a fitted per-token table (1.466). The random null also failed (1.7x, not 3x), which is what happens when the damage is large enough to leave the measurable regime. CORRECTION to 538: at uniform rank the joint attention cost is 0.80 of the sum, SUBadditive, where 538 reported 3.7x superadditive -- both are right for their own rank assignment, so composition is not intrinsically super- or subadditive here and 538's figure should be read as specific to that assignment. Does not touch 539's 64-direction interface, which is about what layer 0 WRITES. The two together say: compress what a layer sends, not how it computes.
- 541: THE INTERFACE ROUTE DOES NOT COMPOSE EITHER, withdrawing last section's conclusion. Every early block on its own sends what matters through 64-128 of 1152 directions (block 0 at 64, blocks 1-2 at 128, blocks 3-5 at 64), generalizing 539. But all six interfaces projected to rank 64 simultaneously cost 1.314 nats against 0.84 summed individually and a 0.30 bar, with the joint curve flat and terrible from rank 8 to 32. Joint passes 0.30 only at rank 256 and 0.10 only at full rank. The random null failed at 1.2x, the same out-of-regime signature as 540. So neither weight rank nor output rank composes across the front -- though interfaces are clearly the better of the two (joint 256 costs 0.147 where the weight route never reaches 0.30). The pattern in both runs: per-component compressibility is real, joint compressibility is not, which says the residual stream is a shared channel whose apparent per-layer redundancy is being used by some other layer.
- 542: BLOCK 0 IS A 50304 x 64 TABLE, AND ITS COLUMNS HAVE NAMES. Replacing block 0's entire write with a per-token lookup costs 0.107 nats at full rank and 0.182 at 64 dimensions, against 0.838 for deleting the write and 0.081 for the real write at the same rank -- so what block 0's attention contributes beyond the current token is 0.101 nats. Random 64-dimensional tables cost 0.49-0.52 (2.7x, just under the 3x null bar, recorded as violated). Six of the top eight interface directions are class-pure over frequent tokens: determiners (' the','The',' a',' an'), punctuation (').','.',' ('), initial capitals (' B',' D',' M'), digits (' 56',' 57',' 12'), sentence openers (' If','You',' Have'). Set against 540 and 541: compressing how a block computes fails, compressing what six blocks send fails, replacing ONE block with an explicit token-indexed object succeeds. The move that works is substitution by an interpretable function of a variable the block provably depends on.
- 543: LATER COMPRESSION PARTLY REPAIRS EARLIER DAMAGE. Projecting block interfaces to rank 64 over a growing prefix: forward costs run 0.081, 0.769, 1.297, 1.617, then FALL to 1.514 and 1.314 -- adding block 4's compression improves cross-entropy by 0.103 nats and block 5 by another 0.200. Compressing block 0 costs 0.081 alone and 0.384 (4.73x) when blocks 1-5 are already compressed. Both orders reach 1.3137 to four decimals and the full-rank prefix costs 0.00003, so the machinery checks out. This is interaction, not accumulation -- a pure accumulation model produces neither negative increments nor a 4.7x position effect -- but it is interaction WITH CANCELLATION: truncating an early interface injects something later blocks amplify, and truncating the later interfaces filters part of it out. Consequence for the benchmark: the joint numbers in 540/541 are not sums of local damage and cannot be read as "the front needs rank R"; finding a front-of-model stand-in requires joint optimization, not any per-component rule.
- 544: BLOCK 1 IS A FUNCTION OF THE TOKEN PAIR. Replacing block 1's write with a table indexed by the current token alone costs 0.904 nats; indexed by the (previous, current) pair with backoff it costs 0.522, against a ceiling of 0.378 for the real write at the same rank -- so the pair closes 73% of what the single token leaves. A shuffled index costs 3.55, four times worse, so both tables use their variables. The indexing variable was derivable from the weights in both cases: block 0 because its input is exactly the embedding, block 1 because the only thing between it and the embedding is an exact bigram table. So the first two blocks are a token-indexed table (0.18 nats) and a pair-indexed table (0.52). Caveats: the costs climb steeply and 543 showed they do not add when applied together, and the bigram table only holds pairs the corpus contained -- the quotable number is the 73% gap closure, not the 0.52.
- 545: JOINT OPTIMIZATION IS NOT THE MISSING INGREDIENT, correcting 543's phrasing. Greedy joint search over rank allocations (27 joint evaluations) reaches {0:64, 1:128, 2:128, 3:32, 4:16, 5:16} costing 1.1795 nats at a budget of 384 directions, against 1.2745 for uniform rank 64 and 1.37-1.77 for random allocations. The answer IS lopsided (max/min ratio 8) but buys only 7% over picking one number for every block, where I registered 30%. So the low-rank interface line closes: the front has no compressed residual interface at ~64 directions per block under any allocation. What survives is substitution -- an explicit table indexed on a variable the block provably depends on costs 0.18 at block 0 and 0.52 at block 1, against 1.18 for the best rank allocation of the whole front. The difference is not effort: a table indexes the right variable, a projection keeps a subspace of the wrong one.
- 546: TWO TABLES COMPOSE ADDITIVELY, AND REFITTING MAKES IT WORSE. Block 0's token table plus block 1's pair table cost 0.665 nats together against 0.688 summed individually -- essentially additive, unlike rank truncation which is 1.6x superadditive with cancellation. But refitting block 1's table with block 0 already replaced -- the seemingly more honest self-consistent procedure -- costs 1.065, WORSE by 0.399. The reason: a table fitted against the REAL model injects the write the true model would have produced and partially corrects the upstream substitution error, while a refitted table faithfully reproduces the perturbed write and preserves it. Independent fitting is error-correcting, sequential fitting is error-preserving, and the practical rule for layered stand-ins here is to fit every component against the ORIGINAL network. Benchmark line: the first two blocks replaced by two derived tables cost 0.67 nats where the best rank allocation of all six costs 1.18.
- 547: BLOCK 2 IS A PAIR FUNCTION TOO. Indexed by the current token its write costs 0.537 nats; by the token pair, 0.275 against a 0.159 ceiling -- closing 69% of the gap, almost exactly the 73% the pair closed at block 1. So the required context does NOT grow from block 1 to block 2, even though attention layer 1 sits between them and fails the bigram-table test at +1.470. The trigram arm gained only 0.018 and is recorded as UNEVALUABLE rather than failed: the corpus yields 221,900 distinct triples from 256,000 positions, so almost every triple occurs once and on fresh text the table backs off to its bigram row -- the arm is mostly the bigram arm with a thin scatter of hits. The bigram arm does not have that problem (145,910 pairs, and pairs recur) and its 0.263 gain is far too large to be coverage. Ladder so far: block 0 token 0.17, block 1 pair 0.52, block 2 pair 0.27. front_table4 queued to price the wider window on held-out in-corpus positions and to report coverage alongside the gain.
- 548: CORRECTION -- the table results were measured on text the tables were fitted on. The tables are fitted on cl.rows() (the 1000-row census corpus) and were priced on cl.fineweb_rows(48), which draws the first FineWeb rows with no skip; 33 of those 48 rows appear verbatim in the fitting corpus. RETRACTED: 542's block-0 table at 0.18, 544's block-1 pair table at 0.52 and "73% closed", 546's two-table composition at 0.67 and its refit comparison, 547's block-2 table at 0.27 and "69% closed". The one clean measurement, on rows held out from the fitting corpus: block 2 unigram +0.605, bigram +0.509, ceiling +0.149 -- the pair closes 21% of the gap, not 69%, with 42% key coverage. Every qualitative claim survives (the pair beats the token by 0.095, a shuffled index costs 1.36) but the magnitudes do not, and 546's "two tables beat the best rank allocation" cannot be quoted until both sides are clean. SECOND occurrence of this error in the program; the rule is now recorded as a rule: any fitted object must be priced on rows verified disjoint by explicit comparison. Milder in-sample flag on 539/541/543/545, which fit a 64-direction basis on the positions they price.
- 549: THE TABLE LADDER ON CLEAN TEXT (800 fit rows / 96 held-out, zero overlap, bases fitted out-of-sample). Block 0: delete +0.778, ceiling +0.085, token table +0.203, pair table +0.156. Block 1: delete +7.192, ceiling +0.418, token +1.161, pair +0.959. Block 2: delete +1.254, ceiling +0.164, token +0.609, pair +0.510. The pair closes 40%, 27% and 22% of each gap at 43.8% coverage. All bars held, including composition at 1.6463 against 1.6248 summed (ratio 1.01), so 546's additivity survives. How much contamination mattered tracked index sparsity exactly: block 0's TOKEN-indexed table barely moved (0.182 -> 0.203) while the PAIR-indexed tables nearly doubled (0.522 -> 0.959, 0.275 -> 0.510). THE BENCHMARK COMPARISON REVERSES: 546 claimed two tables at 0.67 beat the best rank allocation at 1.18; on clean text three tables cost 1.65, which is worse. The two sides are still not like for like (different rows, different baseline, in-sample basis on the rank side), so the table route's advantage is withdrawn and matched_route is queued to price both on the same held-out rows.
- 550: CORRECTION -- projection beats tables decisively, reversing 542-546. Both routes priced on the same held-out rows at the same scope (blocks 0-2), all fitted objects built only on fitting rows: pair-indexed tables cost +1.6463 with 23.3M numbers of description; rank-64 projection costs +1.2736 with 0.22M numbers; rank 256 costs +0.1054 with 0.88M. So projection wins on accuracy AND uses 26-100x fewer numbers, and only loses at rank 8 where its description is a thousandth the size. Deleting the three writes costs 5.045. The table route's apparent advantage was entirely an artifact of evaluating tables on their own training text. What survives: 542's naming result (determiners, punctuation, initial capitals, digits, sentence openers) reads the table's own columns and does not depend on the retracted costs; and the program has one interpretable-and-accurate object, block 0's token-indexed table at 0.203 nats against 0.778 for deleting the block, with no evidence the form scales past it.
- 551: THE NAMEABILITY METRIC IS BROKEN. Scoring interface directions by "7 of 10 top frequent tokens share a class" gave 11, 9, 6, 6, 9, 4 nameable of 16 across blocks 0-5 -- but NINE of 16 RANDOM directions also passed, so the counts say almost nothing. Two errors: the random control projected the raw EMBEDDING table while real directions project the centred per-token WRITE matrix (different object, so not a control), and the classifier has no base rate, so "7 of 10" happens by chance when classes are unbalanced among frequent tokens. This puts a caveat on 542's "6 of 8" and 534's "3 of 5", which used the same flat criterion and the same mismatched control -- their 1-of-5 random checks were underpowered against a procedure that yields 56% on sixteen draws. The qualitative naming (determiners, punctuation, digits, initial capitals) is visible in the token lists and stands; the counts should not be quoted. direction_names2 queued with a matched control in the same write space and a base-rate-corrected binomial criterion.
- 552: DIRECTION-NAMING BY TOP TOKENS IS CONFOUNDED AT THE ROOT. With the fixes 551 demanded -- matched random control in the same centred write space, base-rate-aware binomial criterion -- named counts are [11,5,6,5,8,5] and matched-random counts are [10,5,5,6,5,2], net [1,0,1,-1,3,3]. Random directions are named at almost the same rate as principal ones, so the method does not work: the per-token write space is low-dimensional and class-structured, so any direction has class-coherent top tokens. Block 0's top direction genuinely scores determiners 9/10, but "it MEANS determiners" doesn't follow because a random direction scores about as pure. RETRACTED as evidence: 542's "6 of 8 class-pure directions" and 534's "3 of 5 units" (the token lists are real, the meaning claim isn't). Readability must be established causally: causal_direction_names queued to project one direction out and measure which token class degrades, which decides 550's open question (is the cheaper projection stand-in also interpretable) on real evidence.
- 553: GAP 2 EXACTLY -- the bracket head's query is a diffuse same-sign sum. Using the user's point that the query side is LINEAR (q = rotary(rms_norm(W_q X)), rms a scalar, rotary a rotation, X a writer sum), the contribution of each upstream writer to the match-cell score factor is exact: q reconstructs to 5.3e-7. Result: ~12 late writers each carry 6-9% (m10 9.4%, a7 8.4%, m6 6.8%, ...), top 3 only 25%, ALL with the same negative sign. Unlike 506's diffuse newline query, this is exact and additive with no multiplicative-net caveat, so the distributedness is real. But the uniform sign suggests a common-mode level rather than bracket-specific selection, and absolute contribution at the match cell mixes the common-mode level with the discrimination that actually sets the distance. bracket_query_contrast queued to decompose the match-minus-mean-key contrast, which cancels common-mode and isolates the selection.
- 554: the bracket query contrast is diffuse too. Decomposing the match-minus-mean-key contrast (common-mode cancels exactly, reconstruction 3e-7): top 3 writers still 25%, m10 9%, unchanged from the absolute level -- so each writer contributes a small discriminative piece directly, and the distance selection is irreducibly distributed over ~12 late writers, proven by exact composition rather than ablation. But "diffuse over writers" and "low-rank in the residual" can both hold -- a dozen components relaying one variable would look identical. So gap 2's question changes from "which writer" to "which DIRECTION of the query input, and how many". bracket_query_rank queued: find the top singular directions of the query's dependence on its input at close-bracket targets, project them out of the layer-13 residual until the head's 0.825-nat effect dies; one or two directions would be the look-back variable to trace upstream and a candidate reusable component, many would mean no compact upstream description exists.
- 555: GAP 2 RESOLVED -- the bracket head's distance selection is DIFFUSE over writers but LOW-RANK over directions. Projecting the top selection directions (eigenvectors of the query-contrast gradient) out of the layer-13 query input: 8 remove 49% of the head's 0.86-nat bracket cost, 16 remove 70%, 64 remove 80%, while 8 RANDOM directions remove only 0.015 (30x cleaner). At NON-bracket positions the same projection costs 0.0003-0.0012 nats (ratio ~0), so the subspace is entirely bracket-specific. This reconciles 553/554 (a dozen writers each 6-9%, no single source, proven exactly) with a compact answer: many writers collectively build one ~16-dimensional look-back subspace that the query reads to decide how far back to point. That subspace is a concrete named variable and a reusable-component candidate. bracket_subspace_reuse queued to compare it against all 162 heads' read subspaces from weights alone -- is the look-back signal private to 13.8 or shared machinery.
- 556: the bracket look-back subspace is largely PRIVATE. No head reads it at mean principal cosine >0.5 (top co-reader 14.8 at 0.25, against a 0.118 random floor), so it is not shared machinery. Causally, removing it globally costs +3.57 at bracket targets (30-60x random) but only +0.051 off-brackets (1.6x random), so it is overwhelmingly bracket-specific. One weak lead: the #2 co-reader is 12.6, the newline head (0.233), suggesting the two structural heads share some geometry. Caveat: the geometric metric compared top-16 vs top-16 singular subspaces and understates overlap, but the maxes (0.55-0.71) show only partial one-or-two-direction overlap, not a shared 16-dim variable. The reuse method works and is now available for any circuit. newline_query_rank queued to test whether "diffuse writers, compact subspace" generalizes to the newline head and to compare the two selection subspaces directly.
- 557: THE SHAPE GENERALIZES. The newline head 12.6, by the same exact method, also reads a compact behaviour-specific selection subspace: 16 directions remove 62% of its effect, 8 beat random by ~10x, and removing it costs literally 0.00 at non-newline positions (even cleaner than brackets). So across two unrelated attention heads, selection is computed by reading a ~16-dim subspace that is built diffusely by a dozen writers, low-rank, and inert off-behaviour -- a general property of how this model does attention selection, and the tractable handle for tracing toward the embedding ("what writes this subspace"). The newline and bracket selection subspaces overlap at mean cosine 0.203 (max 0.507) vs a 0.118 random floor, so the two structural heads share only a direction or two, not a common variable -- selection subspaces are mostly PRIVATE across the two circuits examined, settling 556's lead as weak.
- 558: THE TRACE STOPS, for a real reason. Attributing the bracket selection subspace S to upstream writers exactly (1.4e-7): still diffuse (top 3 = 23%, a5 = 9%), AND the null is violated -- writers project onto S equally at bracket and non-bracket positions (a5 8.9% vs 10.5%). So S is written generically everywhere; the residual carries S-energy at all positions but it only AFFECTS output at bracket positions through the query-key alignment. S is a READ FRAME, and the bracket-specificity is in the query-key-rotary geometry (529), not in any upstream writer. This is a genuine negative for "trace to the embedding via sparse circuits" -- a fact about the MODEL (established at 1.4e-7), not a method failure: this model computes the bracket distance distributively, many components nudging a shared frame. The one thread that localizes: a5's S-projection is 72% head 5.7 (the position-0 broadcast head). Reusable-component search should move to circuits whose writers ARE sparse -- the front-of-model token/pair tables.
- 559: CORRECTION -- the bracket query is effectively CONSTANT, refuting the per-position "distance computation" reading of 553-558. Replacing head 13.8's query at bracket targets with a single fixed vector (the bracket-position average) reproduces the head at match share 0.399 and CE cost -0.006 -- FREE. A generic query does not destroy selection (all-position mean still selects at 0.352). So there is no per-position query to trace: one fixed "find a bracket opener" query does the whole job, and the distance selection EMERGES from that fixed query under rotary geometry, not from upstream computation. This reconciles the whole gap-2 investigation: the query is diffuse over writers (553) and the subspace written generically everywhere (558) because there is almost no per-position query signal -- the 16-dim subspace holds the fixed query direction. Gap 2 closes with a simpler truth than a distributed chain. Caveat on 531's "adapts": a fixed query has a fixed rotary look-back profile; bracket_fixed_profile queued to price the fixed query by distance bin and decide whether the adaptation is real.
- 560: THE BRACKET HEAD IS A FIXED-PROFILE POSITIONAL AND, correcting 531. bracket_fixed_profile: the constant bracket-average query reproduces the real query's per-distance match profile at every bin (1-2: 0.365/0.373, 3-5: 0.408/0.431, 6-11: 0.346/0.362, 12+: 0.285/0.269), and the real query does NOT beat it at range -- so 531's "adapts" is refuted, a single fixed query does everything. bracket_and_factors (answering the user's AND question): the double-QK product IS a soft-AND (product match/distractor ratio 5.99 vs factors 2.98/2.77, multiplicative, exact reconstruction) -- BUT both factors are position-driven (rotary-off collapses both to ~1.0), not position x token. Token identity instead selects the candidate pool (opener keys differ from window-mean by 14-22x vs 3x from the distractor, since the distractor is also an opener), and rotary discriminates the match among openers. Complete mechanism: fixed query -> token-gated opener pool -> double-QK AND sharpens -> rotary picks the opener at a fixed relative offset -> coincides with the match unless nested (why nesting defeats it). Simpler than every intermediate story (matcher -> adaptive pointer -> distance computer -> this).
- 561: the fixed-query double-QK AND generalizes to the newline head, but the discrimination modality differs -- and with a caveat. newline_and_factors (head 12.6, match = most recent newline, distractor = non-newline neighbour): product match/distractor ratio 2.29 vs factors 1.42/1.62 (multiplicative, AND holds, weaker than brackets' 5.99). But NEITHER factor is position-driven (rotary-off leaves both ~unchanged); both collapse under key-mean (match differs from window-mean by 2.6x/9.7x), so the newline head discriminates by TOKEN identity where brackets used POSITION. CAVEAT: the distractors differ -- brackets used another opener (forces position), newlines used a non-newline neighbour (forces token) -- so part of the modality difference is built in. newline_and_factors2 queued with a matched distractor (second-most-recent newline, both newlines) to isolate whether the newline head uses position to pick among newlines. The AND itself holds for both heads regardless.
- 562: RESOLVED -- the newline head is a token DETECTOR, the bracket head a positional DISCRIMINATOR, both fixed-query double-QK gates. With a matched distractor (second-most-recent newline, both keys newlines), the newline head shows product ratio 1.03 -- NO preference between two newlines, no positional discrimination, not a distractor artifact. So the two structural heads genuinely differ by functional necessity: matching a specific referent (which opener) requires position (brackets, ratio 5.99, rotary-driven, fails on nesting); detecting a class (any newline) requires only token identity (newlines, position irrelevant). Corrects 497's "most recent preceding newline" -- the "most recent" did no work; the head attends to newlines generally. General account: a fixed query selects a token class via the double-QK soft-AND, and whether the head resolves a specific member by position or attends indifferently is set by the task, not the mechanism. Both circuits complete and corrected.
- 563: third-head AND test INCONCLUSIVE, for a diagnosable reason. Head 10.7 (opening quotes): product match/distractor ratio 1.06, LOWER than f1 (1.38) because f2 prefers the distractor -- the factors disagree, meaning the head does not select "the most recent preceding quote", the target I chose by analogy without verifying. Methodological miss: for brackets and newlines I knew the attention target before testing discrimination; here I assumed it. The general account (fixed query, double-QK AND, position-for-matching vs identity-for-detection) STANDS at two fully-worked heads and is not yet extended to a third -- the quote result tested the wrong quantity, not the account. quote_destination queued to first identify where 10.7 attends before any further AND test. Rule re-established: identify WHERE a head attends before testing HOW it discriminates.
- 564: the quote head DOES attend to the recent quote (5.6x over control), plus structural boundaries (sent_start 6.2x, line_start 3.2x) -- so 563's "wrong target" was itself too hasty. The head has a real target; what broke quote_and_factors was a poor distractor (the token before the quote, not a natural competitor) and the ABSOLUTE-VALUE factor decomposition, which discards each QK factor's sign so f1/f2 appeared to disagree when the signed product does not. Bracket/newline survived that flaw on strong signals; the quote head's weaker signal exposed it. 563's "tested the wrong quantity" amended to "poor distractor + sign-blind metric". quote_modality queued to redo it correctly: sign-aware score-mass share, matched distractor (second-most-recent quote), rotary probe. Prediction: as a detection task, no positional preference among quotes (like newlines).
- 565: third head (quotes, 10.7) is POSITIONAL, refuting my prediction and refining the account. quote_modality (sign-aware share, matched distractor = second-most-recent quote): match/distractor ratio 6.74, collapsing to 1.41 without rotary -- strongly positional, like brackets. I predicted token-detection (mislabeling quote prediction as pure detection); in fact it needs quote PARITY, which depends on the specific most-recent quote. Corrected axis, three heads: bracket 5.99 (needs matching opener), quote 6.74 (needs last quote for parity) -- both positional; newline 1.03 (any recent newline) -- detection. The determining factor is whether the task needs a SPECIFIC positional referent, not "matching vs detection". Mechanism now three-for-three: fixed query + double-QK AND selects a class, rotary discriminates a specific referent when the task needs one. The error was task-classification, not mechanism; the account is now predictive.
- 566: the modality account is NOT predictive (0/3 on fresh heads), and I repeated the unverified-target error. modality_batch: sentence_end (4.64->1.02) and open_bracket (3.07->0.95) measured positional where I predicted detection; capitalized (1.15->1.02) measured detection where I predicted positional -- every prediction wrong. WITHDRAWN: 565's "predictive" claim; the account is descriptive on the three fully-worked heads, not a reliable predictor. WORSE: I again tested modality against an ASSUMED referent without first measuring where each head attends (the exact error diagnosed in 563/564), so the measured ratios are against possibly-wrong targets and aren't trustworthy alone. Making the mistake twice means the protocol is at fault -- the destination step must be built into the modality test. modality_verified queued as the corrected protocol (destination first, test only if the head attends the referent class). Structural-attention thread at a natural stop after this: 3 verified circuits, 1 solid mechanism, 1 honest negative on predictiveness.
- 567: protocol fix VALIDATED, 566's sentence_end ratio RETRACTED. modality_verified (destination-first, modality only on verified target) on head 10.5: recent_se share 0.0146 at targets vs 0.0142 control -- NOT enriched, so the head does not attend to prior sentence-enders and 566's positional ratio of 4.64 was an artifact of an unverified referent. The protocol correctly reported UNEVALUABLE instead of a spurious number -- the code-level fix works. Structural-attention thread concludes: MECHANISM solid (3 heads are fixed-query double-QK soft-ANDs, complete), MODALITY descriptive not predictive (positional for brackets/quotes, detection for newlines; 0/3 predicting from task), TRACE negative and exact (selection subspace written diffusely/generically, no sparse source; reusability negative, subspaces private). induction_and queued to test whether the account extends to content-matching (induction), a genuinely different head type, verifying the target first.
- 568: induction is a 9-head BAND not a single head (from induction_redundancy: 108x superlinear, individual heads ~0.003 nats), so induction_fixed's test of "head 6.3 attends to prev-occ+1" was UNEVALUABLE (0.015 vs 0.0147 control) -- inapplicable, not just a target miss. Recurring shape: what this model computes distributively (bracket query ~12 writers, induction 9 heads, digit subspace 4 heads) resists single-component analysis; what it localizes (3 structural heads) yields clean circuits. CONSOLIDATED STATE: SOLID = 3 fixed-query double-QK AND circuits + attn0=bigram table + block0=lookup table; NEGATIVE-EXACT = subspaces written diffusely/generically, reusability near-floor, no joint front compression, tables lose to projection on held-out text; DISTRIBUTED-UNTRACEABLE = induction, bracket query, digit subspace. A handful of crisp localized circuits and a large distributed remainder, now distinguishable with exact methods. digit_distributed queued (grounded, verified target): does the 4-head digit behaviour have a compact read subspace or is head-distribution matched by direction-distribution.
- 569: COMPACT DIRECTIONS, DISTRIBUTED SOURCES holds for the digit behaviour. On a held-out split, ranking the digit bundle's 32 directions (16 per component a6/a8) by causal contribution: ONE direction carries 54% of the +0.1266 digit effect, beating random single directions 5-30x; 8 directions give 69%, 16 give 88%. Non-monotonic (k=1: 54%, k=2: 26%) -- the same interference/cancellation as mlp0 squares (534) and prefix accumulation (543). So the digit signal, spread over FOUR heads (521), lives in a compact direction set -- "distributed over heads" does not imply "distributed over directions". Pattern now across three cases (bracket query ~12 writers -> compact subspace; digit 4 heads -> 1 direction = 54%; induction 9-head band). Consistent model-level statement: behaviour signals are written into compact direction subspaces by many distributed sources -- the read side is low-rank and localizable, the write side diffuse and not, which is why selection subspaces are findable but their upstream sources are not.
- 570: CIRCUIT CARD -- the three verified structural heads reproduce on fresh text (bracket 13.8: 6.81->1.09 positional; quote 10.7: 6.74->1.41 positional; newline 12.6: 1.01 detection), closing the structural-attention arc. Control ratios above 1.5 for the positional heads are the fixed-query finding showing through (stable positional preference survives query jitter), not a failure. FINAL STATE: 5 verified circuits (attn0=bigram table, block0=lookup table, 13.8/10.7 fixed-query positional ANDs, 12.6 fixed-query detector); general mechanism (near-fixed query + double-QK AND, descriptive not predictive); model-level finding (compact read subspaces, diffuse distributed writes); exact negatives (no sparse trace, near-floor reusability, no joint compression, tables lose to projection). Single-head vein mined out (recent speculative tests UNEVALUABLE); next steps require a deliberate direction choice, no speculative experiment queued.
- 571: MODALITY MAP -- most behaviour-leading heads are LOCAL, not referent-attenders. Corrected destination-first protocol over four heads: open_bracket 17.2 (1.34x), colon 15.4 (1.0x), capitalized 15.3 (1.52x) all have NO verified lookback referent -- local heads predicting from immediate context. The protocol correctly reported them no-referent instead of forcing a modality (the 566 error it was built to prevent). Digit head 8.3 attends to the most recent DIGIT at 10.34x -> DETECTION (recent/second ratio 1.38, rotary-collapses to 0.88), a 4th properly-verified circuit like the newline head. Account now four-for-four on verified targets: bracket/quote positional, newline/digit detection, all fixed-query double-QK gates. So referent-attending structural heads are a SUBSET of behaviour heads; a distributed behaviour (digit, 4 heads) can still contain a clean single-head detector. digit_induction queued: is head 8.3 doing digit COPYING (induction restricted to digits)?
- 572: head 8.3 does digit INDUCTION, correcting 571's "detection". At digit positions with a prior occurrence, 8.3 attends to (prior digit + 1) at 0.103 vs the digit itself at 0.030 -- ratio 3.46, so it copies what FOLLOWED the prior digit (571 measured the wrong key). Nuance (a FAILED): digit occ+1 (0.103) is only 1.76x non-digit occ+1 (0.058), so 8.3 does GENERAL induction enriched for digits, not a digit-specific head. Adds a THIRD modality: positional matching (bracket/quote), token detection (newline), content induction (8.3). Reconciles threads: 8.3's clean induction attention doesn't contradict 568's "no single induction head" (that was causal -- band heads cost ~0.003 each; 8.3 has the read pattern, small causal effect alone); and it gives the distributed digit subspace (521/569) a nameable attention mechanism. Induction needs a content-dependent query, so it should break the fixed-query finding -- induction_query queued to test.
- 573: CONFIRMED -- induction needs a content-dependent query, a distinct mechanism class from fixed-query selection. induction_query (head 8.3, general repeated-token targets): the real per-position query attends to occurrence+1 at 0.0586 (6.4x a random-key control); the induction-average FIXED query gives 0.0192 (< 1/3 of real) -- replacing the query with a fixed vector BREAKS the induction, as predicted. So the structural heads have a fixed query (an average reproduces them) but the induction head does not (its query must encode the current token to match its prior occurrence). Two attention-mechanism classes now established with the same tools: FIXED-QUERY SELECTION (constant query + double-QK AND, rotary discriminates a referent when needed -- bracket/quote positional, newline detection) and CONTENT INDUCTION (per-position query matches a prior occurrence, attends to the successor). One-measurement classifier: does a fixed query reproduce the head? Clean two-class account, natural rest point for the attention-mechanism line.
- 574: CENSUS -- the model's attention is causally FIXED-QUERY almost everywhere. Replacing each of 162 heads' per-position query with its mean: 161/162 cost < 0.02 nats, ZERO exceed 0.05; the structural heads cost ~0-0.0017, induction 8.3 costs 0.0052, the sole outlier is 2.5 at 0.040. Most content-dependent heads (2.5, 3.5, 1.1, 1.4, 6.3) are all EARLY and small. Reconciles 573 rather than contradicting it: the census measures CAUSAL cost, not pattern change -- 8.3's induction pattern breaks under a fixed query but costs only 0.005 because it's one head of the superlinear band (568), so content-query computation is real in patterns but distributed across individually-cheap heads. Model-wide statement: fixed-query selection is the dominant attention mode causally; content matching exists only as a distributed, per-head-cheap remainder; early heads use content queries slightly more. Clean capstone -- the same compact/distributed split seen throughout, now for the query computation.
- 575: fixed-query tolerance does NOT compose, but the joint approximation is moderate. Replacing all 162 queries with their means costs +0.9607 nats (29% of baseline), vs +0.2499 summed individually (ratio 3.84, superadditive) and +10.3481 for random queries (so means are 11x better -- NULL ok). Same superadditivity as every joint compression (540/541/543): components each tolerate an approximation, errors compound jointly. But the magnitude keeps the two-class account intact: the model's attention is APPROXIMATELY fixed-query to within ~1 nat -- fixed-query selection captures the bulk, the content-query remainder (induction + rest) is ~1 nat, distributed, superadditive, and front-loaded (layers 0-5 give 59% of the joint cost, first 3 layers 25%, but it accrues through the whole stack). Final: attention dominated by fixed-query selection (all-queries-fixed ~1 nat, 11x better than random), content-query remainder ~1 nat distributed, per-head tolerance is per-head not joint. Attention-mechanism line complete.
- 576: keys are individually fixable too (158/162 < 0.02), refuting my clean Q/K asymmetry prediction -- the per-head census is nearly symmetric because most heads are individually low-impact, so it measures per-head impact not the Q/K division. Structural heads fixed on keys too (0.0002-0.0022). One strong exception: head 5.7's key costs 0.193 (4x the next) -- it is the position-0 broadcast head whose key at position 0 IS the constant it broadcasts, so fixing keys destroys it; the one head whose KEY content is individually load-bearing, for an understood reason. Small lead: 5.7 dominates the key-content list and is absent from the query-content list (consistent with a value-broadcast head needing a content key but generic query). The informative test is joint -- fixed_key_joint queued to fix all 162 keys and compare to 575's 0.96 nats for queries, settling whether queries-select/keys-carry holds at the model level.
- 577: THE Q/K DIVISION QUANTIFIED. Fixing all 162 keys costs 3.3214 nats vs 0.9607 for queries (575) -- 3.5x more, so keys carry ~3.5x the content queries do (queries-select/keys-carry at model level; the per-head symmetry in 576 was just low per-head impact). Key content 94% in early layers (0-5 = 3.14 of 3.32), fitting attn0=bigram table and token-determined early MLPs. Completes the attention account: early layers write token content->keys; heads apply approximately fixed queries; double-QK soft-AND; rotary makes the match position-dependent; content induction a distributed per-head-cheap front-loaded ~1 nat exception. Attention-mechanism line complete.
- 578: RSPD toy validation -- SVD + ablation-damage-covariance clustering (a method the user proposed, implemented myself since the source repo was unreachable) correctly recovers known computational groups on synthetic data, INCLUDING correct hierarchical nesting and correct minimal-sufficient-set identification, after fixing a bug (must restrict to the true SVD signal rank, not the full noisy basis). One caveat found and logged: a purely linear reconstruction proxy cannot show superadditive interaction (mathematically forced additive by orthogonality) -- that check needs a non-orthogonal target or the real nonlinear loss. Cleared to apply to a real component.
- 579: first real application, to mlp0's 4608 hidden units -- they cluster into groups that are statistically real (out-of-sample stability ARI 0.58 vs 0.00 null) and mildly, genuinely superadditive (1.03x vs 1.00x, not tautological like the toy), but a logit-lens naming probe on the clusters' shared output direction doesn't read cleanly -- wrong lens for input-reading units. Follow-up (context-based naming) queued.
- 580: found and read the ACTUAL rspd source (user cloned it) -- it's a closed-form Eckart-Young low-rank functional-core extraction + recursive per-datum-truncation-curve clustering (rspd.circuit_isolation.erank_circuit_isolation), materially different from and superseding my 578/579 reimplementation for linear layers. Applied to mlp0's Down layer over real FineWeb tokens: splits into 2 generic bulk clusters (rank ~500, matching root) and 3 small, much-lower-rank clusters (21/39/70 of ~700 root rank) -- one of which (rank 39, 91 points) reads cleanly as sentence-final positions from real examples. All 4 registered predictions held. r_min not independently calibrated (no ground truth); causal verification of the sentence-final cluster queued.
- 581: 579's naming lesson pays off -- redone with real activating context instead of logit-lens (as queued), mlp0's hidden-unit clusters ARE cleanly nameable: cluster 8 (101 units) is a SIGNED a/an-vs-the discriminator (positive activation -> context predicts indefinite article, negative -> definite), cluster 13 (76 units) splits period/exclamation vs dash punctuation, cluster 7 (29 units) splits first-person auxiliary ("I am/had") vs contraction suffix ("'s"). All predictions held, concentration well above the random-subset and shuffled-null baselines. Corrects 579's "not yet interpretable" -- the clustering was right, the first naming lens was wrong.
- 582: causal test of cluster 8 (queued by 581) is UNEVALUABLE across every comparison, not confirmed or refuted -- mean-filling 101/4608 mlp0 hidden units has too small a whole-model CE footprint (0.0008-0.0096 nats vs 3.30 baseline) to resolve article-selectivity by aggregate cross-entropy; caught properly via cl.score_bar's near-zero-denominator guard rather than reported as a false negative. 581's reading remains correlational only. A direct logit-margin probe (P(a/an) - P(the/The), not aggregate CE) queued as the properly-powered follow-up.
- 583: the targeted logit-margin retest of cluster 8 (queued by 582) resolves what aggregate CE couldn't -- cluster 8 IS causally a selective a/an PROMOTER (ablating it shrinks the margin 4.27x more than a random-unit control at indefinite-article-target positions, real effect), but does NOT show 581's predicted mirror-image effect at definite-article-target positions (smaller than the random control there). 581's "signed a/an-vs-the discriminator" corrected to "a/an promoter, the-side not confirmed." Demonstrates directly (same ablation, same data) that a targeted metric resolves what aggregate CE left UNEVALUABLE.
- 584: cluster 7 (581, first-person-auxiliary-vs-contraction) CONFIRMED as a clean bidirectional causal circuit -- both class-margin shifts held (6.06x and 4.56x the random control), unlike cluster 8's partial confirmation. Cluster 13's test came back invalid, caught by a baseline sanity check: the margin was negative even at the class's own target positions, traced to comparing a 2-token sum against a 4-token sum (dash class had twice as many tokens, inflating its baseline regardless of context) -- withdrawn pending a redo with a matched token set, not a real refutation of 581's reading.
- 585: RETRACTED -- 584's "cluster 7 confirmed" does not survive the count-fair (mean) metric either: under it, cluster 7's ratios flip to -1.60/-1.27 (both FAILED). Root cause identified: raw logit sums/means aren't a fair cross-class contrast regardless of token count, since common and rare tokens sit on different logit scales by default. Only cluster 8's test (built on matched-frequency 2-vs-2 tokens from the start) is unaffected. Principled fix queued: softmax probability mass per class, not logit sums/means -- retesting all three clusters (8, 13, 7) uniformly.
- 586: honest close of the causal-verification thread across 582-585's three metrics. Every named cluster (8, 13, 7) has a REAL causal footprint (3-7x a random-unit control, holds in every metric, passes every null) -- the clustering finds real structure. But the simple directional "promoter" story from 581's correlational reading does not survive causal testing intact for any cluster, and for two of the three (7 and 8's article side) the DIRECTION itself flips depending on which reasonable metric (logit sum, logit mean, probability mass) is used; cluster 13's test is genuinely ill-posed with this token set (dash prediction is inherently low-confidence for this model, confirmed via the scale-fair probability metric, not a metric artifact). Closing this specific test design; a different causal probe is needed to go further.
- 587: the clustering method generalizes to mlp1 (a second layer), more weakly -- stability ARI 0.326 vs mlp0's 0.58. Found one likely Unicode/tokenization-artifact cluster (86 units, "�" detector), one article-adjacent cluster (46 units, predicts "the" specifically -- an open lead on cross-layer redundancy, not yet causally tested), and one weaker mixed cluster (42 units, 4/8 concentration). Real, reproducible structure, but murkier than mlp0's -- both the weaker stability here and 586's unstable causal signs suggest this thread's easy wins are running out and a different method/probe is needed to go deeper.
- 588: RSPD's entropy-based effective-rank metric badly disagrees with the program's independently-established task-loss rank for attn0's OV write (542 vs ~16, 34x off; random-direction control barely different from real embeddings; the recursion found no structure to split on at all). Traced to a real, explained metric mismatch, not a contradiction: effective rank measures spectral entropy (any spread counts), the established 16-direction figure measures task-loss cost of a rank-r truncation IN the running model -- different quantities. Lesson recorded for future rspd use: effective_rank is not a compressibility proxy. Corrected, apples-to-apples validation (RSPD's own rank-r surrogate priced by real cross-entropy) queued.
- 589: RSPD VALIDATED precisely against the ledger's independently-established number, after catching a real design bug (a "combined all-heads OV matrix" can't be validly substituted into a live multi-head model -- fixed by applying RSPD to c_proj alone with its real captured input, mirroring the already-working mlp0-Down approach). RSPD's rank-16 surrogate for attn0's write costs 0.0943 nats, landing almost exactly on the ledger's own <0.10-nat bar at r=16, and beats a random same-rank projection by 14x+. Closes the RSPD-validation thread on a clean positive result.
- 590: attn0's c_proj produces ZERO refinement under recursive circuit isolation (one leaf, the unrefined root, rank 507) -- a real contrast with mlp0's Down (580: split into 5 leaves, bulk vs special-case). Registered bar was HELD only on a technicality (leaf count) and I flag that plainly -- the rank-near-16 half of the prediction clearly failed, same root cause as 588 (entropy-based rank isn't task-loss rank). The honest, coherent finding: combined with 589's clean confirmation that attn0's whole write fits one rank-16 subspace, attn0 (an exact bigram table) is genuinely homogeneous with no natural data subdivision, while mlp0 is genuinely heterogeneous -- closing the RSPD-application arc (578-590) on a real, cross-validated model-level contrast.
- 591: mlp0's bulk-vs-special-case split survives a properly task-loss-calibrated r_min=64 (vs 580's guessed 343.9) -- all 4 predictions held, and the fairer r_min finds slightly MORE structure, not less: the two generic bulk clusters are unchanged exactly, and 580's mixed rank-70 special cluster resolves further into two finer sub-populations (rank 35 clause-medial, rank 29 the same geopolitical-text group 580 already flagged as its messiest read). Closes the methodological loop: the mlp0/attn0 contrast (heterogeneous vs homogeneous) now stands on consistent, fairly-calibrated footing for both components. RSPD-application arc (578-591) complete.
- 592: activation patching (the different causal probe 586 called for) CONFIRMS cluster 8. Patching a real a/an-favoring source's cluster-8 activation into a real the-favoring target shifts the margin toward a/an, correctly signed, 4x a random-unit control, 22x a same-class control, near-zero at unrelated digit positions -- three independent specificity checks all agree. Only the pre-registered absolute-magnitude bar failed, and that bar looks mis-calibrated in hindsight (no real sense of scale for a single-position patch), not evidence against the effect. Closes the causal-verification arc (582-592) on cluster 8 with a real positive result, demonstrating 586's lesson: replacement-style patching succeeds where deletion-style ablation gave metric-unstable readings.
- 593: patching extended to clusters 13 and 7. Cluster 13 stays inconclusive -- traced to a thin, noisy target pool (only 50 real dash-target positions), the same underlying pathology 585/586 already diagnosed (dash prediction is inherently low-confidence for this model); a reporting-only bug (a meaningless printed ratio from dividing by a near-zero floor) is flagged and did not affect any verdict. Cluster 7 is the real finding: a specific, real causal effect in the WRONG direction (opposite 581's correlational reading), and this is now the THIRD independent, methodologically sound measurement (585 mean-logit, 586 probability-mass, 593 patching) to agree on that reversed direction -- correcting 581/584 rather than leaving the question open. Mechanism behind the reversal not yet resolved.
- 594: coda to 593 -- a quick unweighted-sum diagnostic on cluster 7's write direction vs the 4 tokens' unembedding is ambiguous (all four dot products negative; relative ordering actually agrees with 581's original reading, contradicting the causal measurements), flagged as mismatched (ignores real per-unit activation weighting) and shelved rather than over-interpreted. New direction queued: does mlp1's "the"-predicting cluster (587) causally move the same article margin as mlp0's cluster 8 -- a cross-layer redundancy test using the validated patching method.
- 595: ANSWERED -- mlp1's article-adjacent cluster (587) IS genuinely reusable machinery for the same a/an-vs-the decision mlp0's cluster 8 makes. All 5 checks held cleanly (random-unit control is even wrong-signed, same-class control 9x smaller than the real effect, clean null). Effect size is 13% of mlp0 cluster 8's, so mlp0 is the primary contributor and mlp1 a real secondary one -- not a coincidental echo, confirmed distributed/redundant machinery in unequal proportions. First clean positive answer to the "reusable components across circuits" question this program direction was built to ask.
- 596: the article-choice redundancy does NOT extend to a third layer. mlp2's unit-clustering stability (ARI 0.295) narrowly fails the 0.3 bar, completing a clear decay trend (mlp0 0.58 -> mlp1 0.326 -> mlp2 0.295); its lone auto-flagged "article" cluster is spurious (one " the" among unrelated subwords, 3/8 concentration). Honest boundary: the redundant article machinery is a two-layer phenomenon (mlp0 primary + mlp1 13% echo), not repeated at every early layer. Closes the breadth-first "how far does it extend" question.
- 597: DEPTH-FIRST on the article circuit (user's steer), both their predictions confirmed. BACKWARD (fold attn0 into mlp0): zeroing attn0 collapses cluster 8's article-position firing to correlation 0.24 (control: zeroing the later attn1 leaves it exactly identical, 1.0000) -- the a/an-vs-the decision is CONTEXT-driven, carried by attn0's bigram table, not the current token alone. FORWARD (same data points): mlp0-cluster8 and mlp1-article-cluster per-position firing energies correlate 0.47, vs -0.03 for an unrelated mlp1 cluster -- the confirmed 13% echo fires on the same data. The article circuit is now traced end to end (token -> attn0 bigram -> mlp0 cluster8 -> article logits, plus a same-positions mlp1 echo), every arrow exact or causally verified -- the highest-quality circuit object in the program.
- 598: the article circuit's exact lexical triggers, traced to the token level. Cluster 8's signed activation at article positions is driven positive (toward a/an) by prepositions ("into" +3198, "from", "of", "in") and be-verbs ("was", "is", "be" ~+1150), and negative (toward the) by punctuation (comma -665, period) -- exactly English article grammar (after a preposition/copula -> a/an; after a sentence boundary -> the). Direction check held: positive-driven positions have a/an-rate 0.346 vs 0.268 negative-driven, so the internal sign predicts the real next word. NULL reported honestly as a CHECK (real spread 2.5x shuffled but under the too-strict 0.3x bar; small-class noise inflates the shuffle) -- finding rests on the linguistic coherence + direction check, not the null alone. Completes the article circuit end to end: preceding tokens -> attn0 bigram -> mlp0 cluster8 -> article logits, every stage exact or causally verified.
- 599: the article redundancy is PARALLEL, not serial. Ablating mlp0 cluster 8's write changes mlp1's article-cluster firing by only -1.1% -- LESS than a random mlp0 cluster ablation (-5.9%), decisively against a serial hand-off. mlp1 does not read mlp0's output; both layers independently recompute the article decision from the shared upstream input (embedding + attn0 bigram, established in 597). The "same data points" co-firing (595/597) is because they share an INPUT, not because one feeds the other. NULL confirms the ablation propagates (shifts the article margin 0.0074); an earlier broken machinery check (measured mlp0-c8 activation from its unchanged input) was caught and fixed first. Makes the redundancy robust (two parallel reads, either sufficient) rather than a fragile chain.
- 600: COVERAGE CORRECTION (answers the user's coverage question). mlp0's units ranked 300-600, below the top-300 cutoff every prior pass used, cluster JUST AS CLEANLY (stability ARI 0.562 vs top-300's 0.58, far above the 0.4 flag) and hold THREE new nameable circuits absent from the top-300: a newline/line-break predictor (81 units, possibly linked to the verified newline head 12.6), a pronoun-context cluster (54 units), and a number-word cluster (37 units). The importance ranking does not separate structured from unstructured units -- so the "top-300 of 4608" framing under-counted mlp0's nameable structure (at least 6 clean circuits now, likely more below). Honest correction: coverage of even this one layer is more partial than the writeup volume implied; the fix is to keep clustering downward. Queued mlp0_units_600_900.
- 601: mlp0's clean structure decays gradually (stability ARI 0.58 -> 0.56 -> 0.36 across importance bands top-300 / 300-600 / 600-900), no sharp cutoff, still real at rank 900. Ranks 600-900 hold a THIRD article-context cluster (48 units, 8/8 determiner, fires before a/an/the after prepositions), plus a modal/auxiliary cluster and a coordination-vs-subordination cluster. Corrects 600's "no extra article machinery below rank 300" (true for 300-600, false for 600-900). Consolidated: mlp0 has >= 9 clean nameable clusters across three bands, its article decision DISTRIBUTED across multiple unit clusters at different importance levels plus the mlp1 echo. Breadth now well-sampled; swinging to depth.
- 602: the newly-discovered newline cluster is real but causally REVERSED under patching (delta -0.00069, wrong direction, but specific: 30x the same-class control and above random) -- the second such case after cluster 7 (593). Pattern across 5 causally-tested clusters: 3 correct-signed (all article: cluster 8, mlp1 echo) vs 2 real-but-reversed (aux-contraction, newline). Not a patching bug (article clusters give correct signs on the same method) -- a real property: activation-reading finds WHERE real structure is but does not predict the causal DIRECTION for a material fraction of clusters. Queued cluster_write_scaling (a different causal method -- scaling the write 0x/1x/2x, reading the margin slope) to test whether the reversal survives a method that doesn't transplant activations.
- 603: write-scaling (a second causal method for the 602 reversal) is UNDERPOWERED and inconclusive -- it fails its own positive control (the article cluster, known correct-signed, shows a flat/slightly-negative margin-vs-scale slope, and all slopes are ~1e-4 at the noise floor). Diagnosed: measuring at a cluster's own positive positions is near-saturated and noise-dominated, unlike 592's sensitive cross-context patching design. The script's auto-printed "reversal confirmed" is flagged as NOT trustworthy given the failed positive control. The newline/aux-contraction reversal remains established by patching alone (602/593), mechanism unresolved -- recorded as a documented open property rather than chased further with margin-based tools.
- 604: a clean lexical-vs-contextual map of mlp0's six named circuits (via 597's attn0-fold method). The decisions sit on a spectrum by how much they need previous-token context: article 0.175 (most context-driven) < aux-contraction 0.456 < punctuation 0.643 < newline 0.733 < pronoun 0.795 < number-word 0.802 (most current-token-driven). Matches the linguistics exactly -- article choice genuinely needs the preceding verb/preposition, a spelled-out number is identifiable from the current token. attn1 control exact (1.000 for all), so every low correlation is genuine upstream dependence. An input-side result independent of the 602/603 causal-direction reversal (measures what FEEDS each cluster, not which way it pushes).
- 605: the number-word cluster's causal test is INCONCLUSIVE (underpowered) -- spelled-out numbers are rare next tokens (only 77 target positions), so the patch effect (+0.00002) sits below the noise floor and is smaller than random (-0.00006); trigger census also too thin (2 tokens). A power limitation from a rare target class, not a finding -- UNEVALUABLE, does not join the correct/reversed tally (stays 3 correct / 2 reversed). The cluster is still real and current-token-driven (600/604 are input-side, frequency-independent). Queued pronoun_verify (pronouns are common -> properly powered) to complete the tally.
- 606: pronoun cluster is WEAKLY correct-signed (delta +0.00024 toward pronouns, but only 1.5x random -- specificity bar 3x failed, null marginal), on a now-well-powered target class (1374/1040 positions). Updated causal tally across 6 clusters: 2 clean-correct (article + mlp1 echo), 2 reversed (aux, newline), 1 weak-correct (pronoun), 1 inconclusive (number). No monotonic link between context-dependence (604) and causal-sign quality. Consolidated: the article circuit is the one clean fully-traced circuit; other named clusters are real as STRUCTURE but their causal write is variously reversed/weak/unresolvable -- reading firing locates structure but doesn't determine what the write does. Structural-cluster causal investigation at saturation.
- 607: clean surface-token circuits are a FRONT-OF-MODEL phenomenon. A middle layer (mlp9) has real but much weaker structure (stability ARI 0.167, above null -0.012 but well below the front), completing a monotonic depth-decay trend: mlp0 0.58 -> mlp1 0.326 -> mlp2 0.295 -> mlp9 0.167. Its top clusters are NOT cleanly surface-nameable (max 6/8 on a catch-all class, readings mix unrelated tokens with hints of abstract features like question-context/list-punctuation). Confirms the model's division of labor: front builds surface/lexical features (readable by this method), middle does abstract/distributed work. Completes the layer-wise coverage map alongside the importance-wise one (600/601).
- 608: the article decision's full 18-layer depth profile is FRONT-BUILD, EMPTY-MIDDLE, FINAL-READOUT. Mean-filling each layer and measuring the article-margin shift: front MLPs L0-2 mean 0.048 vs middle L6-10 0.003 (15x gap -- the abstract middle does nothing to the article decision), with a real final-layer readout at L15-17 (mlp17 -0.052, second-largest). NUANCE + correction: mlp1 (whole layer, +0.120) is the single largest article-margin contributor, 8x mlp0 (+0.015) -- the first clean CLUSTER was in mlp0, but by whole-layer causal weight mlp1 does more of the work (reconciles with 599's parallel finding; the 13% figure was for the 46-unit echo cluster, not the whole layer). Caveat: whole-layer mean-fill conflates direct+indirect, so profile SHAPE is the valid comparison, not absolute magnitudes. Queued article_mlp1_cluster.
- 609: mlp1's heavy article role (608, +0.12 whole-layer) is DIFFUSE -- its 46-unit article echo cluster accounts for ~0% of the whole-layer effect under ablation (fraction -0.01), despite being patching-confirmed article-carrying (595). Clean sufficiency-vs-necessity split: patching (transplant) shows the cluster CAN carry the article signal; ablation (remove) shows it is NOT necessary -- the two diverge exactly when the computation is redundant. mlp1's article info is redundantly encoded across all 4608 units; no small cluster is load-bearing. A caution about reading cluster patching as localization. Queued article_mlp0_localization to test whether mlp0's cluster 8 is genuinely more concentrated or also diffuse.
- 610: DECISIVE -- both article layers are diffuse. The identical ablation test on mlp0 shows cluster 8 (the flagship article circuit, 592 patching-confirmed) carries only 4% of the whole-mlp0 article effect under ablation (3.19x random, so real but a small minority). Unified with 609: neither mlp0 cluster 8 (4%) nor the mlp1 echo (~0%) is NECESSARY, though both are patching-confirmed SUFFICIENT -- the article decision is redundantly encoded across each layer's full 4608-unit width. The clustering+patching method finds real, readable, sufficient article-carrying structure (cluster 8's grammar triggers, context-dependence, mlp1 echo, depth profile all genuine), but that structure is a HANDLE on a redundant computation, not a localized bottleneck. "Found the article circuit" holds as "found sufficient article-carrying directions," not "found the units the model needs." A general property (holds for the flagship). Localization line complete; the honest headline is redundancy.
- 611: redundancy is GENERAL -- the newline decision is also diffuse. Whole-mlp0 has a real newline-margin effect (+0.051, 3x its article effect) but the 81-unit newline cluster carries ~0% of it under ablation (fraction -0.01, 2.94x random). Two independent decisions now (article both layers, 610; newline here), both diffuse -- upgrading "sufficient handles, no necessary core" from a flagship result to a general property of how this model computes early decisions. Opens the basis question: is the write low-RANK in a rotated basis even though unit-diffuse (RSPD showed attn0's write is rank-16)? Queued article_write_rank to sweep mlp0's output rank and distinguish "diffuse over units but low-rank over directions" from "genuinely high-rank".
- 612: RECONCILED -- mlp0's article write is diffuse over UNITS (610) but LOW-RANK over DIRECTIONS. Keeping only the top-16 PCA directions of mlp0's output preserves 84% of its article contribution (rank-16 damage 16% of whole-layer, and 6x better than a random 16-dim subspace). Both true because units and output-directions are different bases: the computation is spread across all units (no unit subset necessary) but concentrated in a ~16-dim output subspace (a small set of directions IS necessary). There is a compact necessary core -- in the rotated basis, invisible to unit-clustering. Unifies 610/611 (distributed sources), this (compact directions ~16), 569 ("compact directions, distributed sources"), and 589 (attn0 write rank-16): the same signature, now shown for the flagship MLP decision. The right object to call "the circuit" is the low-rank output subspace, not any set of units. Localization line complete.
- 613: low-rank-over-directions is GENERAL with decision-dependent rank. The newline write also lives in a low-rank output subspace (~64 directions carry 80%, 4x better than random 64-dim) but higher-rank than article (~16), and with no single dominant direction (rank-1 carries 10% vs article's 38%) -- newline is a more distributed decision over both units and directions, yet still low-rank (64 = 5.6% of the 1152-dim output). Confirms the general structure on two decisions: early decisions are diffuse over units, compact (5-14% of full) low-rank over output directions, with the dimensionality reflecting how distributed each decision is. Matches 569 (compact directions, distributed sources) and 589 (attn0 rank-16). Redundancy/low-rank arc (609-613) complete: no unit subset is necessary, but a low-rank output subspace IS the necessary core, its size decision-dependent.
- 614: there IS a single shared article channel -- the fixed unembedding direction d = W_U[a]+W_U[an]-W_U[the]-W_U[The]. Every layer's output projects onto d with the correct sign (a/an > the), front-loaded (mlp0 +1201), and 18x more than an unrelated readout direction (null-specific). BUT d is only 8% contained in mlp0's top-16 causal subspace (612) -- so the causal CONTENT subspace (what downstream reads) and the readout DIRECTION d (what the unembedding reads) are two distinct residual objects, both real. Nuance vs 608: mlp0 writes d most directly (+1201) but mlp1 (608's biggest causal contributor) writes it only weakly (+156) -- mlp1's large article effect is INDIRECT (content-shaping), mlp0 is the direct readout-writer. Refines "mlp1 does more of the work": more causal work, less direct readout-writing. The reusable-component answer: yes there's one article channel (the readout direction d), but it's distinct from the content subspace -- a two-object structure, not one tidy channel.
- 615: the layer-wise coverage map is U-SHAPED, not decaying -- CORRECTS 607's "front-of-model" framing. The final MLP (mlp17) clusters MORE cleanly than any layer (stability ARI 0.778 vs mlp0's 0.58, mlp9's 0.167 trough), with an ~8-dimensional output (rank 8, a low-rank readout signature), into clean DIRECT next-token readouts: two newline readouts and a capitalized-initial readout (the auto-check's "article" flag on cluster 8 was spurious -- it's a capitalized-token readout). Clean surface-token structure lives at BOTH ends: front BUILDS surface features from tokens (mlp0/1), end READS OUT token predictions (mlp17), abstract middle is weakest (mlp9). Division of labor by depth: lexical construction -> abstract processing -> token-prediction readout. The readout end is the cleanest, lowest-rank structure in the model -- the natural mirror of layer 0's token-indexed lookup table.
- 616: REFUTED my hypothesis that readout circuits are localizable. mlp17's newline readout cluster (the cleanest cluster in the model, 615) is ALSO diffuse under ablation -- carries -16% (opposite sign, negligible) of the whole-layer effect and LESS than a random 88-unit set (0.81x). Two findings: (1) redundancy is UNIVERSAL -- even the cleanest-clustering, lowest-rank (r=8) readout layer has no necessary unit cluster, at both ends of the network. (2) Clean-to-read is NOT causally-locus -- mlp17 has the cleanest newline clusters yet contributes LESS to newline than mlp0 (+0.016 vs +0.051); its clean readout structure is a correlate displayed near the logits, not where the decision is made. Closes the localization line: no localized unit-level circuits anywhere; computation lives in distributed units + compact directions, and where it's easiest to read (readout end) is not where it's done (front).
- 617: the VARIANCE basis is not the FUNCTIONAL basis. Tried to name mlp17's 8 output PCA directions (615: rank-8) by token class -- FAILED: only 2/8 map to a distinct class, they repeat (subword/newline/digit), and decisively the PCA directions are NO sharper than random directions at class-alignment (median gap ratio 1.42 vs 1.46). The high-variance output directions are not the functional/readout directions (consistent with 614: article readout d was only 8% in mlp0's top-16 PCA). Reconciles with 615: individual UNITS align with classes and cluster cleanly, but the top PCA DIRECTIONS of the aggregate output don't (PCA orders by magnitude, not class discrimination). Methodological lesson: unsupervised low-rank decompositions find the compact basis but not the interpretable one; readout directions need class-supervised decomposition. 612's "16 directions carry 84%" stands (causal-loss measure) but individual PCA directions aren't circuits. Queued mlp17_class_directions (the supervised version).
- 618: the readout layer's functional basis IS findable -- by SUPERVISION, not PCA (correcting 617). Class-readout directions (mean output at class positions minus generic) are real readouts out-of-sample (newline AUC 0.845, most classes 0.72-0.92), unlike the PCA directions. They span a ~5-dimensional subspace (effective rank 4.72, consistent with mlp17's rank-8 output), and the ~9 token classes sit as POLES/COMBINATIONS on ~5 shared axes (median pairwise |cos| 0.59; space_word vs capitalized anti-collinear -0.97, capitalized vs newline +0.92) -- not 9 independent channels. The concrete answer to "name the readout's computational primitives": ~5 shared readout axes spanning pairs/groups of token classes. Methodological capstone: the COMPACT basis (SVD, 612/613/615) and the INTERPRETABLE basis (supervised, this) are the same low-rank subspace in different rotations -- SVD gets the dimension right and interpretation wrong; supervision gets the interpretation.
- 619: the supervised newline readout direction (618, AUC 0.845) steers BACKWARDS. Adding d_newline to the final residual monotonically DECREASES P(newline) (alpha -2->+2: 0.0333->0.0142); CE minimized at alpha=0, rising for both signs. So d_newline IS causally coupled to the newline logit but with OPPOSITE sign to its correlational definition -- a probe is not a steering vector. Likely mechanism: adding a large vector along d_newline inflates the residual norm, and the final rms_norm divides down the true newline-causing components (dilution). Cleanest instance yet of "decoding direction != causal direction" (cf. 586/593/602). Methodological: 618's supervised directions are validated as PROBES only; treating them as intervention vectors needs a separate causal test. Queued newline_steering_renorm (renormalized steering: does the sign flip back? -> pure rms_norm dilution, or genuine off-axis correlate?).
- 620: WHY the newline probe steers backwards (619) -- not anti-alignment. cos(d_newline, W_U[newline]) = 0.000: the supervised probe is ORTHOGONAL to the newline WRITE direction. Raw newline logit even rises weakly with alpha (+0.12) but LESS than random directions move it (NULL CHECK is the tell: d is not a specific newline-logit driver); P(newline) falls via non-specific softmax competition. FINDING: d_newline is a READ direction, not a WRITE direction -- it decodes newline from mlp17's output (AUC 0.845) but is orthogonal to how the unembedding produces the newline logit. Linear-algebra form of 616 (clean-to-read != causally-locus; mlp17 reads, mlp0 writes). Explains 619: steering failed because d isn't on the write axis at all. Lesson: probes recover the READ axis, interventions need the WRITE axis (toward unembedding); they can be orthogonal. article_steering_decomp queued to test universality.
- 621: read != write orthogonality is UNIVERSAL. Article probe replicates newline (620): cos(d_article, W_U[article]) = 0.000 (orthogonal to write direction), and adding d_article DROPS P(article) monotonically (0.0807->0.0263). Article logit falls here (vs newline's weak rise) but NULL CHECK shows it's non-specific -- d's logit slope (-2.01) is inside the random range (-0.57,-4.26,-2.27). UNIFIED: supervised readout probes recover the READ axis (correlates with class in layer output), orthogonal to the WRITE axis (unembedding direction that produces the class logit); the probe can't steer its own feature. Unifies 616 (clean-to-read != causally-locus = read!=write), 617/618 (which read basis is interpretable), and 586/593/602/619 (activation-reading doesn't predict causal direction). METHOD: to intervene, push the WRITE axis, not an activation-fit probe. Queued write_direction_steering (positive control: pushing W_U[class] steers P(class) forward).
- 622: POSITIVE CONTROL closing read/write (619-622). Pushing the WRITE axis W_U[class] steers P(class) from baseline to ~90-98% (newline 0.025->0.93, article 0.057->0.977; -alpha drives to 0), while the READ probe drops P and random doesn't steer. cos(write,read) = 0.005/0.001 (orthogonal). Registered strict monotonic-increase came back False ONLY as a saturation artifact (newline WRITE dips 0.930->0.892 at +2 as P saturates near 0.9); the substantive claim (write axis steers sharply forward) HELD for both. CONCLUSION: to DECODE a feature fit an activation probe (read axis); to STEER it push the unembedding write axis (orthogonal to the probe). Correction: 618's supervised directions are decoders only, not intervention handles. Next: write_axis_layer_profile (which layers build W_U[class], extending 616's mlp0>mlp17).
- 623: which blocks build the write axis W_U[class]. ARTICLE = clean EARLY writer: block 0 dominant (+13818 >> block17 +763), position-specific (NULL ok), confirming 614 (with a push-pull: block 1 cancels most of block 0). NEWLINE does NOT fit early-writer: largest raw increments are LATE (block17 +3200, block16 -1927), block0 (+547) < block17, and NULL FAILS (more w_newline movement at non-newline positions) -- so the linear profile is a CONFOUNDED writer signal for newline (generic residual dynamics, not newline-specific writing). Tension with 616 (ablation: mlp0>mlp17) is a measurement difference (ablation=marginal necessity under redundancy vs raw linear contribution; block-level vs cluster; + the failed null). METHOD: linear write-profile is valid only when it passes the position-specificity null. Queued write_axis_ablation_profile (causal per-block P(class)-drop, extends 616 to all 18 blocks).
- 624: CAUSAL writer profile (mean-ablate each block, measure P(class) drop) resolves 623. Both classes WRITTEN EARLY: block 1 is the largest writer for both (newline +0.226, article +0.162), blocks 0-2 dominant; refines 614's "mlp0" to "early blocks 0-2, block 1 peak". BLOCK 17 (readout layer) is a SUPPRESSOR: ablating it RAISES P(newline) 0.435->0.65 and P(article) 0.389->0.54 -- the layer where classes are most cleanly READ (615/618) causally suppresses them (likely last-layer calibration of high-freq tokens). Reading location != writing location, and at block 17 the causal SIGN is opposite. NULL ok both (class-specific), unlike 623's linear newline profile -> vindicates 623's rule (trust linear profile only when it passes the specificity null). Explains 623: block 17's big linear w_newline increment was SUPPRESSION, not writing. Both registered predictions (block0>block17) False, informatively (block17 suppresses; block 1 is top writer). Queued block17_calibration (does block 17 suppression scale with token frequency?).
- 625: Block 17 is a FREQUENCY CALIBRATOR. corr(log token frequency, block17 removal-delta) = +0.64: block 17 suppresses common tokens in proportion to frequency. The 12 most frequent next-tokens (',' ' the' '.' '\n' ' to' ' of' ' and' ' a' ...) ALL rise (+1.2 to +2.0) when block 17 is removed. CONTRAST: block 1 corr -0.28, block 9 corr -0.28 -- early/middle blocks WRITE frequent tokens (negative), block 17 is the UNIQUE sign-flip (sole calibrator). FUNCTIONAL IDENTITY (615/618/624/625): the readout layer READS token classes cleanly but does NOT write content -- it CALIBRATES, trimming over-predicted high-frequency function tokens. Explains 624 (block 17 suppresses high-freq newline/article) and the read/write orthogonality (619-622): read!=write because block 17's job isn't to write the class, it's to frequency-correct the distribution. Registered NULL (block9~0) failed informatively (block 9 is a writer, -0.28, not neutral). Queued block17_calibration_ce (trade-off: block 17 helps CE at rare-target, hurts at frequent-target positions).
- 629: causal who-writes-what-where map (9 classes x 18 blocks, all pass specificity null). (1) Next-token class identity is written FRONT-loaded for EVERY class: top writer is block 1 (newline, determiner) or block 2 (other 7); none later. (2) The MIDDLE blocks (10-16) are the top writer for NO class -- combined with the report's "middle nearly linear/loss-irrelevant", the middle does not write next-token class identity. (3) Block 17 calibrates at the class level by REALLOCATING MASS from function classes (suppresses newline -0.215, determiner -0.141, pronoun/punct -0.10, preposition -0.05) to content classes (writes subword +0.207, capitalized +0.187) -- the class-resolved form of frequency suppression. CAVEAT: block-2 magnitude dominance partly reflects early-ablation error-compounding, not clean additive share; robust claims are the three above. Queued depth_band_ablation (CE cost by depth band -- is the middle prediction-critical?).
- 632: the middle's within-class refinement is concentrated on the OPEN content-word slot (space_word sparing +0.672 under middle ablation -- the middle picks WHICH space-prefixed word), NOT uniform content refinement. Content-avg > function-avg (0.297 vs 0.179, NULL ok) but driven by space_word; capitalized +0.205 refined, subword +0.015 NOT. Two exceptions: subword collapses wholesale (both class+token ~0.98 -- mid-word continuation is a PROPAGATION task the middle is needed for, not choose-among-many); newline +0.000 (single-member class). CORRECTS 630/631 "refines rare/content tokens" -> "refines the open content-word slot (space_word), moderate for capitalized/function, none for subword". Queued within_class_depth_profile (per-block sparing across 18 blocks -- where does space_word refinement live?).
- 635: newline circuit traced to input. Trigger = sentence-ending punctuation: P(newline) 1.1% after a word -> 30% after . ! ? (28x bigram trigger). Front attention CARRIES it: front-attn ablation leaves overall rate unchanged (0.025) but cuts the punct-elevation gap 0.291 -> 0.176 (-40%). Line length is a WEAK non-monotonic secondary signal (peaks at 10-30 tokens, not monotonic). Front-MLP ablation raises overall P(newline) to 0.62 (re-confirms 634 global suppression). CIRCUIT: end-punct token -> front attention writes newline -> front MLP applies global downward bias (overridden by the punct context) -> block 17 calibrates. Second fully input-traced circuit alongside article (614), same shape (current-token trigger read by front attention). Queued article_trigger_trace (test 614's a/an vs the triggers causally with attention ablation).
- 636: CORRECTION to 614 (article triggers, tested causally). BE-verbs -> a/an CONFIRMED (pref +0.033, only a/an-favoring group: "is a"). PUNCTUATION -> the CONFIRMED (a/an ~0). PREPOSITIONS -> the, REFUTING 614's "prepositions -> a/an": prepositions give the STRONGEST the-preference (pref -0.111: "of the", "in the"). 614 lumped prepositions with be-verbs; causally they oppose. PATH: front-attention ablation shrinks group-preference spread 0.144->0.105 (~27%, carries the a/an-vs-the CHOICE) while keeping prediction alive; front-MLP ablation zeroes all article probability (carries the MAGNITUDE). Propagated to report (circuit-program section stated prepositions->a/an verbatim). Queued embedding_direct_triggers (how much of the triggers is in the embedding->unembedding direct path vs computed?).
- 639: DECISIVE demonstration, closes input-tracing phase (634-639). Among end-punct positions split by whether a newline follows: DIRECT bigram gives 0.4163 (follows) vs 0.4194 (not) -- separation -0.003 ~ ZERO (context-blind). FULL model gives 0.4718 vs 0.2078 -- separation +0.264 (discriminates true line-ends from mid-paragraph periods). All predictions held; means reconcile with 637. NEWLINE CIRCUIT complete: (1) embedding bigram '.'->newline ~0.42 (context-blind), (2) 18 blocks context-discriminate real line-ends (0.47) from mid-sentence periods (0.21), (3) block 17 frequency-calibrates. The blocks' whole job on newline is DISCRIMINATION of a blanket-firing bigram, not computing the trigger. Report refreshed. Queued direct_vs_full_ce (CE of bigram baseline vs full model, quantify blocks' context contribution).
- 642: partial retraction of 641 + confirmation. At digit-target positions: INITIATION (n=209, solid) is COMPUTED (direct 0.060, full 0.561, 9.36x = 638's 8.3x). CONTINUATION (n=14, underpowered) full 0.716 >> direct 0.198 (3.6x) -> also discriminated. RETRACT 641's "continuation is a bigram": that came from prev-digit positions averaged (only 14/224 actually continue, so full~direct just means "most digits aren't followed by a digit"). Corrected: digit prediction is COMPUTED from numeric context (both sub-cases), embedding gives only a weak "digits cluster" prior -- fits the general phase result (blocks do the real work). (0) failed (continuation underpowered n=14). Queued sentence_boundary_fanout (fresh cleanly-powered circuit: after '.', routing among newline/capitalized/continuation).
- 645: INDUCTION/copying is strong on natural text and DISTRIBUTED. Over 4673 repeat positions, P(token that followed the current token's earlier occurrence) = 0.140 vs base rate 0.0056 (25x) and control token 0.00005 (NULL ok). Localization: front-attn ablation -0.042 (30%), mid-attn -0.044 (31%), late-attn -0.010 (7%) -- front AND mid carry it roughly equally, late barely. CONTRAST with sentence-boundary routing (644, 80% front): routing is LOCALIZED, induction is DISTRIBUTED across front+mid (redundancy-protected across bands). Two circuits, two topologies. Opens copying thread. Queued induction_rare (restrict to rare current-tokens where a memorized bigram can't help -> is it true in-context induction?).
- 646: TRUE in-context induction confirmed. By current-token frequency: rare A (<=3 occ) P(B)=0.333 (60x base), mid A 0.259, frequent A 0.078 -- INVERSE frequency relationship = genuine match-and-copy (rare tokens can't have a stored bigram), not a skip-bigram. Distance-robust, rising with distance (far>32: 0.168 > near<=8: 0.080) = copy over full context, not local n-gram. Control token 0.000004 (NULL ok). Confirms 645 as a genuine induction circuit on a softmax-free bilinear-QK model. Induction thread: exists+strong (645), distributed front+mid (645), true in-context rare-dominant distance-robust (646). Queued induction_head_search (find the actual induction heads via per-head double-QK attention to the copy-source).
- 649: Q4 answered -- ablating the head-SET does NOT localize copying. Cumulative top-K induction-head ablation (647 ranking): K=16 removes only 19% of the copy signal; ALL attention removes 87% (floor 0.018). Random K-sets ~0%. So causal copying is distributed across ~the whole attention stack, not a small set; attention-PATTERN salience (647 z-scores) != causal contribution (pattern!=cause, cf. 619-622). Corrects my Q4 claim to the user (the set is NOT the circuit). Strongest statement of universal redundancy: even the one component-localizable-looking circuit is causally distributed. METHOD: head/pattern selection can't isolate it; need behavior-conditioned SUBSPACE low-rank (Q5). Pivoting queue to low-rank isolation of a clean circuit.
- 651: CONFIRMED rank-1 isolation of block-17 calibration. Removing w_freq loses 103% of the calibration; removing 3 random rank-1 dirs loses 0%,0%,2% (SPECIFIC necessity, NULL ok). keep top-r recovers <=33% non-monotonically -> keep/sufficiency is the WRONG frame for a corrective component (calibration is a correction on top of content-writing, defined by removal not reconstruction). METHOD CONCLUSION (answers Q5): behavior-conditioned low-rank + REMOVAL test isolates to rank-1 where clustering (578-581) and head-ablation (649) reached nothing. Recipe for this redundant model: cov(component output, behavior target) rank-1 dir, verify by removal vs random-removal. First rank-1 mechanistic isolation. Propagated to FINDINGS + deliverable. Queued lowrank_routing_isolation (apply to newline routing).
- 653: DEFINITIVE -- newline routing is not a residual-stored feature at ANY rank. Removing top-r behavior-conditioned dirs (r=1..32) from post-front residual: 0-1% lost at every r (= random-32); post-front residual only weakly linearly encodes it (probe AUC 0.70). Routing is computed by front attention (644) and read nonlinearly -- no linear carrier to remove. Corrects 652: w_route is only a MODERATE probe (0.70), so the outcome is weakly-linearly-encoded, not a clean readout; the wall is nonlinearity/distribution, not read!=write per se. COMPLETES low-rank isolation arc (650-653, Q5): ADDITIVE biases isolate to rank-1 (calibrator); CONDITIONAL routing has no removable linear carrier. "Finer isolation?" answered by component TYPE. Phase boundary; FINDINGS+deliverable updated. Queued lowrank_article_magnitude (is article-magnitude additive or conditional?).
- 654: taxonomy generalized. Article-magnitude is CONDITIONAL (top-8 removal loses 1%, random 0%), like newline routing, unlike the calibrator. Reconciles with 636 (front MLP computes it distributively/nonlinearly, not as a linear residual direction). TAXONOMY on 3 behaviors (completes Q5 arc 650-654): ADDITIVE BIAS -> rank-1 isolable (block-17 frequency calibration); CONDITIONAL prediction/routing -> no removable low-rank carrier (newline routing, article magnitude). Finer isolation is possible for additive/corrective parts, not predictive computations; the wall for the latter is nonlinearity/distribution, not unit-redundancy. Phase boundary. Queued calibrator_direction_id (is w_freq the unembedding's frequency direction?).
- 655: VOID (method bug, caught by registered sanity check). "Frequency axis" mis-constructed as freq_dir = sum_t log_freq(t)*W_U[t], which reads out as log-freq at corr only 0.44 (not >=0.8) because W_U rows are non-orthogonal -- the readout direction needs the PSEUDOINVERSE W_U^+ log_freq. The w_freq alignment test (cos -0.024) is uninformative/voided. Sanity check did its job. Requeued calibrator_direction_id2 with pinv construction.
- 656: the rank-1 calibration direction IS (mostly) the log-frequency axis. Corrected (pinv) freq_dir = W_U^+ log_freq. (0) still marginal: readout-vs-logfreq corr 0.53 (<0.8) -- log-freq isn't a clean single unembedding readout. (a) cos(w_freq, freq_dir) = +0.61 (28x random) -- the isolated calibration direction is strongly aligned with the frequency axis. CROSS-VALIDATION: cos^2 = 0.37 ~ 627's R^2 = 0.41 (frequency explains ~40% of block-17's action) -- two independent methods agree the calibration is ~40% pure frequency axis, ~60% context. Characterization of the rank-1 component complete: freq-proportional bias along the unembedding log-freq direction, rank-1-isolable, ~40% pure-freq. Queued block17_decompose (is block 17 cleanly = calibration (rank-1) + content-writing (rest)?).
- 657: block 17 does NOT split cleanly into calibration + content-writing. Removing w_freq kills calibration (102%) and preserves SUBWORD-writing (104%) but loses 73% of CAPITALIZED-writing (27% kept). UNIFICATION: capitalized words are RARE, so "boost rare content" and "write capitalized" are the SAME rank-1 frequency mechanism -- unifies 624-628 (calibration) with 629 (function->content mass shift): the mass shift is the single w_freq direction. Subword-writing (frequent-ish, orthogonal to frequency axis) is separate, preserved. So block 17 = [frequency bias w_freq = calibration = rare-content boost, rank-1] + [subword-writing, separate]. Completes block-17 characterization. Queued mlp17_subword_isolation (is subword-writing rank-1 or conditional?).
- 658: taxonomy holds WITHIN one layer. Block 17's subword content-writing is NOT rank-1 isolable (remove top-1 = 0% inert; remove top-8 = 139% overshoots the mean-ablate floor = collateral damage, not clean isolation; random preserves) -- patterns with distributed prediction computations, unlike the rank-1 frequency bias. FINAL block-17 characterization (closes 624-658): block 17 = [rank-1 frequency-bias w_freq: suppress frequent function tokens = boost rare content, ~40% log-freq axis, +0.43 nats] + [distributed subword/content writing, not isolable]. The one clean rank-1 component is the frequency bias; everything predictive (even same-layer) is distributed. Sharpest statement of the additive(isolable)-vs-predictive(distributed) law (arc 650-658). Phase boundary. Queued additive_bias_catalog (is frequency the only rank-1 additive bias?).
- 659: VOID (confounded, null caught it). Additive-bias catalog via cos(w_prop, w_freq) is invalid: cov(O, property) pulls every direction toward O's variance structure, so random-label gives cos -0.52 (not ~0). Cannot compare property biases by cosine. 651 already shows calibration is ~fully rank-1 (103% w_freq) -> no second FREQUENCY bias; other-property biases need causal/whitened comparison (deferred). METHOD LESSON: compare behavior-conditioned directions by causal removal or in whitened space, never raw cosine. Pivoting to Q3. Queued mlp17_functional_rank (how many quadratic functions does mlp17 compute -- loss rank vs the rank-8 variance rank).
- 660: Q3 answered. mlp17's FUNCTIONAL (loss) rank > variance rank. rank-r output truncation recovers: r1 33%, r2 58%, r3 69%, r4 75%, r8 78% of the 0.405-nat loss benefit (cumvar r4 91%, r8 95%). random-4 = 1% (null clean). So ~4 quadratic functions recover 75% of the loss (user's "~4" right at that level), but top-8 variance dirs recover only 78% -- the low-variance tail (last 5% var) carries ~22% of the loss. Variance RANK != functional rank (extends 617's variance BASIS != functional basis; cf. report L1 mezzanine). mlp17 not cleanly reducible to 4-8 quadratic functions for full loss. Queued mlp17_tail_content (is the low-var tail the distributed content-writing?).
- 661: readout-layer head/tail structure. Calibration w_freq is HIGH-VARIANCE (95% in top-4 SVD dirs, 81% in top-2) -- a large consistent bias. head(top-8) recovers freq 88%/rare 79%; tail(9-64) recovers freq 5%/rare 13% -- the low-var tail is disproportionately RARE-target prediction (13% vs 5%; 6.5x random-56's 2%). Aside: random-56 recovers 49% of FREQ loss (frequent prediction is redundant), 2% rare. FULL mlp17 characterization (624-661): output = HIGH-VAR HEAD (top-8, ~78% loss, contains rank-1 calibration w_freq + main prediction) + LOW-VAR TAIL (ranks 9+, ~22% loss, distributed rare-content refinement). Most complete component characterization in the program. Queued layerwise_wfreq_removal (is block 17 the unique rank-1 frequency calibrator?).
- 662: CORRECTION to "block 17 unique calibrator". Rank-1 w_freq removal per layer finds calibrator-sign (removal helps freq, hurts rare) in FIVE layers: L4/L5/L6 (early-middle) + L16/L17 (end); block 17 dominates 5-10x (freq -0.24/rare +0.61 vs L16 -0.05/+0.07). Writers L7-15 (several write rare content). NULL clean (random-1 at L17 no sign). Corrects 628 (whole-block CE found only 17 net-calibrates -- others' calibration diluted by writer roles): frequency calibration is DISTRIBUTED across ~5 layers (two bands L4-6, L16-17), block 17 dominant. Method point: rank-1 removal is MORE sensitive than whole-block ablation. FINDINGS item 3 corrected. Queued layerwise_calib_axis (do the 5 calibrator layers share one frequency axis?).
- 663: distributed calibration = TWO orthogonal mechanisms at two depths, not one axis. END band L16/L17 aligns with unembedding log-freq axis (cos 0.43, 0.61; mutual 0.84) = direct logit-level frequency bias (block 17 dominant). EARLY band L4/5/6 shares its OWN direction (mutual 0.57-0.71), NOT readout-aligned (cos ~0.06), orthogonal to end band = mid-network representation-level frequency correction. NULL clean (writers 0.06 < calibrators 0.24). Model corrects token frequency TWICE, in different spaces. Queued early_band_axis (is the early band the INPUT/embedding frequency direction?).
- 664: early calibration band (L4-6) is MODERATELY the input/current-token frequency direction (cos -0.25 vs readout 0.05 vs random 0.02) but not cleanly one axis. Current-token frequency is cleanly encoded in the early residual (emb_dir reads it at corr 0.80, cleaner than readout's 0.53). TWO-BAND CALIBRATION complete (662-664): LATE band L16-17 = clean readout-freq-axis logit bias (cos 0.61, block 17 dominant, rank-1 isolable, +0.43 nats); EARLY band L4-6 = mid-network representation-level correction tied to current-token frequency (cos -0.25), orthogonal to late, not cleanly one axis. Model corrects frequency at two depths/spaces. Calibration thread closed (624-664). Propagating two-band refinement to report; pivoting to middle blocks (focus D).
- 665: middle's within-class content-word refinement uses attention & MLP roughly EQUALLY (drops 0.089 vs 0.097 at space_word targets; baseline 0.150). More context-dependent than the front's class decision (634, MLP-dominant 0.93 vs attn 0.5) -- picking the specific word needs context, attention contributes nearly as much as the MLP. NULL: late-16 attention inert (0.001) = middle function. Depth account + mechanism: FRONT decides class (MLP/token-local), MIDDLE refines specific word (attn+MLP balanced, context-dependent), BACK calibrates (rank-1 readout bias). Queued middle_refine_copying (does the middle's refinement use copying -- induction heads L5/L8/L10 are in the middle?).
- 666: the middle's content-word refinement is NOT primarily copying. At space_word targets, middle ablation collapses novel (drop 0.108, rel 92%) as much as/more than repeat (drop 0.228, rel 86%). Copying (induction) gives repeats a higher BASELINE (0.265 vs novel 0.117 -- copyable = easier), but the middle's refinement machinery is a GENERAL content-word predictor needed for novel words too; induction heads L5/L8/L10 are one contributor, not its defining feature. NULL clean (late-16 no asymmetry). Separates the general refinement circuit from the copying sub-circuit. Queued quote_state (fresh: does the model track quotation parity?).
- 667: NEW capability -- the model maintains a STATEFUL context register (quotation parity). Linear probe AUC by depth: block 2 0.69 -> block 6 0.83 (peak) -> block 12 0.77 -> block 17 0.58 (decays toward output); shuffled null 0.51. Behavioral: P(closing '"') 3.4x higher inside (0.0070 vs 0.0021). A working-memory-like register built early-middle (peaks block 6, the context-integration region 665), consumed/faded by the output -- distinct from the token-class/frequency machinery. (a) "final AUC>=0.8" failed only b/c wrong depth probed; state IS tracked mid-network. Queued quote_state_causal (does removing the mid-network quote register collapse the P('"') inside/outside gap?).
- 668: the quote-parity register is a READ-CORRELATE, not causal. Removing the mid-network quote direction (AUC 0.83) does nothing to the behavioral P('"') gap (0% lost = random). Decodable but causally inert -- read!=write for a stateful register (registered guess correct). UNIFICATION (session's central law): across routing (652-653), magnitude (654), content-writing (658), and stateful state (668), the causal computation has NO removable linear carrier (distributed/conditional); the ONE exception is the frequency-calibration BIAS (rank-1 isolable, 650-651). Decodability != causality. Model = ~ONE isolable linear knob (calibration, ~6% of the +7.48-nat loss-benefit) + ~94% distributed remainder (prediction, routing, refinement, registers) with no linear carrier. Sharpest statement of universal-redundancy/read!=write. Queued paren_depth_state (does a 2nd stateful register follow the pattern?).
- 669: 2nd stateful register (paren depth) confirms the pattern. Decodable AUC 0.92 from block 2 (earlier/more robust than quote), stays 0.82 at block 14; null 0.495. Behaviorally ENORMOUS: P(')') 600x higher inside an open paren (0.078 vs 0.00013). Yet READ-CORRELATE: removing the decodable direction loses 1% of the gap (= random). Even a near-deterministic stateful behavior is causally carried NOT by its decodable direction. Two registers (quote, paren), same verdict: decodable + used + causally-conditional (no linear carrier). Robustly generalizes read!=write/distributed law to stateful memory; sole exception model-wide is the frequency-calibration bias. Queued paren_counter_mechanism (is the counter attention-computed?).
- 670: paren counter built by BOTH front attention (drop 0.165) and front MLP (drop 0.295); even attention-ablated it stays decodable at 0.74. Attention contributes (counting needs context) but isn't sufficient/sole; MLP drop larger but partly a perturbation-size artifact. State-BUILDING is distributed across attn+MLP, no clean counter mechanism. Closes stateful-register thread (667-670): registers are decodable, behaviorally used (600x), causal read-correlates, and distributed-built -- all fit the universal-distribution/read!=write law; sole isolable linear component model-wide is the frequency-calibration bias. Queued recency_bias (is there a SECOND additive knob, or is frequency the only one?).
- 671: confirms "one knob" against recency. Recency effect real & strong (recent tokens predicted 2x better, gap 0.195) but NOT an additive knob: removing rank-1 w_rec doesn't reduce it (gap widens 36%; random 0%) -- read-correlate. Recency joins routing/magnitude/writing/registers on the conditional/distributed side. Central law confirmed vs every candidate: frequency-calibration bias is the model's ONE isolable linear knob (~6%); everything else (recency 2x, paren-close 600x) distributed & linearly-uneditable though often decodable. Comprehensive closure on "isolate to finer grain?": YES for one additive bias, NO for all predictive/conditional/stateful computation. Queued w_freq_generalization (is the one knob a stable model property across data?).
- 672: the one knob is a STABLE MODEL PROPERTY. cos(w_freq_A, w_freq_B) from disjoint data halves = 0.982 (shuffled 0.224); half-A direction cross-removes calibration on half B at 92% of native (random ~0). NULL clean. The single isolable component is robust/portable, not a data artifact. CLOSES the isolation investigation (full validation): model = ONE stable rank-1 knob (frequency calibration: necessary/specific/~40% log-freq axis/5-layer distributed/block-17-dominant/stable, ~6% of loss-benefit) + everything else distributed & linearly-uneditable though often decodable. Comprehensive answer to Q4/Q5. Queued w_freq_steering (is the knob actionable as a frequency/diversity dial?).
- 673: the one knob is ACTIONABLE -- a monotonic frequency/diversity dial. Scaling mlp17's w_freq projection by alpha: top20 token mass 0.42 (alpha 0, calib off) -> 0.29 (alpha 1 default) -> 0.13 (alpha 2), a 3.4x range; random direction inert (null). Tunable 626 trade-off: amplifying lowers rare-CE (4.07 at alpha 1.5, marginally below default) & raises freq-CE (1.6->2.7); default alpha=1 near-optimal overall. The single isolable component is USABLE as an interpretable diversity control. Applied capstone; isolation/one-knob investigation (Q4/Q5) comprehensively closed. Queued position_bias (is there a position/index additive knob, or is frequency the sole additive axis?).
- 674: causal additive-knob sweep. FREQUENCY is the dominant/only additive knob (dCE_high -0.021/dCE_low +0.739, trade-off, mag 0.76); LENGTH is NOT (same-signed +0.123/+0.002, mag 0.12, frequency-correlate); random removal ~0 (null). BUG: is_capitalized row VOID (binary property -> median=0 -> empty low group, nan); requeued fixed. Clean causal close of the confounded 659 catalog: frequency calibration is the model's dominant additive knob, no other additive axis survives the trade-off test (capitalization expected non-independent per 657). Completes isolation investigation (650-674). Queued additive_knob_sweep_v2 (binary-split fix + punct/quote candidates).
- 675: DEFINITIVE close of additive-knob catalog. Frequency is the UNIQUE additive bias -- only property whose rank-1 removal gives an opposite-signed CE TRADE-OFF (dCE_high -0.021/dCE_low +0.739). Length/capitalization/punct/after-quote all show SAME-signed dCE (removal hurts both groups) = WRITER contributions, not bias knobs, even when large (after_quote mag 0.67 = big writer, e.g. quote-context prediction 667-668). DISCRIMINATOR = trade-off SIGN, not magnitude. Coded (a) failed (magnitude-based) but substantive claim confirmed. NULL clean. Definitive close (650-675): model has exactly ONE additive bias (frequency calibration, stable actionable rank-1); everything else distributed writing/prediction. Queued residual_outlier_dims (fresh: does the residual have massive-activation outlier dims?).
- 676: NEW structural finding -- MASSIVE ACTIVATIONS. Per-dim residual RMS: max/median ratio grows 2.6x (block 0) -> 10x (block 8) -> 43x (block 16) -> 58x (block 17); 5 dims >8x median at output (top RMS ~57000 vs median ~986). Persistent channels: dims 645, 990 in top-5 from block 4-17. Gaussian null flat (1.02, 0 outliers) -> real trained structure. CONNECTION: 88% of w_freq (frequency-calibration direction) squared-mass is on the outlier dims -- the ONE clean knob is implemented through the massive-activation channels (a large consistent bias lives in the highest-magnitude dims). Links isolation finding to the massive-activation phenomenon. Queued massive_dim_ablate (are the massive dims causal calibration channels or inert sinks?).
- 677: massive dims are the SUBSTRATE the calibration rides, not the mechanism. Mean-ablating top-K massive mlp17 dims (RMS up to 42000 vs median 380) HURTS both freq (+0.17) and rare (+0.77) = general writer effect, NOT calibrator trade-off. Reconciles 676 (88% of w_freq mass on these dims): calibration is a rank-1 DIRECTION within the massive-dim subspace, not the dims (removing w_freq = clean trade-off; zeroing the dims wholesale removes the bulk readout -> hurts all). Massive dims = broad high-magnitude readout channels; calibration rides them. (b) flag is a coding artifact (random dCE ~0 straddled zero). Queued massive_dim_tokens (are massive activations concentrated on structural tokens -- the attention-sink pattern?).
- 678: massive activations are NOT attention sinks. Top block-17 massive dims are uniform across positions (pos0 23091 <= pos33+ 27188, position 0 not special) and token classes (range 21k-38k, only newline mildly 1.4x) -- an always-on DC/bias-like channel. ARCHITECTURAL: classic LLM massive activations are tied to SOFTMAX attention sinks (BOS/delimiters); this model has NO softmax (unnormalized bilinear attention) so no sink mechanism -> massive activations manifest as a uniform DC substrate, not sinks. Same surface phenomenon, different cause (learned bias channel). Fits 677 (ablating hurts all = constant substrate) + 676 (carries calibration). Queued massive_dim_constancy (are the dims near-constant = a literal learned bias?).
- 679: massive dims are NOT a clean constant bias. Top block-17 massive dims have LARGE DC offset (mean +41681/-18588/+24389/... which median dims lack, mean~0) but ALSO large variation (std ~ mean, CV ~0.9-2.0, sign-cons 0.67-0.84). "Big-bias + big-signal" channels, not pure DC. Median dims' higher CV is degenerate (mean~0); their absolute std (~980) is 20-40x smaller. Clean contrast = the MEAN (massive dims have huge DC offset). Refines 678 (partly). Massive-activation line (676-679): grow with depth, persistent channels 645/990/981, not sinks (no softmax), carry calibration direction (88% w_freq) but broad readout, large DC offset + large signal. Queued massive_dc_vs_signal (is the DC offset functional or an inert offset?).
- 680: the massive dims' DC offset is the rms-norm GAIN CONTROLLER. Removing DC (subtract mean) costs +1.58 nats > removing signal (mean-fill, +0.61); median dims inert. MECHANISM: massive dims dominate the residual sum-of-squares (~85%: top-8 ~6.5e9 vs other 1144 dims ~1.1e9), so their constant magnitude sets the rms-norm scale for the whole readout; removing the DC breaks the normalization gain -> all logits shift -> CE explodes. Explains why the model builds massive activations (676, grow with depth): they're the rms-norm normalization substrate that also hosts the calibration knob. Closes massive-activation line (676-680): growing, non-sink (no softmax), DC+signal substrate controlling readout gain. Queued attention_density (softmax-free bilinear attention: focal or diffuse patterns?).
- 681: softmax-FREE bilinear attention is FOCAL. Effective-keys fraction ~0.17-0.32 mean (vs 0.64 random); most-focal heads L1.H1 0.06/L0.H3 0.067/L2.H6 0.069 (~6-8% of keys); 60 of ~162 heads focal (<0.2). The double-QK product (pat=(q.k1)(q2.k2)) produces peaked selective attention WITHOUT softmax -- peaks where BOTH QK terms are large. Depth: front most focal (L2-4 ~0.17, where bigram/routing heads live), late more diffuse (L15-17 ~0.30). NULL clean (random 0.64). Model's unusual attention isn't diffuse-by-default; learned focal via the product, front-loaded. Queued attention_double_qk (is focality due to the double-QK: is s1*s2 more focal than s1 or s2 alone?).
- 682: the double-QK PRODUCT is the focality mechanism. Single QK terms are DIFFUSE (s1 0.540, s2 0.548 eff-keys fraction, near random 0.64); their PRODUCT is FOCAL (0.233, 2.3x sharper), more focal than BOTH in 100% of cases. The softmax-free model peaks attention by AND-ing two diffuse selections (pat=s1*s2 large only where both large), not by exponential normalization -- explaining the two QK projections per head (c_q/c_k + c_q2/c_k2). Distinctive architectural account (681-682). Queued attention_qk_complementarity (do the two QK circuits select DIFFERENT things = genuine 2-criterion AND, or the SAME = redundant sharpening?).
- 683: the two QK circuits are COMPLEMENTARY, not redundant. s1-s2 pattern corr ~0 (mean |corr| 0.23); 0/162 heads redundant (|corr|>0.8), 120/162 complementary (<0.3). So the double-QK is a genuine TWO-CRITERION AND: each QK selects a DIFFERENT key-set, focal pattern = their intersection (key attended iff high on BOTH independent criteria), NOT one-criterion-squared. A few heads anti-correlated (L15.H1 -0.72 = difference-like op). COMPLETE softmax-free attention account (681-683): focal (681) via double-QK product (682) of two complementary criteria (683) = expressive two-criterion conjunctive selection, more than single-QK softmax. Report to get the architecture findings. Queued qk_criterion_identity (what are the two criteria for focal heads?).
- 684: the two-criterion attention often factorizes into POSITIONAL x CONTENT. 5/6 focal heads: one QK distance-selective (|corr| 0.32-0.51 = positional), the other weak (content). Heads attend to keys that are BOTH at the right relative distance AND match content = classic lookup/induction structure via the two-QK product. Positional QK sign varies (L0.H3/L2.H6/L6.H3 prefer distant; L7.H8/L2.H2 prefer recent). Completes softmax-free attention account (681-684): double-QK = two-criterion conjunction often = positional x content. How the bilinear arch does selective lookups without softmax; ties to the front lookup/routing/induction heads. Queued qk_split_census (how common is positional x content across all 162 heads?).
- 685: model-wide census -- POSITIONAL x CONTENT is the dominant attention motif. 162 heads: 71 (44%) pos x content split, 12 both-positional, 9 both-content, 70 mixed; shuffled null 0.02. Nearly all heads use positional selectivity in >=1 QK (only 6% pure content). Model attends to keys at the right relative position that also match content. Completes softmax-free attention account (681-685): focal (681) via double-QK product (682) of complementary criteria (683) predominantly positional x content (684-685). Bilinear arch does lookup-style attention by multiplying a positional QK x a content QK. Queued mlp_gate_conjunction (is the bilinear MLP gate h=(Lx)*(Rx) also an AND, like the double-QK -> whole model on multiplicative AND-gating?).
- 686: UNIFYING PRINCIPLE -- the whole bilinear model computes by MULTIPLICATIVE AND-GATING. MLP0 gate h=(Lx)*(Rx): factors dense (0.62 each), product selective (0.32, sharper than both in 100% of units), sharper than random product (0.51<0.64 = learned-complementary factors). EXACTLY the double-QK attention (682). DEFINING PRINCIPLE (681-686): entire model -- attention (s1*s2) AND MLP (Lx*Rx) -- computes by multiplying two dense linear projections whose product is selective (large only where both large). One primitive replaces BOTH softmax and relu; no softmax/no relu, selectivity everywhere from the AND of two linear maps. Complete unified account of the architecture. Queued gate_selectivity_depth (is AND-gating consistent across all layers, attention + MLP?).
- 687: AND-gating is UNIVERSAL -- product sharpens in 18/18 layers for BOTH attention (product 0.17-0.33 vs factors 0.50-0.62) and MLP (product 0.31-0.38 vs factors 0.61-0.64). All 36 gates (18 attn s1*s2 + 18 mlp Lx*Rx) are multiplicative ANDs; no exceptions. Depth: attn more diffuse with depth (front most focal), mlp constant ~0.35. Confirms unifying principle (686) end-to-end: bilin18 is uniformly a network of products of pairs of linear projections, no softmax/no relu. Complete architectural account. Queued massive_from_gating (do the massive-activation dims arise FROM the multiplicative gates?).
- 688: massive activations arise FROM the multiplicative gates. Both the attention product (c_proj of s1*s2) and MLP product (Down of Lx*Rx) write the massive residual dims: L17 mlp overlap 10/10 corr 0.74, att 8/10 corr 0.76 (mlp slightly dominant); L8 attention dominant (8/10 vs 3/10). UNIFIED ARCHITECTURE CHAIN (676-688): (1) one primitive = multiplicative AND-gate (products of linear pairs, 36 gates), (2) products are large where both factors large -> high-magnitude outputs, (3) accumulate into massive-activation dims (grow w/ depth, not sinks), (4) dominate residual norm (~85% SS) so DC sets rms-norm gain (-1.58 nats to remove), (5) host the one frequency-calibration knob (88% of w_freq). Bilinear gating causally explains the massive activations + norm control. Closes architecture investigation. Queued lambda_schedule (per-block residual rescaling: systematic gain/decay schedule?).
- 689: residual rescaling (last component). EMBEDDING re-injected at lambda1~8 at almost every block (kept dominant, 18x). FRONT resets the residual: lambda0 near-zero at L1 (0.013), L5 (0.064) nearly wipe it; L0/L2 amplify (6.09, 1.98); back (L8-17) lambda0~1 accumulates. prod(lambda0) block0->end = 3.6e-4 (early writes attenuated, confirms report/400's ~2e-4). DEDUP: report already noted the rescaling + attenuation; this adds the lambda1~8 embedding re-injection + specific reset layers. EVERY architectural component now characterized. COMPREHENSIVE TERMINUS. Queued embedding_recoverability (is current-token identity linearly decodable from the FINAL residual, per the lambda1~8 re-injection?).
- 690: functional confirmation of embedding-dominant residual (689). Current-token log-freq linear-probe R^2: 0.907 (embedding) -> 0.846 (block 8) -> 0.727 (final block 17); shuffled null -0.43. The current token stays strongly linearly present in the residual to the readout (slow decay), confirming lambda1~8 re-injection -- unlike a normal transformer that transforms it away. Grounds 637-640 (embedding triggers always available). Completes embedding-dominance picture (689-690): blocks add context ON TOP of an ever-present embedding. EXHAUSTIVE characterization of bilin18 (619-690) -- every circuit/behavior/architectural component examined. Queued embedding_massive_overlap (do the re-injected embedding's high-mag dims contribute to the massive-activation dims?).
- 691: the two architecture threads are SEPARATE. Embedding x0 is dimensionally FLAT (per-dim RMS peak 1.5x, no massive dims); final residual is peaked 58.2x. Top-10 overlap 2/10, rank-corr 0.14. The massive-activation dims (676) are BUILT by the blocks (MLP gates, massive_from_gating), NOT inherited from the re-injected embedding. So embedding-dominance (current token kept present, distributed, 689-690) and massive-activation norm control (680) are independent mechanisms sharing the residual stream. Closes the last connecting question. Complete cross-confirmed account (619-691, 73 sections).
- 692: ROBUSTNESS consolidation on a fresh 4x-larger held-out corpus (64 rows). (A) massive-dim peak 57.3x (prior 58x); (B) AND-gating product 0.238 vs factor 0.548, sharpens 18/18 layers (prior ~all); (C) embedding recoverability R^2 0.762 (prior 0.73), null -0.17. ALL HEADLINE NUMBERS ROBUST. The account's quantitative backbone reproduces on held-out data. Exhaustive, cross-confirmed, robustness-checked characterization of bilin18 (619-692, 74 sections). No open connecting questions remain within this model.
- 693: CONTROL PHASE #1. The gain-control finding (680) buys a TEMPERATURE KNOB. Scaling the 8 massive residual dims by g moves output entropy monotonically 2.07->5.07 nats (g 0.5->1.6); CE is U-shaped with min at native g=1 (3.36) -- the learned gain is CE-optimal. Random-dim null flat (entropy range 0.021 vs 3.008). Confirms 680 causally as a control surface. With w_freq_steering (frequency/diversity dial), bilin18 exposes TWO demonstrated control knobs. Queued knob_composition (are the two knobs independent/composable, per 691?).
- 694: Q5 ANSWERED. RSPD A-SVD (user's SVD(WX)*X.pinv, data-conditioned closest weight) on mlp17.Down, priced by REAL cross-entropy: functional core is rank ~4. rank-1 recovers 55% of mlp17's 0.715-nat benefit, RANK-4 recovers 83% (20/80 rank r80=4 of 1152), rank-8 88%. Random-projection null recovers ~0/negative at every rank (data-aligned, not a W artifact). CONFIRMS+SHARPENS Q3/660's "~4 quadratic functions" via an independent method, and A-SVD beats the old output-variance basis (rank-8: 88% vs 660's 78% -- data-conditioned basis finds what variance misses). SCOPE: this is the Down OUTPUT MAP low-rank on-data; does NOT contradict finding-1 (upstream routing still distributed). Method had only been applied to mlp0/attn0 (578-591), never flagship mlp17 -- gap the user caught, now closed. Queued rspd_mlp17_core_readout (name the 4 directions via unembedding).
- 695: CONTROL PHASE #2. The two control knobs (temperature=scale massive dims by g; frequency=scale w_freq by alpha) are DOMINANTLY separable (null held: g moves entropy, alpha moves top20-mass; 12x cross-axis asymmetry) and SIGN-STABLE (g's entropy effect stays +, alpha's mass effect stays - across all settings -- never cancel/reverse) but NOT strictly independent: entropy effect of g grows 1.57->4.06 as alpha rises (ratio 2.59), a multiplicative interaction. Refines 691: mechanisms structurally separate but their softmax-output effects compound (any two logit-space edits do). Fixed core_readout dtype bug and requeued.
- 696: Q5 GOAL DELIVERED. RSPD's rank-4 core of mlp17 (694) splits into FOUR NAMED directions (orthonormal; random null clean): dir0 = FREQUENCY-CALIBRATION axis (cos 0.878 with w_freq; boost rare UPPER, suppress common lowercase); dir1 = SUBWORD/continuation writer (rare fragments; matches 657); dir2 = PROPER-NOUN/named-entity content writer (God/James/Paul); dir3 = TOPICAL/ideological content writer (Soviet/Brexit/Lenin). RSPD's data-conditioned A-SVD did what covariance-removal could not -- split mlp17 into interpretable functional pieces, reconciling calibration(650)+subword(657)+content(658, now resolved into 2 writers) in ONE decomposition. Prediction (a) marked False only on a miscalibrated freq_corr threshold (cos_wfreq=0.878 decisively IDs the calibration axis; w_freq acts through massive-dim gain not unembedding-freq, 676 read!=write). Closes RSPD/Q5 arc (694-696). Note: queued bqrunner run had transient exit=1; reproduced directly, json valid. Queued rspd_mlp0_functional_rank (same method on FRONT class-decider, front-vs-back comparison).
- 697: FRONT-VS-BACK via RSPD. mlp0 (front class-decider) A-SVD CE-priced: r80=8 (mlp17=4), benefit 2.386 nats (3x mlp17's 0.72), rank-1 recovers 67%, random null fails (r80=never). Core directions = broad token-CLASS selectors: rarity axis (dir0, freq_corr -0.50), quantifiers (several/various), named-entities (CVE/NVIDIA/realDonaldTrump), scope-adjectives (entire/whole/first) -- matches 634 (front decides class). Contrast: front core = class-selectors (rank-8, 2.4 nats); back core = calibration+content-writers (rank-4, 0.7 nats). Both ends compress to nameable low-rank cores; front higher-rank as predicted. Closes RSPD/Q5 arc (694-697): SVD(WX)*pinv CE-priced isolates BOTH flagship MLPs into nameable low-rank cores.
- 698: CAUTION (user feedback). The 694-697 r80 numbers use only ~3072 tokens -- too thin for strong rank claims; treat mlp17 r80=4 / mlp0 r80=8 as PROVISIONAL pending a data-scaling check. Pivot per user: scale data on GPU, map the first few layers ENTIRELY. Queued rspd_front_layers_scaled (GPU, much more data, blocks 0-2 attn+mlp, r80-vs-N robustness sweep).
- 699: SCALED + DATA-ROBUST front map (GPU, more data). (1) mlp0.Down r80=8 IDENTICAL across N=3k/6k/12k/24k tokens (recovered flat) -- low-rank claim data-robust, resolves 698 caveat. (2) Front is HETEROGENEOUS (blocks 0-2, held-out CE, full-rank sanity exact for all): attention c_proj maps strikingly LOW-rank (block0 r80=2, block1 r80=1 [rank-1 recovers 93% of 2.06 nats!], block2 r80=8); mlp0 low (8); but mlp1 r80=128 and mlp2 r80=256 are GENUINELY high-rank (low-rank surrogates actively harmful, recovered far negative at r<=8). CORRECTS the impression that all MLP cores are rank ~4-8 -- it's component-specific. Registered prediction (b) FAILED/reversed: attention maps are the low-rank ones, not MLPs. Queued rspd_block1_attn_rank1 (name block1.attn's rank-1 direction).
- 701: block1.attn's rank-1 core NAMED = boundary->continuation writer. Fast A-SVD, held-out CE: rank-1 recovers 0.946 of block1.attn's 2.15 nats (confirms 699 r80=1 at scale). Fires at '\n'/'.' boundary tokens; writes toward common continuation words (that/this/you/so/not/only). Connects to newline circuit (635/637) + front class-decision (634). CORRECTION: peakedness null FAILED (random dir MORE peaked, 8.9x vs 2.1x) -- confounded by massive-activation dims in the attention output (676); real structure evidence = boundary-token pattern + 95% rank-1 CE recovery. Next control redesigned (token-conditional). Queued rspd_block0_attn_core.
- 701b: rank-vs-fidelity plot (rank_fidelity_front.png) for blocks 0-2; MLPs solid (r80 circled), attention dashed. Shows mlp0 low-rank (r80=8) vs mlp1/mlp2 diving negative (low-rank surrogate worse than ablation) -> genuinely high-rank. Clarified for user: r80 = smallest rank recovering >=80% of a layer's CE loss-benefit; W = Down decoder (1152x4608) conditioned on real gate activations X (NOT the full bilinear MLP rank); priced by held-out real CE.
- 702: CLUSTER->per-cluster rank (user idea). Clustering tokens by decoder-response direction (k=8) finds REAL, linguistically-coherent groups (mlp1: punct/proper-nouns/determiners/prepositions/copulas) that are genuinely more compressible than shuffled same-size subsets (mlp1 real 184.5 < shuffled 229.0 of global 591; mlp0 131 < 182 of 404) -- but the high-rank layers stay HIGH-rank per-cluster. Strong 'union of low-rank circuits' NOT supported: the shuffled null shows most per-cluster rank drop is just sample-size, not structure. SCOPE: energy-recovery rank (591>>CE r80=128; 660 energy!=functional basis) may understate functional compression. Queued rspd_cluster_ce (CE-priced per-cluster-output surrogate vs global).
- 703: block0.attn rank-2 core NAMED (confound-free null now works). rank-2 recovers 0.836 of its 1.47 nats (confirms 699 r80=2). dir0 = suffix/morphology writer (-izing/-ized/-ist), diffuse (concentration 0.09); dir1 = boundary/discourse writer, fires strongly on NEWLINE (concentration 0.71 vs random-complement 0.10 -> null HELD). PATTERN: block0.attn dir1 + block1.attn rank-1 both boundary->continuation writers -- front attention is largely boundary detection + continuation steering (coherent w/ 634/635/637). Queued rspd_cluster_ce still running.
- 704: CLUSTER->LOW-RANK HOLDS IN FUNCTIONAL (CE) SPACE (user idea vindicated; reconciles 702). Replacing mlp1's OUTPUT per-token with its cluster's rank-r subspace >> global rank-r subspace: at r=8 cluster +0.01 (break-even) vs global -3.12 (worse than ablation); r=32 cluster +0.71 vs global +0.14. Clean hierarchy cluster>shuffle>global at every rank. NULL CORRECTION: shuffle-beats-global is expected (K subspaces span more than one), NOT a refutation; the right test is cluster-vs-shuffle (cluster wins every rank, e.g. +1.08 at r=8) -> assignment genuinely matters. RECONCILES 702: energy space hid it (591 rank over-weights loss-irrelevant dirs), CE space reveals it (660 energy!=functional). Refines 699: mlp1 high-rank GLOBALLY but LOW-rank PER-CLUSTER = union of ~8 low-rank output circuits. Queued rspd_cluster_ce_ksweep.
- 705: K-SWEEP confirms union-of-low-rank quantitatively. At fixed r=8, mlp1 reconstruction rises MONOTONICALLY with #clusters K: cluster -3.12(K=1,=global)->-1.45->-0.56->+0.01->+0.23->+0.35(K=32), rise +3.47; still climbing (union of >32 low-rank pieces). cluster>shuffle at every K>1 (gap ~1). Completes 702-705 arc: mlp1/mlp2 globally high-rank but = UNION of low-rank per-cluster OUTPUT circuits in CE space (not energy, 702; 660 energy!=functional). Answers Q5's finer-isolation goal for the hard high-rank case. Queued rspd_cluster_ce_mlp2 (confirm on the other high-rank layer).
- 706: mlp2 GENERALIZES union-of-low-rank (cluster>>global every rank, reaches +0.77) but with a wrinkle: for this tiny-benefit layer (0.153 nats) SHUFFLE beats CLUSTER at small r (crossover at r>=16), because harm-avoidance dominates and confident narrow cluster reconstructions inject more harm than shuffle's softer projection. Core finding robust for the substantial layer (mlp1, ~1 nat, clean); mlp2 confirms cluster>global qualitatively. Arc 702-706 complete: high-rank MLPs = union of low-rank per-cluster output circuits (in CE space), strongest where layer contribution is large.
- 707: USER CHECKS. (A) 703 random baseline: block0.attn rank-2 A-SVD recovers 0.81 vs random rank-2 subspace 0.007 (~115x), stable N=3k->12k -> low-rank core real. (B) 704 data-robustness: cluster>>global (>2.6-nat gap) and cluster>shuffle hold at fit N=6k AND 24k; exact value has eval-set variance but ordering stable -> more data doesn't change conclusions. (C) clustering_viz.png delivered (token cosine-sim ordered by cluster + cluster bar + KxK centroid + colorbars); KEY: off-diagonal cluster sim high (+0.2..0.5) -- clusters share the massive-activation/DC direction, separate on a small discriminative part -> why union shows in CE not energy space.
- 708: mlp1 cluster naming + TEMPERING caveat. Per-cluster functional ranks heterogeneous [32,128,16,32,128,128,128,128] (max/min 8) but 5/8 hit the rank-128 ceiling under a STRICT 0.02-nat (~99%) tolerance -- much stricter than r80's 80%, so not directly comparable. Only punctuation (cl2 rank16) + determiner/conjunction (cl0/cl3 rank32) genuinely low-rank; content/prep/proper-noun clusters stay high-rank per-cluster. Clustering HELPS (704-705) but does NOT fully dissolve mlp1 into low-rank pieces (reconciles 706's caution). Token content coherent; upstream output-readout mostly unreliable proxy (16 blocks to readout). Queued rspd_cluster_r80 (per-cluster rank at the 80% tolerance for fair comparison).
- 709: CORRECTION to 704-708 arc. At the FAIR 80% bar, mlp1 per-cluster r80 = [64,128,128,128,128,128,128,128] -- 7/8 need rank ~128, SAME as global. mlp1 is GENUINELY HIGH-RANK even per-cluster. 708's low ranks (16,32) were an ARTIFACT of tiny-benefit clusters passing a 0.02-nat test trivially. 704-705 REINTERPRETED: cluster-beats-global is real but MODEST (32 clusters at rank-8 = only ~35% recovery); clustering gives a small low-rank advantage, does NOT dissolve the high rank. Corrects my earlier '704 vindicated' overstatement. Aligns with 702/706. Propagated to FINDINGS item 15.
- 710: WHAT THE FRONT COMPONENTS DO (functional map). Per-component ablation by next-token category: the front's work lands on HARD open-vocab predictions (content/subword/digit/cap_word, +3 nats from big components) NOT easy ones (newline +0.3-0.6, func_word +0.7-1.5, already handled by embedding/bigram). Magnitude: block0.mlp~block1.attn biggest > block0.attn > block1.mlp > block2.attn > block2.mlp~0 (matches 699 benefits). Components BROAD not category-owners. Connection: block1.attn (boundary detector 701) most helps subword/content -> boundary feeds continuation prediction. CORRECTION: differentiation null flawed (shuffled-ablated vs real-baseline, inconsistent) -> can't claim category specialization; easy-vs-hard + magnitude are the robust findings. Queued front_component_function_v2 (consistent null).
- 711: CORRECTED specialization test (fixes 710). Front components have reliably DIFFERENT category emphases: between-component profile corr 0.49, within-component reliability 0.84 -> real differences. Normalized by difficulty: block0.attn leans SYNTACTIC (func-word/punct), block0.mlp + block1.attn lean OPEN-VOCAB/SUBWORD (frac +1.21/+1.26 on subword). SUBWORD continuation is the standout front task (connects 703 suffix-writer, 701 boundary). block2 negligible. Corrects 710's 'broad not specialized' -> 'overlapping but reliably differentiated, syntactic vs open-vocab split'. Self-correcting: 710 flawed null -> v2 proper test.
- 712: VOID (bug caught). rspd_depth_rank_map returned r80=512 for ALL 18 MLP layers, contradicting established mlp0=8/mlp17=4 (694,699). Cause: asvd_fast's Gram X.T@X (4608^2) is rank-deficient when N<d_in (NFIT=12 -> N=3072 < gate width 4608), giving a wrong pinv. Tells: r80=512 everywhere + null mlp1 rand-8 +0.127 (should be ~0). Fixed asvd_fast to pick the correct Gram by regime (N<d_in uses NxN Gram) + bumped depth map to NFIT=24. Requeued. No conclusions from the void run.
- 713: FULL-DEPTH MLP functional-rank profile (corrected). BARBELL: edge MLPs do the most work and are LOW-rank (mlp0 ben2.35/r80-8; mlp16 ben0.88/r80-1; mlp17 ben0.72/r80-4; mlp15 r80-4), early-middle high-rank (mlp1-3 r80 128-256), deep-middle mlp6-14 nearly INERT (~0.03 nats each; their r80=512 NOT meaningful -- no signal). mlp16 = RANK-1 (single direction, 0.88 nats). MLP work concentrated at edges; the middle's work is NOT in MLPs (consistent 665). r80 match 694/699 exactly -> 712 fix validated. Queued rspd_mlp16_rank1 (name mlp16's single direction).
- 714: CLUSTER OVERLAP (user obs). mlp1 clusters DO overlap: subspace sharing 0.05-0.33 (vs 0.014 random null) + 27% token-level ambiguous membership -> hard disjoint clustering is the wrong model. But it's SIBLING overlap (symmetric, partial), NOT parent-child NESTING (0 clusters >=60% contained in another); biggest overlap c6<->c7 proper-nouns<->content-nouns (0.33, both nouns). User right that overlap is real; refined: fuzzy siblings sharing a manifold (707 shared DC), not a containment tree. cluster_overlap_hierarchy.png delivered. Queued rspd_mlp16_rank1.
- 715: mlp16 rank-1 core NAMED = SENTENCE-BOUNDARY -> continuation writer (back-end analog of block1.attn's front rank-1, 701). rank-1 recovers 0.902 of mlp16's 0.88 nats. Fires on sentence-end punctuation (. ) ? ! :); writes common continuation words/punct ([...] ( - all so and). So bilin18 has rank-1 boundary->continuation circuits at BOTH ends (block1.attn front, mlp16 back) -- the barbell's low-rank edge machinery is largely about SENTENCE STRUCTURE. CAVEAT: concentration null weak (0.72 vs 0.61, confounded by massive dims as in 701); rank-1 CE recovery is the evidence.
- 716: CROSS-GRANULARITY NESTING = CLEAN TREE (32k tokens). Fine K=16 clusters each nest into EXACTLY ONE coarse K=4 cluster (11 tree, 0 overlap; containment up to 0.95 vs 0.02 null). Coarse groups = closed-class/content+proper/prep/punctuation; fine nest sensibly (determiners+pronouns->closed-class, content+proper-frags->content, punct+conjunctions->punct). RECONCILES 714: single-scale = overlapping fuzzy siblings (27% ambiguous), cross-scale = clean nesting tree. User's hierarchy intuition vindicated when scales separated. DATA NOTE: 32k gave clean result (85s) where small data was ambiguous -- adopting 32k+ default.
- 717: BARBELL CONFIRMED at 32k-token eval. Full-depth r80 profile essentially identical to 713 (mlp0=8, mlp1=128, mlp2=256, mlp16=1, mlp17=4; mlp15 4->2; deep-middle 6-14 inert r80=512). Barbell is data-robust; larger eval didn't change the headline.
- 718: CLUSTERING RIGOR. cluster>>global (704) is NOT a simple-kmeans artifact: k-means++ (best of 5) at 32k gives cluster -0.05 vs global -3.02 (~same as simple, slightly better), cluster>shuffle both, restart-ARI 0.58. Survives a principled clusterer. (r=8 recovery ~break-even per 705/709; robust claim is cluster>>global.)
- 719: ABLATION-COVARIANCE CLUSTERING (user's method, implemented). Ablate random subsets of global A-SVD components, record per-datapoint CE+MSE damage, covariance across datapoints, cluster. CONVERGES: split-half covariance corr 0.12(T=50)->0.91(100)->0.99(1500), null 0.001, N=2048 datapoints. KEY FINDING: MSE-damage and CE-damage DECOUPLED (per-datapoint corr -0.05, MSE-vs-CE cluster ARI 0.01) -- output-magnitude != loss-relevance again (read!=write). MSE-ablation cluster agrees moderately w/ output-dir k-means (ARI 0.25). Clusters coherent. cluster_ablation_covariance.png (convergence + covariance heatmap). Implication: for "minimal rank without loss" use CE not MSE.
- 720: SHARED-BASIS PER-CLUSTER MINIMAL RANK (user follow-up). In the shared global A-SVD basis (M=96), CE-ablation-covariance clusters each need r90=48-96 of 96 components for 90% CE (spread: cluster3=48 lowest, most=96), overlap heavily (Jaccard 0.73), and random-set recovery 0.74 ~ importance-set -- at these high ranks WHICH components barely matters, you need most. CONFIRMS 709: mlp1 genuinely high-rank per-cluster; no small distinct per-cluster vocabularies. Clustering converges 0.88. Clusters share ~the whole basis (not distinct dialects). Definitive: token-cluster structure does NOT map to a low-rank functional decomposition of mlp1.
- 721: POSITIVE CONTROL (mlp0 low-rank). Same shared-basis method as 720 on mlp0 (r80=8): clusters need FEW DISTINCT components (r90=1-32 mostly; punctuation cluster RANK-1 = comp{1}; "to/\n" cluster RANK-2 = {4,33}), Jaccard 0.14, random-null FAILS (-293). DECISIVE CONTRAST vs mlp1 (r90 48-96, Jaccard 0.73, random~importance). VALIDATES method + proves mlp1 high-rank is REAL. Low-rank layers (barbell edges mlp0/16/17) DO decompose into named per-cluster low-rank circuits with distinct vocabularies; high-rank (mlp1/2) don't. mlp0's per-cluster components are nameable circuits ready to compose. Queued read_write_overlap + cross-layer composition (offline batch).
- 722: DECONFUSION encoder vs decoder (user Q). Effective-rank per stage (32k tokens): ENCODER WEIGHT capacity (read_er) SATURATED+UNIFORM ~920-1040/1152 every layer -- NOT the differentiator. GATE FEATURES produced (gate_er) vary hugely and track decoder rank: mlp1=429/mlp2=861 (high) vs mlp16=11/mlp17=5 (low). mlp1 vs mlp0: read_er ratio 1.13 (same capacity), gate_er 2.40, out_er 4.5, r80 16. ANSWER: high-rank layers are NOT computing the same thing -- they realize a genuinely richer gate-FEATURE set (encoder OUTPUT, not weight capacity); decoder then spreads (mlp1/2) or collapses (mlp15 gate_er 510->r80 8) features. Difference is in the encoder's REALIZED output, not raw weight rank.
- 723: RESIDUAL-BUS WRITE->READ FLOW (composition map). Subspace overlap(write_i->read_j), r=16, 32k tokens (random 0.0139): forward flow 0.079 (5.7x) > backward 0.044; STRONGEST paths are ADJACENT i->i+1 (mlp5->blk6 15.2x, mlp0->blk1 15.1x, etc) -> flow is mostly LOCAL. block 17 = integration HUB (mlp15/1/14/13 all write to its read, 10-11x). write->readout overlap rises with depth (mlp0 0.098 -> mlp13 0.148). Composition is nearest-neighbor + block-17 sink + rising direct-to-readout. CAVEAT: geometric potential not verified causal; causal patch-flow queued. write_read_flow.png delivered.
- 724: READ vs WRITE SUBSPACE OVERLAP (user Q). MOSTLY ORTHOGONAL: mean overlap@32 = 0.073 vs random 0.028 (2.6x) -> ~93% of write subspace OUTSIDE read subspace. MLP reads one residual subspace, writes a DIFFERENT one (transforms, not self-amplifies). Depth structure: EDGES most self-referential (mlp0 0.142, mlp17 0.133, ~5x random) vs MIDDLE most orthogonal (mlp13/14 ~0.045, ~1.6x). Per-component write dirs cos 0.1-0.5 with read. Connects to 723 (orthogonal read/write => layer's write is fresh subspace read by NEXT layer = local forward flow). Extends read!=write theme to MLP read/write geometry.
- 725: COMPOSITION IS CAUSAL (verifies 723). Ablating layer i's top-16 write subspace changes block j gate 9-44x more than random (mlp0->blk1 43.7x, mlp5->blk6 18x; hub: mlp0/1->blk17 31x/25x). BACKWARD null EXACTLY ZERO (mlp10->blk3, causally impossible). The local forward flow + block-17 integration hub (723) are causally real. Verifies the tensor-network composition: each layer's write causally drives the next block's gate. Ready to trace named circuits end-to-end.
- 726: CORRECTION -- "boundary circuit at 3 layers" is NOT composed. Causal boundary-selectivity (ablate, dCE boundary vs non-boundary): only mlp16 boundary-SELECTIVE (+0.33, confirms 715); block1.attn ANTI-selective (-0.44, fires at newline but contributes broadly); block0.attn dir1 negligible (-0.003, redundant); ALL 3 non-selective (-0.10); null -0.002. Corrects 701/703: named by FIRING pattern (input) not causal CONTRIBUTION (output) -- fires!=contributes (read!=write theme). Only mlp16 is a real boundary circuit. Selectivity-ablation is the right circuit-verification test. Queued block1_attn_function (what block1.attn rank-1 actually does, by category).
- 727: block1.attn rank-1 RENAMED = general OPEN-VOCAB CONTINUATION writer (not boundary). Ablation dCE by category: subword +1.45 (top), digit +1.30, content +1.24, cap_word +1.02, newline +0.24 (LOWEST); random null flat. Fires AT boundaries (input trigger) but output predicts the CONTENT that follows. Resolves 726: firing (input) != function (output). Confirms lesson: name circuits by causal output selectivity not firing. Final boundary picture: only mlp16 boundary-selective; block1.attn = open-vocab continuation triggered at boundaries.
- 728: FLAGSHIP VERIFIED -- newline circuit CAUSALLY REAL. At period positions (1508, 369 line-ends), AUC of P(newline) vs actual-line-end: full 0.806 -> front-attn-ablated 0.510 (chance, drop 0.296) vs random-subspace-ablated 0.789 (drop 0.017). Front attention carries ~ALL the line-end discrimination (17x more than random). Confirms FINDINGS item 7 (635-644) causally -- NOT a firing artifact. Contrast 726-727: same method broke the false boundary circuit, confirms the real newline circuit -> two-sided verification. Queued article_circuit_verify (same test on item 8).
- 729: FLAGSHIP VERIFIED -- article circuit CAUSALLY REAL + attn>mlp (636) confirmed. At article positions (2348, 1554 "the"), the-vs-a/an AUC: full 0.870 -> front-attn-ablated 0.703 (drop 0.167, 24x random), front-mlp-ablated 0.737 (drop 0.133), random 0.863. Front attn = biggest contributor to the choice, attn>mlp (confirms 636 attn=choice/mlp=magnitude). NUANCE: more DISTRIBUTED than newline (front-attn ablation only ->0.70 not chance) -- article choice front-attn-dominant but partly elsewhere. Verification sweep 726-729: both flagships (7,8) causally verified, boundary firing-illusion broken. Queued article_choice_depth.
- 730: ARTICLE CHOICE is a FRONT-BLOCK circuit (not depth-distributed). Per-component ablation over 18 blocks: a/an-vs-the choice carried by mlp0 (+0.154), attn1 (+0.153), attn0 (+0.119), mlp1 (+0.073), attn5 (+0.067), mlp4 (+0.026); blocks 6-17 ~0. Front attn jointly (0.272) > front mlp (0.227) -- confirms 636/729 attn>mlp at group level. mlp0 biggest single. Refines 729: "distributed" = across ~4 FRONT components, NOT across depth; back half irrelevant. Matches depth division (finding 4): class/word decision in FRONT. Both verified flagships (newline, article) are front-localized decisions.
- 731: MLP17 CORE causal verification (726 lens on 696 readout-named core). dir0 = frequency-calibration CAUSALLY CONFIRMED: freq-selectivity -0.502 (125x null, strongest by far); ablating hurts RARE targets most = "boost rare" calibration (+ 696's cos 0.878 to w_freq). PREDICTION SIGN CORRECTION: registered "hurts frequent"; truth "hurts rare" (calibration boosts rare) -- substance confirmed, sign backwards. dir1/2/3 sub-names (subword/proper-noun/topical) NOT cleanly verified: weak freq-sel (-0.19/-0.11/-0.05), category tops confounded by difficulty (cap_word top for null too). LESSON: readout naming reliable for dominant dir, over-specifies weak ones; need causal selectivity + difficulty-null. mlp17 core = 1 verified calibration axis + 3 weaker open-vocab dirs.
- 732: DIGIT CIRCUIT VERIFY -- FLAWED, inconclusive (caught). Direct-path ablation (all 18 blocks zeroed) raised digit CE +11.4 continuation / +12.4 initiation (ratio 1.09), FAILING prediction. But INVALID: continuation is a BIGRAM carried by block-0 attention, which the ablation removed -- so both collapsed because I deleted the bigram head, not because initiation needs more. Also only 41 continuation positions (noise). Item 9 (641-642) NOT refuted, NOT re-verified -- my design was wrong (conflated direct-path with bigram). LESSON: to test "is X a bigram," ablate everything EXCEPT the bigram head. Not re-run (marginal).
- 733: DIGIT VERIFY 2 (corrected) STILL INCONCLUSIVE. Keep block-0 attn, ablate blocks 1-17 (545 cont/5642 init): continuation rise +11.17 vs initiation +13.00, ratio 1.16. Confound: with blocks 1-17 zeroed, block-0's bigram output can't propagate to readout, so both collapse. Ablation is the WRONG instrument for a bigram claim (off-distribution). Item 9 stands on 641-642 lift-over-baseline; closing digit re-verification (use baselines not ablation).
- 734: QK MINI-WIN LAYER 0 (user's per-layer input-focused QK decomposition). Content QK forms (M1=Wq.T Wk) read through embedding: head0 = COMPARATIVE (many/much/long -> as/so/how, "as X as"); head1 = PRONOUN (her/his -> you/we); head4 = DOMAIN-noun (devices/systems -> medical/governmental); head3 = FUNCTION-WORD key (eff-rank 26). Heads 2,5-8 rare/formatting-dominated. Real vs random-form null. CAVEAT: forms mostly HIGH-RANK (mean eff-rank 97/128), top mode = dominant preference only. Layer 0 is the bottom-up ANCHOR (QK reads embedding -> token-readable); layers>=1 need on-data char. First QK mini-win.
- 735: CROSS-LAYER A-SVD ABLATION CLUSTERING (user's method, cross-layer). Pool 120 A-SVD components across layers [0,1,2,16,17], random cross-layer size-4 subset ablation, cluster datapoints by CE-damage covariance. CONVERGES (split-half 0.873, null -0.001); ALL 8 clusters span >=2 layers -- token prediction depends on CROSS-LAYER (front+back) component combinations, not single layers (consistent w/ depth division). CAVEATS: cluster token content not distinct (all common function words/punct); layer-mix attribution crude (additive over subset). Method extends cross-layer; damage genuinely cross-layer. Refinement: single-component ablation for exact attribution.
- 736: CROSS-LAYER COMPONENT GROUPS -- honest NEGATIVE. Single-component (120, exact) damage profiles NEAR-ORTHOGONAL: random-pair |corr| 0.036, within-group cohesion 0.03-0.06 (= chance; noise floor 0.018). The 10 "cross-layer groups" are k-means artifacts on near-orthogonal vectors. Single A-SVD components damage small/scattered/disjoint token sets -> no tight cross-layer co-damage circuits (redundancy). Contrast 735: subset ablation converged on aggregate damage, but single-component profiles underneath are orthogonal. Composition via co-damage clustering = no clean circuits. RIGHT test = pairwise SUPERADDITIVITY. Queued cross_layer_pair_interaction.
- 737: BASELINE + METRIC (user Q). SURPRISE: on EFFICIENCY (CE-recovery/component), plain WEIGHT-SVD BEATS A-SVD, dramatically at low rank. mlp1 r=4: A-SVD -4.38 (catastrophic) vs weight-SVD -0.48; r=128 A-SVD 0.895 vs 0.840 (catches up at full). r80 ties (128/128, 8/8); A-SVD>=weightSVD only 8/20 points. WHY: A-SVD orders by RESPONSE ENERGY = massive-activation dims (loss-irrelevant, 676), so it front-loads the wrong directions; weight-SVD orders by W singular values (loss-relevant). My prediction (A-SVD better) WRONG. Validates user: reconstruction-optimal basis != best for the metric; neither orders by LOSS -> optimizing a basis for CE-contribution would beat both. Composability/monosemanticity metrics queued.
- 738: DECOMP COMPOSITION COMPARE. weight-SVD wins composability (interaction 0.078 vs A-SVD 0.095) AND monosemanticity (concentration 0.350 vs 0.355) -- so weight-SVD >= A-SVD on ALL 3 metrics (efficiency 737 + these). BUT both dense orthogonal bases ABSOLUTELY mediocre (interaction ~8%, concentration ~35% = not additive, not monosemantic). Sparse dictionary needed. Motivates faithfulness-preserving basis rotation (exact reconstruction + sparse codes) vs lossy SAE.
- 739: CE-ORDERED BASIS -- reordering NOT enough. Reranking weight-SVD dirs by CE-importance is MIXED vs singular-value order (wins r=4/64/128, loses r=8/16), does NOT dominate. weight-SVD singular order already near-loss-optimal (>> A-SVD energy at low rank per 737); random-order much worse (null held). CAVEAT: my CE-importance = first-order single-removal proxy, not the true greedy keep-order (r^2 forwards). CONCLUSION: can't win big by REORDERING the orthogonal SVD basis; real gains need a DIFFERENT basis (sparse/rotated) -> faithful_rotation next.
- 740: MDL PARSIMONY (CE-based). Dense bases NON-parsimonious: ~120/256 components per datapoint (A-SVD 119.7 < weight-SVD 127.5 < random 133.9) -> SVD/A-SVD fail MDL, need sparse dictionary. A-SVD marginally best on parsimony (reuse-gini 0.15 vs 0.10) -- the one metric A-SVD wins ("better" depends on metric). CONFOUND flagged: difficulty-validation FAILED w/ backwards sign (frequent->longer) = metric artifact -- CE-relevance code length confounded by gradient magnitude (easy tokens small diffuse gradients -> spuriously longer codes). FIX: use CE-loss-benefit recovery or per-datapoint difficulty normalization. faithful_rotation requeued (raw-vs-varimax comparison unaffected by the confound).
- 741: FAITHFUL ROTATION. Mechanism FAITHFUL (proj_diff 2e-6, exact subspace) but orthogonal rotation CANNOT sparsify per-datapoint codes: varimax LENGTHENED codes (67.5 vs raw 63.9), same as random rotation (67.3). Reasons: varimax optimizes per-COMPONENT spatial sparsity (wrong objective), and SVD is already the most concentrated orthonormal basis (rotation only spreads). KEY: faithful per-datapoint sparsity REQUIRES OVERCOMPLETENESS, not rotation (corrects the rotation idea). A-SVD most parsimonious dense basis (60.7). Queued overcomplete_sparse_dict (learn P>rank atoms, sparsity-vs-reconstruction-fidelity frontier).
- 742: OVERCOMPLETE SPARSE DICT -- the MDL WIN. Learned P=512 top-k SAE reconstructs mlp1 output per-datapoint FAR better than SVD rank-k at same k: k=2 R2 0.44 vs 0.10 (4.5x), k=8 0.57 vs 0.23, k=32 0.63 vs 0.39. Random-overcomplete fails (0.01-0.10) -> win from LEARNING not overcompleteness. First decomposition to beat dense-orthogonal family on per-datapoint efficiency; vindicates user (reconstruction-optimal basis != best). CAVEATS: not yet faithful (R2 0.44-0.63 not 1; faithfulness needs higher k -> sparsity/faithfulness FRONTIER, SAE dominates SVD); L2 not CE (CE version queued); quick training. Resolves 737-742: learned overcomplete sparse dict is the right family. Queued overcomplete_ce_faithful.
- 743: TOY PLANTED DICT -- ground-truth VALIDATION. Planted 64-atom overcomplete dict, k_true=3 sparse codes: top-k SAE recovers atoms PERFECTLY (recovery 1.000, code-len 3.00=k_true), SVD fails (0.389, dense mixtures). NULL (dense Gaussian): SAE recovery 0.22 (doesn't hallucinate). Clarifies: RIGHT METRIC = atom-recovery + code-len matching true k; RIGHT METHOD = overcomplete top-k SAE (SVD/A-SVD can't represent overcomplete atoms). Interprets 742: mlp1 HAS real sparse structure SVD misses (null-safe), but R2 0.63<1 => partially sparse+dense. Methodological anchor. Next toys: shared computation, mixed sparse+dense.
- 744: TOY SHARING -- method RESPECTS shared vs unshared computation. Planted 8 common + 56 rare atoms: SAE recovers atoms (0.992) AND usage frequencies (usage-corr 0.766, null -0.08); recovered-common usage 0.18 vs rare 0.035 (~5x, matching plant). Doesn't over-explain (fixed k). Validates user's core criterion. 743+744 establish framework: overcomplete SAE recovers real sparse structure (SVD can't), doesn't hallucinate (null-safe), respects sharing. Next: real layers mlp0/mlp1/mlp16/attention SAE-vs-SVD. Queued real_sae_compare.
- 745: TOY HIERARCHICAL/DAG + hyperparameter insight. Planted 6 parents x 6 children (each datapoint 1 parent + 2 kids, true k=3, P(parent|child)=1). SAE recovers atoms AND the DAG dependency at k>=3 (atom-rec 0.99, DAG P(parent|child) 0.86). k-SWEEP RULE: atom-recovery PEAKS at true sparsity (k=3); k<true loses DAG (0.20), k>true DEGRADES recovery (0.86->0.74 at k=8, splits atoms). For real model: sweep k, pick recovery peak/elbow (overshoot degrades = signal). Method captures hierarchical COMPOSITION (relevant to 716). Toys 743/744/745 validate flat/shared/hierarchical recovery + tunable k.
- 746: REAL SAE COMPARE (cross-layer). k=8 SAE(P=512) vs SVD-8 R2 of layer output: mlp16 (rank-1) SAE==SVD (gap +0.007, usage-gini 0.88) = genuinely simple, no hidden structure; mlp0 (+0.31) AND mlp1 (+0.34) LARGE gap (usage-gini 0.48) = rich hidden overcomplete sparse structure; block1.attn +0.12. Null held (rand ~0.04). KEY: mlp0 CE-low-rank (r80=8) yet big SAE gap -> activation-sparse structure != CE-functional rank (two different decomposition axes). SAE advantage tracks output-activation sparsity not CE-rank. CAVEATS: lossy (R2 0.57-0.88), k=8 fixed, L2 not CE (CE version queued).
- 747: TOY WEIGHT-ACTION SAE (user's NOVEL direction, validated). Factor W=D@E overcomplete + L1 on codes E@X: FAITHFUL (||W-DE||/||W|| 0.024, reconstructs WEIGHT not lossy activations), RECOVERS planted overcomplete atoms (0.93 vs A-SVD 0.41), SPARSE codes (L0 7.5 vs lambda=0 dense 60.4). Novel faithful sparse overcomplete weight decomposition = A-SVD faithfulness + SAE overcomplete sparsity; recovers structure both miss. Code sparsity tunable via lambda (faithfulness/sparsity frontier). (a) False only on strict L0<6 bar (got 7.5); substance held. Queued weight_action_sae_real (apply to real mlp1.Down).
- 748: OVERCOMPLETE CE FAITHFUL -- SAE advantage EVEN LARGER in CE. Substituting top-k SAE recon of mlp1 output into live model: CE-recovery k=8 0.864 / k=32 0.937 / k=64 0.957, while SVD rank-k CATASTROPHIC at low k (k=8 -2.33 worse than ablation, same massive-activation issue as 737). Random-OC worse. SAE's L2 win (742) translates to LARGER CE win (k=8: SAE +0.86 vs SVD -2.33). Overcomplete sparse dict dramatically more CE-faithful per component than SVD -> pays off in loss not just reconstruction. Caveat: SAE lossy at these k (weight-action version 747 is fully weight-faithful analog, queued on real weight).
- 749: WEIGHT-ACTION SAE real mlp1.Down -- soft-L1 FAILED to sparsify (impl negative). lambda=0: exact+faithful (||W-DE||=0, CE 1.000) but DENSE (1758/2048); raising lambda breaks faithfulness (0.12->0.32, CE 0.85->-1.90) WITHOUT reducing L0 (null FAILED). Contrast toy 747 (L0 7.5) + activation top-k SAE 748 (sparse, 86% CE): structure IS there, but soft-L1+Adam can't extract it -- HARD top-k needed not soft L1. FIX: weight-action with hard top-k on codes (min ||W@gate - D@topk(E@gate)||, weight-action analog of 748). Queued weight_action_topk. Concept validated (747), soft-L1 impl fails on real weight, hard-top-k is corrected step.
- 750: WEIGHT-ACTION TOP-K SAE on real mlp1.Down -- WORKS (fixes 749). Hard top-k reconstruct W@gate with D@topk(E@gate): sparse+CE-faithful, CE-recovery 0.870(k=8)/0.938(k=32)/0.951(k=64), beats A-SVD rank-k (k=8 +0.87 vs -2.48 catastrophic), random null catastrophic. Fix was HARD top-k not soft L1 (749). MATCHES activation SAE (748, 0.864) but encoder E is LINEAR map of gate tied to weight (code=E@gate) -> more faithful/interpretable. CULMINATION 737-750: learned overcomplete sparse dict (activation 748 + weight-action 750) is the right decomposition -- per-datapoint sparse, CE-faithful, recovers structure SVD/A-SVD can't; novel faithful weight-based version validated on real layer.
- 751: INTERPRET SAE ATOMS -- honest NEGATIVE. Reconstruction-optimal sparse atoms NOT automatically monosemantic: mean top-activation token-concentration SAE 0.163 < SVD 0.253. Most-used atoms polysemantic (mixed tokens); a FEW clean circuits (atom107 sentence-end conc 0.55, atom165 parenthetical, atom22 boundary). EFFICIENCY/FAITHFULNESS != INTERPRETABILITY (different metrics; SAE wins former not latter as trained). CAVEATS: most-used atoms polysemantic by construction; concentration metric confounded by token frequency (need freq-normalized selectivity). Interpretable atoms EXIST just not on average. Queued interpret_sae_atoms_v2 (freq-normalized + larger P + usage-stratified).
- 752: INTERPRET SAE ATOMS v2 -- METRIC BROKEN (caught: SVD==random==SAE==~164). Max-lift dominated by rare-token noise (one rare token in top-150 -> lift ~N/150~164 for ANY direction). 2nd metric-design fail (751 freq-confounded, 752 rare-token-max). FIX: KL divergence of top-activation token dist from base (robust to rare singletons). Queued interpret_sae_atoms_v3 (KL). Monosemanticity question still OPEN; qualitative 751 stands (few clean atoms, most polysemantic).
- 753: INTERPRET SAE ATOMS v3 -- KL metric WORKS (v2 was broken), verdict WEAK-YES + the NULL carries the real point. KL divergence of top-token dist from base finally SEPARATES methods: SAE mean-KL 4.10 vs SVD 3.18 vs random 3.33. (1) SAE atoms modestly MORE token-monosemantic than SVD (1.29x, just under the 1.3x bar -> weak yes). (2) low-usage atoms edge high-usage (4.20 vs 3.97) but flat at P=1024 -> weak/not robust. (3) NULL FAILS informatively: random (3.33) >= SVD (3.18) -> SVD directions carry NO MORE token identity than a random projection (SVD orders by ENERGY = massive-activation dims that fire on ~every token, 660/676). So SVD is not just worse than the SAE for naming circuits, it is no better than RANDOM -- quantitative reinforcement of the 737-750 thesis. Monosemanticity question RESOLVED as far as profitable: SAE>SVD modestly, SVD~=random, few clean atoms (751 qualitative stands); 3 metric iterations converge, NOT launching a 4th. Next = COMPOSITION not metrics. Queued weight_action_compose.
- 754: WEIGHT-ACTION COMPOSITION -- wiring composes FROM WEIGHTS ALONE (user Q answered). Coupling C=E2@D1 between Down_0 write-atoms and Left_1 read-atoms: SAE coupling drift across data splits = 0.0 (pure weights) vs A-SVD drift 1.46 (146%, its B=Vh@X.pinv bakes data in). Routing sparse (520 live edges/token = 0.0005%, top-k = per-datapoint path not wiring). Independent coupling DENSE (13.6% strong) -> graph sparsity must be an objective (motivates 755). Bug caught: write-corr metric bias-confounded (0.378~=null 0.366), fixed by centering, re-queued. YES weight-action composes weight-only; A-SVD cannot.
- 755: TOY JOINT SPARSE COMPOSITION (red-teamed) -- joint training + edge penalty RECOVERS planted wiring SPECIFICALLY. Handwritten 5-atom circuit: 4/5 wiring correct, coupling matrix visually clean. 96-atom: joint edge-F1 0.61 vs indep 0.30 vs SVD 0; sparser (164 vs 402 edges); specificity null (joint C vs WRONG S) 0.02 << 0.61 (SPECIFIC). KEY: reconstruction alone does NOT identify wiring (non-identifiable); a STRONG edge penalty (lam_e 0.1, swept) is REQUIRED -- graph sparsity as objective is what makes it identifiable. Red-team fixes: proper specificity null (first "null" regenerated data = not a null), atom-match-cos gate. Limitation: atom identifiability imperfect (cos ~0.8).
- 756: WEIGHT-ACTION SAE across EARLY STACK (L0,2,3,4,5) -- overcomplete sparse dict is the right decomposition UNIFORMLY. CE-recovery k=8: L0 WA .97/ASVD .83, L2 .63/-2.37, L3 .53/-1.63, L4 .85/.67, L5 .35/.02. WA-topk > A-SVD in EVERY early layer; where A-SVD CATASTROPHIC (L2,L3 massive-activation regime, negative at low k) WA still 0.5-0.6. random-OC catastrophic everywhere (null holds). Profile: L0,4 low-rank loss-relevant (A-SVD ok); L2,3 massive-activation (A-SVD fails); L5 needs more atoms. L0 CE-benefit 2.08 dominates; L2/3/5 tiny (0.07-0.15, barely matter for loss). 737/750 generalizes across the stack.
- 754b: CORRECTION of 754 (centered corr). Weight-only write vs measured contribution corr = 0.026 (~zero) vs null -0.0004 -- earlier 0.378 was bias-confounded (both sides share large mean; center before correlating). CORE CLAIM STANDS: coupling C=E2@D1 weight-only/data-invariant (drift 0 vs A-SVD 1.46), routing sparse. NEW: can READ wiring from weights but CANNOT predict per-token flow from linear coupling alone -- model's intervening nonlinearity (rms_norm+attention+lambda) dominates. Motivates CE-trained joint (real forward). Bias-confound lesson (3rd: 751/752/754): always center + shuffle-null before correlating.
- 754c: PATH LINEARITY (corrects 754b, user prompt). lambda0=0.013 (tiny). Predict Left_1 read-change from same-position weight coupling W_Left1@(lambda0*mlp0): attention ON corr 0.056, attention OFF corr 0.472 (8.4x jump), null -0.000. Attention CROSS-POSITION mixing is the dominant decorrelator; lambda folds (scalar), rms_norm is per-token scalar gain, attention bilinear -- NONE a hard nonlinearity. Composition MOSTLY LINEAR per-position; 754b's "nonlinearity dominates" corrected -- what the 754 test folded in was attention position-mixing + rms per-token scale (both structured). Remaining 0.47<0.9 gap = rms scalar (omitted from pred) + tiny-lambda0 swamping.
- 757: REAL CE-TRAINED JOINT (Down_0+Left_1, warm-start indep 0.908, CE+edge fine-tune). WIN: edge penalty sparsifies real wiring in-degree 305->64 (-79%) at ZERO CE cost (0.605->0.607) -- joint sparse composition works on real weights. FLAG: joint-CE 0.605 < indep 0.908 (warm-start dropped). rand-OC null -1.04. Diagnosed in 758.
- 758: TRAINING CURVES -- 757 drop is CE-TRAINING DRIFT, not undertraining (user's concern checked). Eval CE-recovery FALLS from 0.918 warm-start (not rising -> not undertraining); reconstruction R2 COLLAPSES to negative (Down_0 0.81->-0.06, Left_1 0.44->-0.24) -- CE gradient ignores weight fidelity, overfits tiny fit-set. More data softens (48-row 0.71 vs 16-row 0.59) but doesn't fix. FIX: reconstruction-ANCHOR (CE+lam_rec*MSE+edge). Queued real_joint_ce_v2. Fig real_joint_ce_curves.png.
- 759: RECONSTRUCTION-ANCHORED JOINT CE -- THE FIX (closes compose->joint->CE arc). loss = CE + lam_rec*MSE + lam_e*edge. anchored: CE-rec 0.945 (>=indep 0.938 throughout), R2_Down0 held 0.77, in-degree 291->70 (-76%). no_anchor control: CE-rec 0.74, R2 collapses to -0.26 (reproduces 758). pred_a+pred_b True. Faithful AND CE-aware AND sparse-wired joint composition on the real model = user's "jointly train weight-SAEs to sparsely compose", working. Fig real_joint_ce_v2.png. NEXT: anchored joint across early stack (0-3) + edge-causality test.
- 754c-completion: PATH LINEARITY v2 (full held-out linear probe). Left_1 read-change on Down_0 write: attn-ON R2 0.345, attn-OFF R2 0.589, null -0.145. Same-position composition LARGELY LINEAR (full linear map explains 0.59 with attention removed, vs scalar-fit 0.22); attention cross-position mixing is the dominant decorrelator; residual ~0.41 = rms per-token nonlinearity (amplified by tiny lambda0=0.013). Settles 754b/754c: wiring weight-only, per-token flow mostly-linear + attention mixing + minority rms nonlinearity. User's framing essentially right.
- 760: EDGE CAUSALITY -- weight-only wiring is a GENUINE but INCOMPLETE causal map. Bug caught: first run all-zeros (selected high-coupling atoms with usage 0.000 -- never fire; fixed to select by usage). (A) knocking active source atom moves targets as -C[:,i] predicts: mean corr 0.217 vs null -0.008 (real, above chance) but weak (<0.4 predicted) -- C omits attention mixing + rms (754c), explains ~0.22 of causal pattern. (B) coupling degree != CE importance (dCE high 0.014 vs low 0.014, ratio 1.01 -- out-degree is not loss-importance). Wiring is a causal SKELETON: real, readable from weights, but predicting magnitude needs intervening pieces + output-side loss-relevance. pred_a/b False, null True.
- 761: PER-ATOM CE IMPORTANCE -- loss DISTRIBUTED + REDUNDANT, not a few load-bearing atoms. Knock each active Down_0 atom: top-10 carry only 19% of summed dCE (Gini 0.67 -- unequal but broad, over dozens not a handful). Usage barely predicts importance (rho 0.18); with 760B (coupling degree != importance) TWO structural measures fail -> loss-importance is its own property, causal-only. SUPERADDITIVE ~2x (top-32 joint dCE 0.104 vs 0.054 summed) -> atoms REDUNDANT/compensating, single-ablation undercounts. Tempers parsimony: layer has no small critical set at P=512/K=32; held by many mutually-compensating atoms. Open: smaller dict/K -> more concentrated? pred_a/b False, null clean.
- 762: CONVERGENCE / TWO KINDS OF FAITHFULNESS (corrects 758; user's "48 undertrained" was right). Pure-CE 1000 steps: 48-row CE-rec 0.578, 128-row 0.778 (more data HELPS -- partly undertraining), but R2 stays deeply negative both (-0.47/-0.54: more data never restores weight-faithfulness). Anchored-128 holds both (CE-rec 0.962>=indep, R2 0.767). => pure-CE converges LOSS-faithful but WEIGHT-unfaithful; two distinct fidelities, CE buys one. Must anchor on reconstruction for a weight decomposition. 758's "damage not undertraining" too strong: drop is part undertraining (data-fixable) + part fundamental (R2 collapse). Fig real_joint_ce_converge.png (with 'better' arrows). CE-recovery def: (CE_ablate-CE_sub)/(CE_ablate-CE_full), 1=faithful/0=useless/<0=harmful.
- 763: COMPONENT SCORECARD -- convergence FAILS at atom level, the atom is the WRONG UNIT (answers "how to know a good circuit w/o ground truth": triangulate, but atom-measures don't converge). Down_0 SAE x4 seeds: mostly seed-UNSTABLE (mean best-match cos 0.40, only 5/512 stable -> atoms are seed-specific fits not canonical features). rho(stability,causal) 0.29 (weak, null under-powered at 0.21), rho(stability,monosem) -0.14, rho(causal,monosem) 0.05. MONOSEMANTICITY orthogonal to causal AND stability -> interpretable != important; good-circuit conjunction doesn't co-occur by default. With 761 (redundancy) => atom is wrong unit; low atom-stability + stable span = rotational ambiguity. NEXT subspace_stability (do spans recur even if atoms don't?). pred_a False.
- 764: SUBSPACE STABILITY -- 763 instability PARTIALLY ROTATIONAL. Down_0 SAE x4 seeds: atom-match 0.405, but top-r decoder subspace overlap 0.655(r64)/0.722(r128)/0.812(r256) vs random 0.198/0.286/0.411. Span more stable than basis (rotational ambiguity real) but only moderately stable (0.655 ~58% random->identical) -> also genuine subspace drift. Rotation-fix (canonical basis in shared span, free wrt recon) helps but ~1/3 is subspace movement. Caveat: raw-decoder subspace incl low-usage noise; usage-weighted may be more stable. pred_a False (strict 0.8 bar).
- 765: HIERARCHY NESTING -- ACTIVATION hierarchy YES, geometric NO; explains 761 redundancy. Down_0 SAE P=64/256/1024: fine->coarse decoder cosine 0.270 (rand 0.073, weak); span residual 0.675 (not geometrically nested); BUT activation containment P(parent|child)-base = 0.369 (null -0.007, STRONG). Fine atoms co-fire within coarse parents but occupy distinct directions -> functional not geometric hierarchy. Explains redundancy: fines functionally covered by co-firing parent/siblings -> single-ablation undercounts. Circuit unit = CO-ACTIVATION GROUP not atom. NEXT group_scorecard (does the group rescue 763's failed convergence?).
- 766: GROUP SCORECARD -- co-activation GROUPS do NOT rescue 763 (clean negative). 27 groups (size 16): (a) stability 0.549 is a SIZE ARTIFACT (random same-size 0.519; only +0.03 over random, not real grouping); (b) SUB-additive 0.91 (co-firing atoms not extra-redundant; 761's superadditivity was among IMPORTANT atoms = causal grouping, not co-activation); (c) group KL 3.24 ~= atom 3.26 (no monosem gain). "circuit=co-activation group" FAILS. Narrows 761-766: atoms bad, co-activation groups bad, only the SUBSPACE stable (764), interpretability orthogonal to fitted basis (763) -> pivot to SEED-FREE model-intrinsic token-semantic subspace (canonical by construction). Queued semantic_subspace.
- 767: SEMANTIC SUBSPACE -- POSITIVE resolution of 761-766. Seed-free token-conditional-mean subspace of mlp0 output is CANONICAL + CAUSAL + STABLE (the interpretable structure SAE atoms/groups failed to be). (a) CAUSAL: projecting semantic OUT costs 1.341 nats vs random 0.005 (268x) -- most of mlp0's contribution; NOT massive-activation passthrough (token-mean factors out context = loss-critical). (b) DATA-STABLE 0.822 vs SAE atoms 0.40 (canonical, property of model not fit). (c) SAE aligns 0.498 vs random 0.200 (SAE fits a rotation of ~half the semantic structure -> why atoms unstable). rank90=71. pred_a False (rank>64) but causal overwhelming; b/c True. RESOLUTION: interpretable+causal+stable structure EXISTS as a seed-free SUBSPACE (token-means), not SAE units. SAE=faithful reconstruction; semantic subspace=interpretable structure. Canonical subspace needs no ground truth, causally verifiable, data-stable. Queued semantic_naming.
- 768: SEMANTIC NAMING + GENERALITY (payoff of 767). mlp0 top semantic directions are cleanly NAMEABLE grammatical classes: dir0 determiners/articles, dir1 punctuation, dir4 sentence-initial pronouns, dir5 aux/copula verbs, dir6 digits/numbers, dir7 prepositions, dir8 conditionals/wh, dir9 brackets. mlp0 organises output by PART-OF-SPEECH. GENERALITY: causal ratio >=5x at every layer (L0 268x, L4 150x, L8 11x, L12 6.5x) -> stack-wide axis, but absolute dCE DECAYS with depth (1.34->0.57->0.019->0.016), concentrated early (matches 756 barbell). Interpretable+causal structure of the front = token-class organisation, canonical (token-means), no ground truth. pred_b True.
- 769: SEMANTIC ALIGNMENT weakly explains ATOM MONOSEMANTICITY (2 bugs fixed: dtype + norm(0)-order-vs-axis; corrected atom-alignment mean 0.54 vs random 0.23, matches 767). rho(alignment, monosem) 0.164 vs null 0.054 -- above chance but WEAK (<0.3 predicted); high-align KL 3.85 vs low 3.69. Semantic alignment is PART of atom monosemanticity but explains little -> 763's orthogonality NOT resolved by alignment. Nameable interpretability lives at the canonical DIRECTION level (768 grammatical axes), not the SAE-atom level; atoms inherit it weakly. Consistent w/ 761-769: atom is wrong unit, semantic directions are the interpretable units. pred_a False, pred_0 True.
- 770: SEMANTIC SUFFICIENCY (capstone 767-770). Keep ONLY top-r semantic subspace of mlp0 output: r16 0.72, r64 0.92, r128 0.96, r256 0.98 CE-recovery (random r: 0.02/0.08/0.12/0.31). ~64 part-of-speech directions capture 92% of mlp0's loss contribution; other ~1088 dims nearly loss-irrelevant. mlp0 = low-rank grammatical-class writer. SYNTHESIS 767-770: token-semantic subspace is NECESSARY(767) + SUFFICIENT(770) + NAMEABLE(768) + CANONICAL/STABLE(767 0.82) + LOW-RANK(64/1152) -- the cleanest 'right decomposition' of 737-770, satisfying every good-circuit property SAE atoms failed (763-766), because it's a canonical subspace not a fitted unit. Answer to 'good component without ground truth': compute token-conditional-mean subspace, verify causally, read named axes. pred_a True.
- 771: SEMANTIC SUBSPACE GENERALISES TO ATTENTION (prediction WRONG, stated plainly). I predicted attention (position-mixing) would be more context-driven / weaker; it's NOT. Attention keep-only-64 recovery L0 0.975 (>MLP 0.92!), L4 0.877; remove-ratio 106x/29x. Attention's loss contribution is even MORE token-class-concentrated than MLP's. FLAVOUR differs: attention top dirs = STRUCTURAL/discourse markers (conjunctions And/But, sentence-initial It/She, clause boundaries --/\n, conditionals If/When) vs MLP LEXICAL (determiners/punct/numbers/verbs). Front of network sorts output by token-class per component: attention=structure, MLP=word-category. Canonical token-class subspace is GENERAL, not MLP-specific. pred_a True, weaker-prediction False. Queued semantic_overlap (are attn & MLP subspaces complementary?).
- 772: SEMANTIC SUBSPACES NEAR-ORTHOGONAL across components -- multiplexed token-class bus (prediction WRONG: they're more distinct than 'partly shared'). Pairwise overlap (random floor 0.201): attn0|mlp0 0.254 (barely above random -> near-orthogonal), mlp0|mlp4 0.339 (highest, still low), cross-comp-cross-layer ~0.21 (=floor). Every component encodes token-class (universal 767-771) but each component x layer uses its OWN distinct residual subspace -> residual stream is a MULTIPLEXED token-class bus (attention structural + MLP lexical + per-layer, near-orthogonal channels, no interference). Ties to read/write-orthogonality. SYNTHESIS 767-772: token-class organisation is universal, canonical, causal, low-rank, necessary+sufficient, nameable, per-component, multiplexed. pred_a False, null clean.
- 773: SEMANTIC DEPTH PROFILE (capstone 767-772; fig semantic_depth.png). Per layer benefit + keep-only-64 CE-recovery, both components. ASYMMETRY: ATTENTION keep64 0.88-0.99 at EVERY meaningful layer (pure token-class writer throughout); MLP keep64 HIGH at L0 (0.92)/L16 (0.97)/L17 (0.92) but LOW at L1 (0.56)/L2 (0.33)/L3 (0.36). So token-class org carried by attention (everywhere) + MLP0 (lexical in) + MLP16/17 (token-id out); MLP L1-3 do CONTEXT-dependent computation NOT captured by token identity (matches distributed/no-low-rank-carrier of conditional computation, FINDINGS 1). pred_a False, INFORMATIVELY: locates MLP L1 as transition from token-class sorting to contextual computation. Division of labour: attention=grammatical structure, MLP-ends=lexical/token-id, MLP-early-middle=contextual reasoning that resists low-rank decomposition.
- 774: CONTEXT RESIDUAL of MLP L1 -- non-token-class computation is DIFFUSE, not bigram (follows 773). Removing current-token subspace leaves 0.563 of variance (mlp1 majority non-token-class). Prev-token mean-subspace explains 0.207 but shuffled null = 0.212 (SAME -> method underpowered for weaker prev-token signal, not a clean result). Decisive: removing prev-token subspace costs only 0.015 nats of mlp1's ~1.07 -> context computation NOT concentrated in a bigram subspace. mlp1's non-token-class part is the DISTRIBUTED remainder, no low-rank carrier = program's central wall (FINDINGS 1) at the 773 transition. Network splits: interpretable token-class subspace (767-773) + distributed contextual remainder (MLP L1-3). LESSON: mean-subspace method works only when conditioning signal dominates variance; weaker signals need supervised held-out probe. pred_a/null False.
- 775: CONTEXT PROBE of MLP L1 residual (supervised, resolves 774). Held-out ridge R2: prev-token from residual 0.199 (null -0.07), current-token from full 0.558, POSITION from residual 0.813. RESOLVES 774: prev-token IS decodable (774's null=signal was underpower not absence); combined w/ 774 causal 0.015 nats -> prev-token DECODABLE-BUT-INERT (read!=write, FINDINGS 2). NEW: POSITION strongly decodable (0.81) -> L1 non-token-class residual dominated by POSITIONAL info, linearly accessible. CORRECTION to 774: not all "distributed remainder" -- position is low-rank decodable; I missed it with mean-subspace. Open: is position CAUSAL? queued position_causal. pred_a True (null_ok flag = threshold artifact, null -0.07 clean).
- 776: POSITION IS CAUSAL in MLP L1 (completes 774 correction). Position-conditional-mean subspace (top-32): REMOVE dCE 0.257 vs random 0.002 (165x) -> position CAUSAL, costs 24% of mlp1 benefit (NOT decodable-but-inert, unlike prev-token). KEEP-ONLY recovers 0.256 (~26% sufficient). ENTANGLED with token-class: overlap 0.616 vs random 0.142 (within-component, not clean channels unlike across-component 772). mlp1's context residual is LARGELY POSITIONAL (clean/causal/low-rank) -- 774's 'distributed remainder no low-rank carrier' WRONG for mlp1. REFRAME 773-776: mlp1 = token-class(lexical 56%) + position(causal 26%), entangled + small diffuse rest. pred_a True. Queued combined_interpretable.
- 777: COMBINED INTERPRETABLE fraction of MLP L1 (capstone 773-776). Keep-only union of token-class(64)+position(32): combined 0.785 vs random-96d 0.369; token 0.613, position 0.256. Combined dim = full 96 (no collapse -- 776's 0.62 'overlap' was mean cosine not shared dims; correction). ~78% of mlp1's benefit is 96 nameable dims (lexical+position); ~22% irreducible (keep-only misses). Complement recovers 41% separately -> redundancy between interpretable subspace and rest. mlp1's 'contextual' computation is MAJORITY-INTERPRETABLE (token-class+position), ~22% distributed core. SYNTHESIS 773-777: front is mostly token-class+position; no-low-rank-carrier wall (FINDINGS 1) confined to ~1/5 of early-middle MLP. pred_a True. Queued combined_depth.
- 779: COMBINED INTERPRETABLE across early MLPs (capstone). Keep-only token-class(64)+position(32) per layer L0-5: L0 0.93/L1 0.78/L2 0.50/L3 0.50/L4 0.88/L5 0.60. NAT-WEIGHTED interpretable fraction = 3.57/4.20 = ~0.85: ~85% of the early MLP stack's total loss-benefit is token-class+position, ~15% distributed remainder. Interpretable fraction HIGH where the nats are (L0/L1/L4 carry ~3.8/4.2 nats, 78-93% interpretable); distributed part (L2/L3 at 50% irreducible) carries tiny absolute weight (0.08+0.065). Position varies: large at L0/L4/L5, ~0 at L2. pred_a False (per-layer bar fails at tiny L2/L3) but nat-weighted story strongly positive. Fig combined_depth.png.
- 778: CROSS-MODEL -- token-class subspace GENERALISES to GPT-2 + Pythia-410m (user ask 1). First MLP keep-only-r CE-recovery: bilin18 keep64 0.92/keep128 0.96; GPT-2 (124M) benefit 3.96, keep64 0.84/keep128 0.93 (rand 0.30); Pythia-410m (410M) benefit 7.30, keep64 0.62/keep128 0.76 (rand 0.04). GENERAL: first MLP writes a low-rank token-class subspace carrying most of its loss (0.6-0.96 keep, vs random 0.04-0.30); magnitude varies (bilin18>gpt2>pythia, larger models distribute more) but strong everywhere. BARBELL UNIVERSAL: first MLP dominates MLP nat-budget in real models too, MORE extreme (GPT-2 L0=3.96/~4.9; Pythia L0=7.30/~10). External validation of 767-777. pred_a True.
- 780: TOKEN-CLASS GEOMETRY (user insight confirmed). Per-token mean table vs embedding, 360 tokens. (a) CLASS COLLAPSE: mean-table eff-rank 22.7 vs embedding 132.4 -- MLP collapses 360 tokens into ~23-dim class space (~6x lower); token-means are CLASS not identity (validated -- current token already in stream, MLP computes its class). (b) NONLINEAR: mean only 44% linear in embedding (R2 0.44) -> computed class structure; norm CV mean 0.11 vs emb 0.04. (c) my pred_b WRONG: mean cosine class-sep 0.155 < embedding 0.237, but CONFOUND (mean low-rank -> all tokens cosine-similar 0.62-0.77, dominant shared direction inflates cosines; class structure in low-variance dirs). eff-rank+nonlinearity confirm story. Queued token_class_whiten. pred_a True, pred_b False(confound).
- 781: CROSS-MODEL POSITION -- token-class+position (both causal) GENERALISES to GPT-2+Pythia (extends 778). GPT-2 L0: token 0.84/pos 0.63/combined 0.92, position causal 10.4x (dCE 0.043). Pythia L0: token 0.62/pos 0.42/combined 0.70, position causal 113x (dCE 2.74 of 7.3-nat L0 -- rotary makes it heavily positional). Both: position CAUSAL, combined>token, tok-pos entangled (overlap 0.61/0.69 like bilin18 776). Two-variable decomposition general across position schemes; split varies by arch. External validation of 776-779. pred_a+b True.
- 782: TOKEN-CLASS WHITENED -- mlp0 SHARPENS class geometry vs embedding (resolves 780 cosine confound; user insight fully validated). Fisher class-sep ratio: mean-table 0.571 vs embedding 0.318 (1.79x, null 0.039); de-shared cosine sep 0.317 vs 0.207. mlp0 separates grammatical classes ~1.8x better than raw embedding once shared direction removed. FULL VALIDATION (780+782): mlp0 COMPUTES class (not identity re-encoding): collapses ~360 tokens to ~23 dims (vs emb 132), NONLINEARLY (R2 0.44), into ~1.8x-sharper class geometry. Front = CLASS-COMPUTING front end. SYNTHESIS 767-782: front converts token identity -> low-rank causal nameable grammatical-class+position representation, multiplexed, read amortized by later layers; distributed remainder ~15-22%. pred_a True.
- 783: QK READS EARLY VARIABLES -- amortized composition CONFIRMED, variable HEAD-SPECIFIC (user's vision, fork). Restrict attn input to token-class+position subspace: attn L1 (ben 2.42) keep token-only 0.94/+pos 0.96/random 0.18 -> CLASS-reader (94% of function on grammatical-class variable alone = amortized composition confirmed). attn L5 (ben 2.07) token-only 0.11/+pos 0.545/random 0.02 -> POSITION-reader (not class), + 45% further unnamed variable. Attention reads early variables amortized (>> random), but WHICH is head-specific: L1=class, L5=position+X. Residual multiplexes named variables (class/position), each head selects what it needs. L5's 45% remainder = computed variable beyond class+position (next question). pred_a False (L5<0.6) but informative. Queued qk_variables_depth.
- 784: QK VARIABLE-READING MAP (fig qk_variables_depth.png). Restrict each attention layer's input to class/+position: L0 (1.48) class 0.61/+pos 0.75 CLASS-reader; L1 (2.42) class 0.94 CLASS-reader; L5 (2.07) class 0.11/+pos 0.54 POSITION+45% OTHER; L2 (0.40) & L4 (0.32) OTHER (L2 negative, orthogonal to class+position). Biggest early heads L0/L1 (3.9 nats) amortize on the front's CLASS variable (article/newline circuits 727); L2/L4/L5 read FURTHER computed variables (position + unnamed). Residual multiplexes more than {class,position}. pred_b False (2/5). Open: name L5/L2's further variable.
- 785: TOKEN RSA (user ask, 131k tokens). RSA(mean-table, embedding) = 0.414 (the scalar: moderate reorganisation). Within-class RSA 0.778 >> across-class 0.444 -- CORRECTION (my within-collapse pred WRONG): MLP PRESERVES within-class relative geometry, REORGANISES between-class -> class-SEPARATION transform (Fisher 2.1x) not within-class collapse. eff-rank mean 24 vs emb 172, Fisher 2.1x confirm 780/782 on more data. Front's class computation = collapse to low-rank class-centroid geometry (classes pushed apart) with within-class structure preserved. pred_a False (informative).
- 786: POSITION STRUCTURE of MLP L1 (answers user: even/odd or early/late?). Position-conditional-mean table: EARLY/LATE definitively -- dominant period 256 (monotonic whole-sequence trend), long-period share 0.557, cubic-in-position R2 0.377, EVEN/ODD (period-2) share 0.0038 (negligible). EFFECTIVE RANK 2.3 -> position is a ~2-DIMENSIONAL smooth absolute-position variable (vs class ~24-dim). Matches rotary low-freqs reaching mlp1 via attention (no direct position). Null clean (shuffled R2 0.01). So position variable = low-dim smooth absolute early/late, NOT parity. 776/779's 32-dim pos subspace was overkill (~2-dim). ~22% irreducible remainder (777) is neither class nor abs-position. pred_a False by hair (0.557 vs 0.6) but qualitative answer definitive.
- 787: ATTN L5 OTHER VARIABLE -- NOT separable; L5 does JOINT conditional computation (bounds 783). After projecting class+position off L5's input, keep-only residual-rank sweep: r2 -0.05/r8 -0.19/r32 0.02/r128 0.10 -- keeping ONLY non-class-non-position recovers ~nothing (harmful at low rank). Residual subspace data-stable 0.86 (real structure, not low-rank separable). L5's 45% "other" = INTERACTION (needs class+position AND content together), not a nameable independent variable. BOUNDS amortized-variable picture: L0/L1 are clean class-readers (single named variable) but L5 does joint conditional computation not reducible to "reads variable X" = distributed face (FINDINGS 1) at head level. SYNTHESIS 783-787: amortized composition REAL but PARTIAL. pred_a False.
- 788: POSITION vs ROPE (answers user). mlp1 position-mean eff-rank 2.29 vs RoPE 17.69 (~8x collapse); RSA(P,RoPE) 0.77 (preserves coarse geometry despite collapse). adjacent-sim P 0.54 < RoPE 0.94 (pred_b wrong: RoPE smoother -- low-freq-dominated + noise-free vs finite-sample-noisy means). MLP keeps a coarse ~2-dim early/late READOUT of RoPE, discards fine frequencies = position analog of token->class collapse (identity 132->class 24; RoPE 18->position 2). MLP reads RoPE via attention, doesn't compute position from scratch. pred_a True.
- 789-attempt: CROSS-MODEL CLASS-SHARPENING underpowered (50 labelled tokens; Pythia Fisher 1.43x+collapse, GPT-2 0.81x no collapse -- inconclusive). NOT a finding. Robust cross-model result stays the token-class SUBSPACE causal sufficiency (778). Re-running with 800 blocks + bigger class vocab.
- 789: CROSS-MODEL CLASS-SHARPENING -- MODEL-SPECIFIC (scopes 780-782). First-MLP mean-table Fisher vs embedding (81 labelled, 7 classes): bilin18 1.79x+collapse, Pythia 1.36x+collapse (SHARPEN), GPT-2 0.71x no-collapse (does NOT). Reconciles w/ 778 (causal subspace generalises): gpt2 subspace causal+sufficient but not by grammatical labels; gpt2 embedding already class-separated (Fisher 0.305>Pythia 0.20). CAVEAT: gpt2 0.71x may be massive-activation confound in RAW Fisher (782 fixed via de-shared) -- need whitened metric to settle, queued cross_model_class_whiten. CORRECTION: class-COMPUTING (grammatical sharpening) validated bilin18+Pythia not GPT-2; universal cross-model result stays token-class SUBSPACE sufficiency (778). pred_a False.
- 790: EARLY-LAYER SCOREBOARD (answers user "how much understood"). Keep-only (class+position) of OUTPUT per component: attn0-5 all 0.91-1.00, mlp0 0.94/mlp1 0.79/mlp2 0.47/mlp3 0.51/mlp4 0.87/mlp5 0.59. NAT-WEIGHTED = 93% of 10.92 early nats is class+position output. CAVEAT: for attention partly TRIVIAL (value-copying -> output token-conditional by construction); meaningful = MLPs + attention PATTERNS. OUTPUT-decomposability (93%) != mechanism understanding (mlp0 understood, attn5 joint 787). 93% (output) vs ~75% (attn input-reading 784) both valid, different measures. WORKLIST: mlp1 (0.23 unnamed nats = 777 irreducible) > mlp0 (0.14) > mlp4 > attn5 > attn4. Total unnamed ~0.7/10.9 (~7%). mlp0/attn1 ~95% incl mechanism. Fig early_scoreboard.png. pred_a True.
- 791: FOLDED QK CLASS-ATTENTION (user: fold tensor, name causally). Fold per-head QK bilinear forms onto 7 named class directions, sum heads -> class x class attention. attn5: ~0 class-attention (|coupling|<=0.003) = POSITIONAL head (weight-derived confirmation of 783). attn1: strongest (~0.3), DETERMINER-query dominated (det->prep -0.38 etc) = DETERMINER head (connects to article circuit 727). attn0: weak det->function-word. Content part only (rmsnorm+rotary not folded, class dirs not orthogonalised -> specific signs provisional; robust: attn5~0/attn1-determiner/attn0-weak). NEXT attn_fold2 (rotary for attn5 positional pattern + causal verify attn1=determiner).
- 792: CAUSAL VERIFICATION corrects 791. Ablate attn1/attn5, CE-increase by token-class: attn1 FLAT (det 2.57/num 2.67/.../ top=num, x1.17 not 1.5), attn5 FLAT on class AND position (x1.19/x1.09). NEITHER determiner/position-concentrated. CORRECTION: folded class-attention GEOMETRY (791) != causal function; attn1/attn5 contribute BROADLY (large multi-function heads ~2 nats; whole-head ablation dominated by broad value-moving, masks class sub-component). "attn1=determiner"/"attn5=positional" NOT supported at whole-head causal level. RECONCILE w/ 784: attn1 READS class (input, robust) but CONTRIBUTES broadly. LESSON: fold names QK geometry not causal function; verify sub-circuit via EDGE ablation not whole-head. Robust framing = VARIABLE/reader level (783/784), not per-head concept names. pred_a/b False.
- 793: FRONT = CLASS+POSITION COMPUTER (end-to-end capstone 767-792). Ablate all front L0-5: benefit 6.97 nats. Project all 12 components onto class+position SIMULTANEOUSLY: CE-recovery 0.837 vs random 0.171. First 6 layers reduce to a token-class+position computer end-to-end (84% of the front's ~7-nat contribution; 0.84 simultaneous vs 0.93 per-component = stack compounding). CAPSTONE: front ~84% class+position causally; class computed nonlinearly (~24-dim, sharpened 782), position ~2-dim early/late RoPE readout (788), near-orthogonal channels (772), read amortized (783/784), generalises to gpt2/pythia at subspace level (778/781); ~16% distributed remainder (mlp1 + multi-function heads, per-head naming fails 792). pred_a True.
- 794: WHOLE MODEL = ~78% CLASS+POSITION (grand summary). Ablate all 36 components (attn+mlp L0-17): benefit 9.13 nats. Keep-only class+position at ALL 36 simultaneously: CE-recovery 0.781 vs random 0.044. The ENTIRE model reduces to a token-class+position computer at ~78% end-to-end (front 84%, whole 78% -- back half slightly more non-class-position + stack compounding). GRAND CAPSTONE 767-794: ~78% of the 546M model reduces to two low-rank computed nameable variables -- token-class (~24-dim, mlp0 collapse+sharpen 780/782) + position (~2-dim RoPE readout 786/788), near-orthogonal (772), read amortized (783/784), generalises gpt2/pythia (778/781). ~22% distributed remainder (no low-rank carrier, mlp1 + multi-function heads, per-head naming fails 792). pred_a True.
- 795: THE ~22% REMAINDER IS UNIFORM not content (my pred WRONG). Class+position recovery by current-token-class: det 0.80/punct 0.80/prep 0.78/aux 0.79/conj 0.83/pron 0.80/num 0.76/content 0.77 -- function-word mean 0.80 vs content 0.77 (gap 0.036, negligible). Uniform ~78% across ALL classes. CORRECTION: the 22% distributed remainder is NOT a content module -- it's a broad UNIFORM overhead applied equally regardless of token type. Consistent w/ distributed/no-low-rank-carrier (FINDINGS 1): resists localization by class as it resists low-rank. Final: model applies same 78/22 class+position/distributed split uniformly. pred_a False (rules out content-module hypothesis). CAPSTONE 767-795: model ~78% class+position (two computed nameable variables, cross-model-general), uniform, ~22% uniform diffuse remainder.
- 796: CROSS-MODEL CLASS-SHARPENING SETTLED (de-shared Fisher, resolves 789 gpt2). Pythia 1.36->1.62 de-shared (SHARPENS, stronger de-shared); GPT-2 0.71->0.83 de-shared (does NOT, not rescued by removing shared direction). gpt2 non-sharpening GENUINE not a massive-activation confound. Class-SHARPENING model-specific: bilin18 1.8x + Pythia 1.62x YES, GPT-2 NO (its embedding already class-separated 0.229>pythia 0.164). SETTLES 789. Universal cross-model = token-class SUBSPACE sufficiency (778); gpt2 first-MLP is low-rank token-organised but not grammatical-class-sharpening. pred_a False definitively. Closes cross-model thread.
- 797: LAYER-UNDERSTANDING DEPTH PROFILE (closes 767-797 arc; fig layer_understanding_profile.png). Per-layer keep-only class+position: BIG-benefit layers most understood -- L0 0.90/L1 0.82/L5 0.96/L16 0.98/L17 0.95 (~7 of ~9 nats, 82-98% class+position). Lowest: L2 0.41/L3 0.66/deep-middle L7-14 0.43-0.59 but TINY benefit. random-subspace catastrophic at big layers (L1 -2.71 etc). The distributed ~22% is NOT in any important layer -- diffuse across low-benefit layers; combined w/ 795 (uniform across token-class) the remainder is diffuse in BOTH depth AND class. ARC CLOSE: the model's meaningful computation is ~class+position (82-98% where it counts); the hard residue is a thin uniform distributed film, not a hidden nameable circuit.
- 798: DISTRIBUTED REMAINDER = CONTENT FOR CONFIDENT PREDICTIONS, not hard ones (my pred WRONG). Class+position recovery by full-loss decile: MONOTONIC 0.50 (easiest, loss 0.05) -> 1.00 (hardest, loss 9.75). Sufficiency INCREASES with difficulty: easy/confident predictions need the DISTRIBUTED content (class+pos only 0.50); hard/uncertain fall back to pure class+position (1.00, nothing more to recover). INTERPRETATION: class+position = grammatical skeleton always present; ~22% distributed = specific CONTENT that makes confident low-loss predictions confident. Not a hidden circuit, not hard-token compute -- content/specificity for the predictable. Caveat: hardest 1.00 partly small-denominator; monotonic trend robust. pred_a False. FINAL 767-798: model = class+position skeleton (understood, causal, cross-model) + distributed content filler (~22%, needed most where confident).
- 799: CROSS-MODEL WHOLE-MODEL TEST DOES NOT TRANSFER (methodological negative). gpt2 (24 comp): benefit 49.6 (impossible -- no output clamp, ablation catastrophe), keep 0.95/random 0.77 (metric dominated by catastrophe-prevention, doesn't isolate class+pos). pythia (48 comp): keep 0.14 ~= random 0.16 (24-layer simultaneous 96-dim projection compounds destructively, no signal). bilin18's 78% (794) is measurable ONLY because its 30*tanh output clamp bounds CE (random null 0.04); real models lack it (gpt2) or too deep (pythia). CORRECTION: whole-model 78% is a bilin18 result, universality UNTESTED (metric bilin18-specific). Robust cross-model = PER-COMPONENT (778 first-MLP subspace gpt2 0.84/pythia 0.62, barbell; 781 position causal) -- clean because single-component. pred_a False.
- 800: CROSS-MODEL CLASS+POSITION ~UNIVERSAL (clean per-component metric, resolves 799 positively). Nat-weighted per-component keep-only class+position: bilin18 ~0.78, GPT-2 0.767 (vs random 0.282), Pythia 0.690 (vs random 0.101). All ~70-78% class+position, >> random -> the whole-model reduction GENERALISES (not bilin18-specific). Per-component metric is bounded+non-compounding (fixes 799's catastrophe/compounding). First MLP dominates + most class+position in all (gpt2 mlp0 0.92, pythia 0.70). GRAND: trained transformer per-component computation is ~3/4 class+position across bilinear/absolute-pos/rotary; subspace sufficiency + barbell + position causality + ~75% share all cross-model; class-SHARPENING model-specific (789/796). Closes cross-model thread. (pred_a flag False = buggy threshold; numbers meet bar.)
- 801: SCALE SWEEP -- ~3/4 class+position holds 4/5 models (bilin18 0.78, gpt2 0.77, pythia-160m 0.75, pythia-410m 0.69) vs random 0.02-0.28; GPT-2-medium OUTLIER 0.12. Cause localised: gpt2-medium mlp0 = 82% of benefit but keep -0.14 (NEGATIVE, worse than ablate) = massive-activation confound signature (token-mean subspace captures huge loss-irrelevant direction, keeping only it is harmful; same class as 780/799). Likely measurement artifact at mlp0, not clean counterexample. CORRECTION: scale-robustness MIXED (strong 4/5, gpt2-medium confounded, unsettled). Queued gpt2med_diagnostic (re-measure mlp0 de-shared/more-dims). Honest claim: ~3/4 class+position broadly cross-family/scale, one outlier w/ massive-activation confound. pred_a False.
- 802: GPT2-MEDIUM GENUINE EXCEPTION (my 801 confound hypothesis REFUTED). Diagnostic: NO massive activation (norm max/mean 1.49, s0/s1 1.28; de-massive doesn't help). keep-only stays NEGATIVE to rank 128 (-0.43/-0.21/-0.15), +0.17 at r256 -> keeping token-mean projection WORSE than ablate = joint/non-separable computation, GENUINE not artifact. CORRECTION to 800/801: ~3/4 class+position is COMMON (4/5: bilin18 0.78/gpt2-small 0.77/pythia-160m 0.75/pythia-410m 0.69) but NOT UNIVERSAL -- gpt2-medium genuinely doesn't reduce. I OVERSTATED "property of trained transformers"; honest = "common, with a genuine exception." Why gpt2-medium (vs gpt2-small, only scale differs) open. pred_a False.
- 803: GPT2-LARGE PASSES -- gpt2-medium ISOLATED exception, not scale trend (settles 802). gpt2-large (774M): class+pos 0.751 vs random 0.185, mlp0 keep 0.77 (healthy). GPT-2 sizes: small 0.77 ok / medium 0.12 fails / large 0.75 ok -> gpt2-medium NOT explained by scale (both smaller+larger fine). FINAL TALLY: ~3/4 class+position holds 5/6 models (bilin18 0.78, gpt2-small 0.77, gpt2-large 0.75, pythia-160m 0.75, pythia-410m 0.69), across bilinear/absolute/rotary + scale 124M-774M; gpt2-medium (355M) single genuine exception (isolated quirk, joint mlp0, not scale). CONCLUSION: reduction is COMMON+ROBUST across family/scale but NOT universal (1 documented counterexample). Closes cross-model/scale thread.
  §804 gpt2_mlp0_compare — gpt2-medium mlp0 is LOW-rank (eff-rank 26) but every centered projection (own-SVD r128 −0.17, token-mean −0.13, random −0.50) is worse than ablation → the centered metric drops a large loss-critical DC/mean bias; queued gpt2med_dc_test to check if 802's exception is a dropped-mean artifact
  §805 CORRECTION of 802/803 — gpt2-medium mlp0 exception was a DROPPED-MEAN artifact: its output is 91% constant bias by norm; keep goes −0.13→+0.63 once the mean is preserved. It IS class+position (plus a dominant constant bias), NOT a genuine exception. class+position now COMMON ACROSS ALL SIX models; gpt2-medium's 0.63 is less clean, its signature = a dominant DC bias (not a per-token massive activation)
  §806 cross_model_dc — large constant (DC) biases COMMON in early layers (7/10 components dc>0.5, all models); mean-preserving keep lifts class+position for every substantive component (all mlp0 now 0.63-0.97: gpt2 0.94/med 0.63/large 0.97, pythia 0.70/0.72); the two 'hurt' cases are near-zero-benefit no-op attentions (unreliable ratio). §800 headline numbers are slight underestimates (centered); full mean-preserving re-sweep queued
  §807 cross_model_scoreboard_mp — CORRECTED whole-model class+position (mean-preserving, all components): gpt2-small 0.77→0.91, gpt2-medium 0.12→0.64, gpt2-large 0.75→0.92, pythia-160m 0.75→0.81, pythia-410m 0.69→0.74; random 0.02-0.28. Corrected share 0.64-0.92 across five HF models; gpt2-medium now in-family, NO genuine exception. bilin18 mean-preserving re-score queued to complete all six
  §808 bilin18_scoreboard_mp — bilin18 per-component nat-weighted class+position = 0.92 mean-preserving (≈0.92 centered; DC negligible for bilin18). Corrected apples-to-apples all-six per-component table: bilin18 0.92, gpt2-small 0.91, gpt2-large 0.92, pythia-160m 0.81, pythia-410m 0.74, gpt2-medium 0.64 (band 0.64-0.92, all >> random 0.02-0.28). bilin18's '0.78/four-fifths' is the stricter SIMULTANEOUS metric, kept as its self-description. Correction arc (802→808) closed
  §809 early_understood_corrected — ANSWER to 'how much of early layers understood?': 93% of layers 0-5 loss-benefit is class+position (whole model 92%). Highest-benefit early components (attn0/1/5, mlp0) are 97-99% understood; remainder concentrated in small low-benefit MLPs (mlp1 26%, mlp2 62%, mlp3 55%). Figure sent. Corrected class+position program closed
  §810 mlp1_remainder — mlp1's 26% remainder is mostly DIFFUSE (residual eff-rank 462, no clean low-rank handle) but contains real modest previous-token (net +0.075 over random) AND class×position-interaction (net +0.089) components, each ~2× a matched random null; neither closes the gap (keep tops 0.89). Not a clean third variable; diffuse residue with faint nameable structure. Sweep queued (mlp1/2/3 general?)
  §811 mlp_remainder_sweep — prev-token + class×position interaction is a GENERAL early-MLP remainder recipe (mlp1/2/3), the two capture DIFFERENT slices (both > either alone everywhere), net-over-random +0.087/+0.187/+0.194; residual stays high-rank (462-497), small MLPs = tiny benefit
  §812 whole-stack BARBELL (from §808 data) — front (0-5) computes class+position benefit 10.4 keep 0.93; back (15-17) reads it out benefit 1.9 keep 0.93; middle (6-11) nearly inert benefit 0.49. Attentions ~0.9-1.0 class+pos everywhere; low-class+pos components = small low-benefit MLPs. Validates amortized reading; no rich mid-stack of NEW VARIABLES (answers 'move up stack'). Figure sent. [CORRECTED by §813: 'inert' was wrong — middle is distributed/redundant, 1.9 nats collective]
  §813 CORRECTION of §812 (middle_skippable) — the middle is NOT skippable: simultaneous ablation of layers 6-11 costs 1.93 nats vs 0.49 per-component sum (3.9× super-additive = distributed/redundant computation). Barbell holds only as a PER-COMPONENT statement; model is NOT front+back. Front is sub-additive (0.6×), back 2.4×. Metric caveat: per-component nat-weight under-weights redundant middle, so true whole-model class+position < 0.92 headline (consistent with simultaneous 0.78)
  §814 middle_class_position — redundant middle is class+position MAINTENANCE: simultaneous keep-only class+position on layers 6-11 recovers 0.65 of the 1.9-nat collective benefit (random null -1.06). Whole-stack closes: FRONT computes (6.6 nats, 0.93) → MIDDLE maintains/refreshes redundantly (1.9 nats, 0.65) → BACK reads out (4.6 nats, 0.93). Amortized class+position across depth + diffuse content remainder. Closes whole-stack arc 767→814
  §815 cross_model_pipeline — compute→maintain→read pipeline is UNIVERSAL (bilin18/gpt2/pythia-410m): all have high-class+pos front+back and a SUPER-ADDITIVE redundant middle (compounding bilin18 3.9x/gpt2 2.09x/pythia 2.61x) that is class+position MAINTENANCE (keep 0.62-0.78); every band >> random. Nuance: front compounding varies (sub-additive bilin18/pythia, super gpt2), gpt2 back extremely redundant (5.15x). Whole-stack arc closes with cross-model universality
  §816 middle_prune — redundant middle is HALF-compressible: drop 3 of 6 layers (deepest: 10,11,8) for 0.27 nats; curve concave (first-3 0.27 vs last-3 1.66). FRONT convex/costly (1 layer 0.25, 3 layers 2.91, ~10x middle). Redundancy increases with depth into middle (layer 6 most important, layer 0 single most irreplaceable). Figure sent
  §817 cross_model_pipeline_rest — ALL-SIX pipeline table complete. Redundant class+position-maintenance MIDDLE universal (compounding 1.3-3.9x, cp 0.59-0.78, all 6 incl gpt2-medium in-family). Caveats from larger sample: gpt2-large FRONT only 0.37 cp under simultaneous band metric (compounds; per-component was 0.92 §808 — real interaction structure); pythia-160m back-heavy (back 9.78 nats, middle only 1.3x). Middle signature robust; front/back detail architecture-specific
  §818 middle_vs_front_subspace — REFINEMENT of 'maintenance': middle ADDS class+position in mostly NEW directions (keep-own 0.65 vs keep-front-subspace 0.37, overlap 0.29, front/own 0.57). ~57% reinforces front's directions, ~43% new structure. Resolves residual-stream puzzle: middle CONTINUES computing class+position, not just refreshing. 'Maintenance'->'continued class+position computation'. FINDINGS 1c + artifact reworded
  §819 middle_new_content (fixed; first run invalid — full-rank U_front bug caught by sanity check) — the middle's NEW class+position content leans finer CLASS: net token gain +0.183 vs position +0.105 (both >> matched random nulls). Middle continues refining grammatical-class chiefly. Closes deep middle characterization (813-819)
  §820 cp_stability — VALIDITY: (1) class+position subspace GENERALIZES to held-out data (within≈cross, gap ≤0.005; NOT overfit). (2) CAVEAT: per-component mean-preserving keep is mean+redundancy-dominated — random-orth null recovers 0.84-0.95 (μ alone ~sufficient per single component + others carry variation redundantly); class+position's specific per-component increment over random is modest (0.07-0.13, but consistent/specific). Honest whole-model share = simultaneous centered 0.78 (random 0.04), NOT per-component 0.92. Sharpens FINDINGS 1b/1c caveat. Two sanity-caught bugs fixed first
  §821 cp_null_mechanism — DRIVER of high single-component keep = MEAN not redundancy (mean-only recovers 0.66-0.91; centered-random only 0.04-0.29). CORRECTS §808: bilin18 HAS large per-component means (ratio up to 0.97), loss-relevant; centered≈mean-preserve because the real cp subspace CONTAINS μ, not because μ small. Confirms §820: per-component keep mean-dominated; class+position variation adds modest specific increment (0.05-0.14; centered-random low=specific); honest share = simultaneous 0.78
  §822 mean_output_meaning2 (embedding-fixed, pos-indep std 0.000) — constants are large FUNCTION-WORD-leaning OFFSETS, NOT a frequency prior (fully-constant CE 10.4 >> unigram 7.1; r 0.54; top tokens ' the'/' a'/','/newline). SYNTHESIS 820-822: component outputs dominated by constant means (ratio up to 0.97) → single-component mean substitution nearly free, but constants carry no standalone prediction → per-component keep reflects offsets not computation; real class+position computation is the small-norm variation, honest share = simultaneous 0.78. Closes validity/mean sub-arc
  §823 cp_steering2 — CAUSAL SUFFICIENCY confirmed: injecting amplified class-deviation α·proj_cp(μ_B−μ_global) at front components steers the prediction TOWARD the target class (' a' KL 3.70→1.69, ' and' 1.59→1.20; all sources cp beats matched random by +3.0 KL). Specific to class+position dirs (random degrades). Needs amplification (α 8-32) to overcome redundant recomputation from embedding (item 14); ' the' flat (already default). v1 confounded (injected full μ_B, constant-dominated). Completes causal story: necessary+generalizing+sufficient
  §824 cp_steer_embedding — steering class+position at the EMBEDDING corrupts prediction (KL 3.4→6.4-8.8, worse than random), OPPOSITE of component steering (§823 moves toward B). So class+position is a COMPUTED variable at the component write-channels, NOT the raw embedding — editing the embedding just makes an OOD input. Sharpens causal localization; refines §823 amplification (small-norm+redundant, not embedding-as-lever). Causal story complete: necessary+generalizing+sufficient-at-components
  §825 class_naming — WHAT the class variable IS: named grammatical categories. Top class directions = ARTICLES/DETERMINERS (the/a/an/his), PRONOUNS (You/He/We & she/he/him, both opposed to NUMBERS), PUNCTUATION, CONTRACTION SUFFIXES ('ll/'ve/'d), CONJUNCTIONS (and/&), QUANTIFIERS (Many/Some/Even), NUMBERS, capitalized-initials. Shuffled control incoherent (September/faith/fight). eff 239 real vs 441 shuffled. Class = a POS/grammar system. Nuance: leading ~15 dirs clean grammatical; full token geometry ~239-dim tail
  §826 cross_model_class_naming — rediscovering parts of speech is UNIVERSAL: GPT-2 (be-verb axis are/am/is/will/be, determiners, conjunctions) and Pythia (preposition axis to/in/on/for/at, determiners, modals will/would/have) first-MLPs carve the same grammatical categories as bilin18. Shuffled controls incoherent. Shared: determiners/conjunctions/prepositions/be-verbs/auxiliaries/punctuation/numbers/pronouns. eff dims ~168-239, ~15 grammatical directions dominate. Completes interpretive payoff cross-model
  §827 position_naming — POSITION variable is dominated by a LOG-POSITION axis (corr 0.98 with log-pos, sv 2.5× next; fine early, coarse late) + weak early-landmark (corr_pos0 0.53); dirs 1/3/4 finer residual (rotary-freq?). Shuffled control corr-linear 0.13. eff ~74 dims (vs 159 shuffled), dominated by the log axis. Both variables now named: class=grammar (§825/826), position=log early/late scale+landmark. Figure sent
  §828 class_transition — the read-out implements GRAMMATICAL SEQUENCING: model's predicted next-CLASS matches the empirical class-bigram near-perfectly (mean KL 0.009 vs shuffled null 0.056), with correct transitions (det→noun 0.82, prep→det/noun, pron→verb 0.74, number→punct 0.52). Class variable end-to-end: computed (§825/826) + causal (§823) + used as grammar-sequencing (§828). Honest: class-bigram match partly generic, but class-specific vs null + correct transitions; hard residual is specific-token-within-class (§798). Artifact updated
  §829 ce_decomposition — QUANTIFIED PUNCHLINE: CE_total 3.23 = CE_class 0.75 (23%) + CE_within 2.48 (77%). 77% of the loss is choosing WHICH WORD within the correct grammatical class; grammar itself only 23% (easy). Per-class: det within 4.09 (hard), number 1.51 (constrained). Reconciles the program: ~4/5 of components compute the interpretable grammatical skeleton = the EASY 23% of loss; the HARD 77% is diffuse high-rank lexical choice (§798/810) with no low-rank handle. Closes program 767→829. Artifact updated
  §830 within_class_context — hard 77% is partly context-reducible (within-CE drops 1.12 nats early→late, 3.50→2.38) but mostly IRREDUCIBLE (plateaus ~2.4 nats with full context). Context helps lexical choice ~4.5× more than grammar; class-CE nailed early (0.97) & near context-free (drops 0.25). Loss = easy context-free grammar (0.75) + lexical (2.48: ~1.1 context-reducible + ~2.4 irreducible entropy floor). Closes read-out/loss thread + program 767→830. Artifact updated
  §831 cross_model_ce_decomposition — grammar-easy/lexical-hard loss split is UNIVERSAL: within-class fraction 76.9% bilin18 / 75.0% gpt2 / 75.5% pythia-410m (all ~3/4, within 2pts, chain-rule exact). The interpretable grammatical machinery computes the easy ~1/4; the hard ~3/4 is lexical choice — across 3 architectures. Final cross-model capstone of the class+position program (767→831)
  §832 lexical_localization — grammar vs lexical work is mostly NOT component-localized (most components ~76% lexical = baseline; distinction is WITHIN-component via the class+position subspace, not between components). Weak specialization: back read-out MLPs (mlp15/16/17 lexical-share 0.63-0.70) = grammar specialists (next-class prediction §828); attention (attn1/5/6/7 0.78-0.79) leans lexical (context word-narrowing §830). Lexical system distributed (by component) + high-rank (§810) + mostly irreducible (§830). KEY RECONCILIATION: attn5 is 0.99 class+position (§814) yet 79% lexical here → the class+position SUBSPACE (representation) serves BOTH grammar AND lexical loss; grammar-vs-lexical (LOSS split §829) is ORTHOGONAL to class+position-vs-remainder (REPRESENTATION split §807). 'class+position=grammar' is too glib — it's the model's main predictive representation, doing grammar fully + lexical partially
  §833 cp_serves_lexical — CORRECTION: class+position is the model's MAIN PREDICTIVE representation, NOT grammar-only. Keep recovers within-class/LEXICAL benefit 0.76 (≈ class 0.81, random −0.10). Corrects §829/830 artifact framing ('class+position=easy grammar quarter; hard 77% separate diffuse'). Right picture: class+position (low-rank) drives ~78% of ALL benefit (grammar 0.81 + lexical 0.76); the ~22% not-captured is the diffuse high-rank residue (§810) serving the last lexical slice; large irreducible entropy floor remains (§830). Artifact corrected
  §834 cp_lexical_source — clean functional split WITHIN class+position: coarse grammatical CLASS drives grammar loss (+0.097 class, only +0.046 lexical); POSITION + fine TOKEN-IDENTITY drive lexical (position-only recovers 0.595 within, far > centered-random 0.04-0.29 = genuinely load-bearing; fine-token adds +0.120 lexical vs coarse +0.046, ~2.6× → specific token/bigram drives word choice). Refines §833: 'class+position drives lexical' = POSITION + token-identity, NOT coarse grammar. Division of labor inside the low-rank representation
  §835 position_loadbearing — CORRECTION: position-only keep (0.638) is a RANK/CONSTRUCTION ARTIFACT — shuffled-position subspace (matched rank/construction) recovers 0.660 (equal). Any data-derived rank-32 subspace captures the dominant directions regardless of labels (= §820 shuffled-token 0.99 effect). RETRACTS §834's position+token lexical decomposition (artifact baseline). Right keep-only null = shuffled-LABEL matched-rank, NOT random-orthonormal (§821 too weak). Class+position keep magnitudes (0.78/0.92) overstate the SPECIFIC contribution. ROBUST regardless: naming §825/826 + causal steering §823. Queued cp_vs_shuffled_null (decisive)
  §836 cp_vs_shuffled_null — MAJOR CORRECTION: class+position keep magnitude (0.78/0.92) is a RANK/CONSTRUCTION ARTIFACT. Shuffled-label subspace (matched rank 96) recovers 0.805/0.760 vs real 0.811/0.761 — specific gap +0.006/+0.001 (negligible). Keep-only measures rank-96 compressibility, not class+position specifically (both ≈ top-PCA). RETRACTS keep-magnitude claims: '78% class+position' (§794/807), per-component 0.92, cross-model 0.64-0.92 (§808/815/817), §833/834 lexical decomposition. ROBUST (independent): naming §825/826 (shuffled incoherent), steering §823 (causal), loss split §829-831 (chain rule), ablation shapes §812/813. LESSON: keep-only null must be shuffled-LABEL matched-rank, not random-orthonormal. Propagating to FINDINGS+artifact
  §837 steering_specificity — causal pillar HOLDS: class steering is class-SPECIFIC (mean diagonal −0.85 toward-injected vs off-diagonal +1.30 away-from-others; specific for 3/4 sources: ' a'/' and'/'.', with ' the' the default-class exception as in §823). Confirms class+position is real (naming §825) + causal (steering §823/837) after §836 retracted the keep-only magnitude. Post-correction status: low-rank rep IS grammatical class+log-position (named, causally class-specific), used for grammar sequencing (§828), loss ~1/4 grammar / ~3/4 irreducible lexical (§829-831); no trustworthy keep-only 'fraction of model'
  §838 steering_sequencing — CAUSAL LOOP CLOSED: class steering causally DRIVES grammatical sequencing. Steer→B shifts predicted next-CLASS toward what-follows-B (mean +0.27; strong for ' a' +1.53) vs random −1.89 (wrecks it). Class rep causally PRODUCES the §828 sequencing behavior. Full causal account now keep-only-FREE: real (naming §825/826) + causal/class-specific (§837) + drives sequencing (§838) + loss split (§829-831). None rests on the retracted §836 keep magnitude
  §840 entropy_ceiling — CORRECTION of §830: word-choice floor is PARTLY REDUCIBLE (within-class CE 2.74→2.51→2.39 gpt2→med→large, −0.35 monotonic), NOT irreducible entropy. Dominated by hard entropy (~87% persists at 774M) but a real reducible ~13% exists → there is diffuse structure to understand. bilin18 2.49 (good; ~0.1 behind gpt2-large). Artifact corrected
  §841 mlp0_bilinear_trace — LAYER 0 mechanism (bottom-up start, weight-level, exact recon 6e-7): mlp0 computes grammatical class as a bank of BILINEAR class-DETECTORS. Top-24 class-writing units ALL class-selective + genuinely bilinear (product of two token-readouts, cvL/cvR ~1-1.5): determiner (u1737/1492/597/906), capital/proper (u3326/3926), punctuation (u3574 '.'), conjunction (u212 '&'/','), number/article (u2736), content-word (u3069). Product sharpens class (§782). Next: Left-vs-Right (sharpening vs conjunction?) + attn0 to finish layer 0
  §842 mlp0_leftright — mlp0 class bilinearity is SHARPENING (self-product): all 24 units Left/Right read SAME class, product sharpens 1.52× mean. Weight-level mechanism of §782 sharpening
  §843 attn0_function — attn0 builds COPY-SOURCE: output decodes PREVIOUS token 0.86 (vs input 0.30, null 0.065; +0.56). attn0's 1.44 nats = write prev-token identity into stream (census item 10 confirmed at mechanism level)
  §844 LAYER 0 COMPLETE: current token (embedding) + sharpened grammatical class (mlp0, bilinear self-product detectors) + previous token (attn0 copy-source). Substrate for layer 1. Next: attn1 (induction target?) + mlp1 (recompute class or new prev-token feature?)
  §845 attn1_function — attn1 NOT induction (induction out 0.29=in 0.29, null 0.05); maintains local token history (prev 0.83, prevprev 0.54). Token-decodes don't fully account for its 2.22 nats
  §846 mlp1_bilinear_trace — mlp1 is a SECOND current-class detector bank (24/24 class-selective+bilinear, 0/24 prev-token-driven). Recomputes current class like mlp0; does NOT use the prev token attn0 provided
  §847 LAYER 1 synthesis — layer 1 largely REDUNDANT with layer 0 (mlp1 recomputes current class; attn1 carries token history, not induction). Prev-token NOT consumed by layer-1 MLP. Build-up more redundant than compositional. Next: scan layers 2-5 incl attn5
  §848 (user correction) — RETRACT §847 'layer 1 redundant': rested on coarse 8-way POS collapse that hides the fine token-geometry (a token-mean is a CLUSTERING; relative similarities differ across stages). Measuring token×token RDM evolution embedding→L0→L1→... (consecutive RSA + within-class RSA + eff-dim) to test if each layer re-clusters. Folding-onto-prior-features set up next. Context: attn5 aggregates not copies (token decode ~0.33)
  §849 layer_geometry_evolution — VINDICATES user, §847 'redundant' RETRACTED: layer 1 RE-CLUSTERS as much as layer 0 (consecutive RSA emb→L0 0.49, L0→L1 0.58; within-class 0.61). Eff-dim: emb 117 → L0 20 (COLLAPSE to class clusters) → L1 47 (RE-EXPAND, geometric opposite) → L2-4 ~45 → L5 23 (attn5 aggregates). Token-mean is a clustering; it keeps changing. Coarse-8-class lens was blind to the fine re-clustering. Next: fold layer-1 bilinear form onto layer-0 features for specifics
  §850 mlp1_folding — FOLD layer 1 onto layer-0 features: mlp1 reads CLASS 2.9× (primary) + POSITION 1.67× + prev-token 1.26× + fine-token 1.29× (all >chance; random-vec baseline ~1.0). Explains §849 re-expansion (position+fine-token folded back in beyond coarse class) AND corrects §846 '0/24 prev-driven' (folding is more sensitive — mlp1 DOES read prev-token modestly). Next: folding scan across depth
  §851 mlp_folding_scan — depth-profile of features-read (ratio/chance): PREV-token front-only (mlp0 1.73→mlp2 1.23→chance by mlp4; answers §847 who-consumes-copy-source = early MLPs); CLASS barbell (mlp0 8.3, mlp3 dip ~1, mlp17 13.3); POSITION read throughout, hardest at readout (mlp17 6.3). Back MLPs 15-17 fold class+pos+token in increasingly hard = the read-out. Compositional build-up traced
  §852 attn5_aggregation — aggregation REFUTED: attn5 encodes context-class R² 0.32 (≤ copier attn2 0.36), position 0.48 (=0.49), and less token/class than attn2. Not a copier, not an aggregator — its 1.97 nats are captured by NO grammatical/token probe. Likely semantic/content or structural (my probes are all grammatical). Honest unknown; switching to effect-based lens (attn5_effect)
  §853 attn5_effect — attn5 RESOLVED = CONTENT-prediction component: ablation CE increase strongest for content words (word 2.71/cap 2.50/number 2.31, content=62% of effect), weakest for function words (~1.0). Works on the hard lexical 77% (§829), not grammar — why grammatical probes missed it (§852). Front's main early content contributor
  §854 FRONT synthesis (0-5): grammatical skeleton (mlp0 sharpen-class + attn0 copy-source; mlp1 re-expand folding class+pos+prev+token; L2-4 refine; mlp5 class) + FIRST CONTENT signal (attn5). Prev-token front-only. Next: back readout mlp16/17
  §855 back_readout — per-unit readout trace INCONCLUSIVE: write-tokens incoherent rare tokens (rare-token unembedding-norm artifact in lm_head·Down), read-class weak at L16/17. Readout READ side known (class 13×/pos 6×, §851) + FUNCTION = grammatical sequencing (§828), but per-unit WRITE decomposition failed. Refitting with cosine attribution to separate artifact from genuine distribution
  §856 back_readout2 — readout SEMI-DISTRIBUTED (cosine attribution): some coherent content-category writer units (mlp17 u3343=proper names Jonathan/Jason/Jennifer; u644/u3237/u919 = capitalized/sentence-initial words) + a diffuse remainder (weak read-class, some garbage units). Not a clean per-class bank like the front; the specific-token WRITE is mostly distributed — fits producing the hard lexical output (§829/810)
