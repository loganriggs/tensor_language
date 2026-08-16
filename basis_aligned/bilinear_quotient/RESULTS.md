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
