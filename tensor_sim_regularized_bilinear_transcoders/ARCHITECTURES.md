# Hierarchical architectures — exact shapes, code, and the swept parameter

Three *different* things have been called "hierarchical" in this program. They are genuinely different
architectures with different swept parameters, so they are separated here. Every one of them is fit
**data-free** (loss = closed-form `L_fid` only; no forward passes, no data).

Notation, fixed throughout:

```
x̃ = (1, x) ∈ R^d          lifted input,  d = d_in + 1        (the 1 is the bias/constant coord)
h = (L x̃) ⊙ (R x̃) ∈ R^r   bilinear hidden ("features")
y = D h        ∈ R^K       output
        L, R : (r, d)      D : (K, r)          y_k = Σ_ij A_kij x̃_i x̃_j ,  A_kij = Σ_h D_kh L_hi R_hj
```

A "prime" (`L′, r′, …`) is always the **transcoder** being fit; unprimed is the **target** being explained.
The loss is always `L_fid = ‖A − Â‖²_Λ / ‖A‖²_Λ`, closed-form in the CP factors (`tensor_sim.py`).

---

## A1 — Block-hierarchical **single** layer (`e2_structural_priors.py`, `e3_hierarchy_spectrum.py`)

**Idea.** Partition the `d_in` input coordinates into `n_blocks` groups. Each hidden unit is assigned to one
block and may only read coordinates from that block. Information cannot cross blocks — a one-level hierarchy.

**Shapes** (`D_IN=16`, so `d=17`; transcoder rank `r′=32`; `K=12`):

| tensor | shape |
|---|---|
| `Lt`, `Rt` (transcoder factors) | `(32, 17)` |
| `Dt` | `(12, 32)` |
| `cb` coord → block | `(16,)` |
| `ub` hidden unit → block | `(32,)` |
| `hard` / `offblk` masks | `(32, 17)` |

**The mask.** Column 0 (the constant coord) is always readable — otherwise no unit could express a bias.

```python
cb    = torch.arange(D_IN) % n_blocks                     # (16,)  coord  -> block
ub    = torch.randint(0, n_blocks, (r_tc,))               # (32,)  unit   -> block
inblk = (cb[None, :] == ub[:, None]).float()              # (32, 16)
hard   = torch.cat([torch.ones(r_tc, 1), inblk], 1)       # (32, 17)  1 = may read
offblk = torch.cat([torch.zeros(r_tc, 1), 1 - inblk], 1)  # (32, 17)  1 = forbidden
```

**Two ways to impose it — and they behave completely differently (FINDING 5):**

```python
# HARD  — a structural CONSTRAINT: off-block weights are identically zero (masked in the forward pass)
Le, Re = Lt * hard, Rt * hard
loss   = fid_loss_mean(Dg, Lg, Rg, Dt, Le, Re, Sig, mu, aa=aa)

# SOFT  — a graded PENALTY: off-block weights are merely discouraged with strength s
Le, Re = Lt, Rt                                            # params stay dense
loss   = fid_loss_mean(Dg, Lg, Rg, Dt, Le, Re, Sig, mu, aa=aa) \
       + lam_l1 * (Lt[:, 1:].abs().mean() + Rt[:, 1:].abs().mean()) \
       + s      * ((Lt * offblk).abs().mean() + (Rt * offblk).abs().mean())
```

### ► SWEPT PARAMETER: `n_blocks` ∈ {1, 2, 4, 8, 16}

`n_blocks = 1` is the dense/no-hierarchy baseline; 16 is maximally fine (one coord per block). Crossed with
**hard vs soft** (`s = 0.03`) and with two ground truths (`gt=random` — respects no partition; `gt=block` —
genuinely 4-block). Secondary knob: the soft strength `s`.

**Why it matters.** Hard fidelity *breaks* iff the mask is mis-specified, so the finest `n_blocks` that still
holds `tsim = 1.0` **reads off the layer's true block structure** (4 for the block gt, 1 for the random gt).
Soft never costs fidelity and only improves recovery. **Fit soft; test hard.**

---

## A2 — Cross-layer **tree** (`e4_multilayer_hierarchy.py`) — hierarchy held across depth

**Idea.** Two stacked bilinear layers, with the block confinement enforced at *every* layer including the
routing between them: information entering block *b* stays in block *b* for the whole depth.

```
x (d=8) ──layer 1──> z (dz=4) ──layer 2──> y (K=3)
```

**Shapes** (target `r1=6, r2=4`; transcoder `r1′=12, r2′=8`; `NBLK=2`):

| tensor | shape | masked by |
|---|---|---|
| `L1, R1` | `(r1′, d) = (12, 8)` | `m1` `(12, 8)` — unit reads only its block's **x**-coords |
| `D1` (routing x-features → z) | `(dz, r1′) = (4, 12)` | `mD1` `(4, 12)` — unit writes only to its block's **z**-coords |
| `L2, R2` | `(r2′, dz) = (8, 4)` | `m2` `(8, 4)` — unit reads only its block's **z**-coords |
| `D2` | `(K, r2′) = (3, 8)` | — (unmasked) |

```python
cx, cz = torch.arange(D) % NBLK, torch.arange(DZ) % NBLK   # x-coord -> block ; z-coord -> block
u1 = torch.randint(0, NBLK, (r1,))                         # layer-1 unit -> block
u2 = torch.arange(r2) % NBLK                               # layer-2 unit -> block
m1  = (cx[None, :] == u1[:, None]).float()                 # (r1′, d)   layer-1 factor mask
mD1 = (cz[:, None] == u1[None, :]).float()                 # (dz, r1′)  ROUTING mask — this is what makes it
m2  = (cz[None, :] == u2[:, None]).float()                 # (r2′, dz)  a cross-LAYER tree, not two flat priors

eff = (D1 * mD1, L1 * m1, R1 * m1, D2, L2 * m2, R2 * m2)   # masked forward
loss = deep_fid(target, eff, Sigma, aa=aa)                 # degree-4 metric (tensor_sim_deep.py)
```

The metric here is **degree-4**: two stacked bilinear layers are degree 4 in `x`, so `⟨A|Λ|Â⟩` needs
`E[∏ of 4 quadratic forms]` (set-partition/cumulant expansion, MC-verified).

### ► SWEPT PARAMETER: the *arm* (which structure is imposed) × the *ground-truth kind*

Arms: `dense` / `+L1` / `+cross-layer hierarchy` / `MSE baseline`. GT kinds: `tree` (truly hierarchical) vs
`random` (respects no blocks). The cross-layer prior scores tsim **0.966 on the tree gt but 0.543 on the random
gt** — the FINDING-5 diagnostic replicating at depth.

**Verdict: this is the weaker tool.** It presupposes *which* coordinates group together. A3 does not.

---

## A3 — **Depth bottleneck** (`e5_hierarchy_via_depth.py`, `e5b_hierarchy_spectrum_depth.py`) ← the good one

**Idea.** Don't impose hierarchy at all. A 2-layer bilinear stack is exactly the set of degree-4 maps that
**factor through a bottleneck `z`** — and *that bottleneck is the hierarchy*: layer 1 builds mid-level
features, layer 2 composes them. So sweep the bottleneck width and let the fidelity curve report the structure.

**Architecture — no masks anywhere. The only structural knob is `dz′`.**

```
x (d=12) ──layer 1 (r1′=16)──> z (dz′ = THE SWEPT KNOB) ──layer 2 (r2′=12)──> y (K=3)
```

| tensor | shape |
|---|---|
| `L1, R1` | `(r1′, d) = (16, 12)` |
| `D1` | `(dz′, r1′) = (dz′, 16)` ← **only this shape changes as we sweep** |
| `L2, R2` | `(r2′, dz′) = (12, dz′)` ← **and this one** |
| `D2` | `(K, r2′) = (3, 12)` |

```python
b = [rnd(dz, R1H), rnd(R1H, D), rnd(R1H, D),      # D1 (dz,16), L1 (16,12), R1 (16,12)
     rnd(K, R2H),  rnd(R2H, dz), rnd(R2H, dz)]    # D2 (3,12),  L2 (12,dz), R2 (12,dz)
for _ in range(STEPS):                            # DATA-FREE: no x, no y, only weights
    loss = fid_deep(target, tuple(b), I, aa) + lam * (b[1].abs().mean() + b[2].abs().mean())
    loss.backward(); opt.step(); opt.zero_grad()
```

The target is a genuinely compositional layer: `x∈R^12` = 4 disjoint groups of 3 → each group is squeezed into
**one** mid-level feature → layer 2 mixes the 4 densely. **True width = 4. The transcoder is told nothing.**

### ► SWEPT PARAMETER: `dz′` ∈ {1, 2, 3, 4, 5, 6} — the bottleneck width

`tsim(dz′)` is a **scree plot for mid-level features**: where it saturates = the *effective* number of
sub-features; how sharply it turns = whether they matter equally.

| `dz′` | balanced gt (4 equal) | skewed gt (1, ½, ¼, ⅛) | 1-feature gt (control) |
|---|---|---|---|
| 1 | 0.529 | 0.799 | **1.000** |
| 2 | 0.758 | 0.943 | 1.000 |
| 3 | 0.973 | 0.996 | 1.000 |
| **4** | **1.000** ← knee at the true width | 0.999 | 1.000 |
| 6 | 1.000 | 1.000 | 1.000 |

Secondary knob: `λ_L1`. At `dz′=4`, raising it 0 → 0.03 lifts the **group purity** of the recovered layer-1
features from 0.466 to **0.813** (chance 0.328) at **zero** fidelity cost — so the fit also discovers *which
coordinates form each sub-feature*, never having been told the groups exist.

**A hierarchy is described by two numbers, not one:** its *width* (where the curve saturates) and its
*balance* (how sharply it turns). "Different degrees / spectrum" is literally the shape of this curve.

---

## Every swept parameter in the program, in one place

| symbol | meaning | swept over | experiment | what it revealed |
|---|---|---|---|---|
| `n_blocks` | input-partition granularity (A1) | 1, 2, 4, 8, 16 | E3a | finest hard partition holding tsim=1 ⇒ true block structure (FINDING 5) |
| `s` | soft off-block penalty strength (A1) | 0 (dense), 0.03 | E3a | soft never costs fidelity; improves recovery to 0.986 |
| **`dz′`** | **bottleneck width (A3)** | **1 … 6** | **E5b** | **hierarchy width is a spectrum (FINDING 8)** |
| `t` | metric temperature, `Σ_t=(1−t)Σ_data+tI` | 0, .01, .05, .2, .5, 1 | E3b | a **1% ridge** undoes the blindness (FINDING 6) |
| `r′` | transcoder rank | 8 … 64 (toy); 32 … 1024 (real) | E2, E7 | sim=1 with sparsity from ~2–4× overcompleteness |
| `k` | BatchTopK sparsity | 1, 2, 4, 8, 32 | E6 | — |
| `λ` | weight on `L_fid` next to MSE | 0, .1, 1, 10 | E6 | MSE-only ⇒ tensor-sim ≈ 0; λ=0.1 ⇒ 1.000 at no MSE cost |
| `λ_L1` | factor-row L1 | 0 … 0.03 | E2, E5b | breaks CP non-uniqueness; recovers the planted groups |

**Fitting rule of thumb, across all three architectures: use *soft* penalties to FIT, and *hard* constraints
(a mask, or a narrowed bottleneck) to TEST** — a correct structural hypothesis costs no fidelity, and a wrong
one must break it. That asymmetry is the whole instrument.
