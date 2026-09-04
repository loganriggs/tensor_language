# Frontier: collapse a5's dictionary to a single constant — preregistration

Registered 2026-09-04T08:55Z (`date -u` read in the same tool call that composed this header, per §2858's rule). Before the run.
Immutable; the rung's frozen-hash check refuses to execute if this file changes.

## SIGN CONVENTION — stated first because it is the rule this rung most depends on

Frontier L2 numbers are **CE ADDED ABOVE THE REAL MODEL, so LOWER IS BETTER** (§2135; §312: "+2.6735 beating +2.84/+2.93"). A collapse
**cost** is therefore `L2_F(collapsed) − L2_F(baseline)`, **POSITIVE = WORSE**. §2128/§2129/§2133/§2134 are RETRACTED for reading higher
L2 as better; **§2125 STANDS** — Fisher selection does not install into the §312 frontier, which remains norm-2304 at **2.6735**.

## Why this, and why now

"attn5's write = the price cliff" has stood as one of the three largest gaps in the explained fraction for weeks, and the explained
fraction (5.348% / 10.923% / 4.727 nat / 0 of 68) has not moved all night. §2858–§2870 built and validated a circuit instrument but
touched no frontier quantity by construction. This rung attacks a named largest gap **in the frontier's own currency**.

What is already established about attn5 in the REAL model: its write is one fixed vector — |cos| **.9999996** with its own mean, gain
CV **.081**, rank **1 of 36** for constancy (§2834/§2835) — and substituting that constant costs **.1286** nats of document CE against
the **2.2109** nats that deleting it costs, i.e. **94.2% recovered**. §2830 made it the price cliff: 3rd of 36 in document CE and
**20.4×** disproportionate per unit written.

What is NOT established: whether that survives **inside the §312 construction**, where a5 is not the raw attention but a fitted `attnd`
dictionary — a 10-row class-conditional constant table `CV[c]` selected by an input probe, plus linear maps `LW[k]` for the non-constant
classes. I recorded this as blocked on Codex twice; it is not blocked. §2125's rung 30 script `ops/frontier_fisher8.py` reruns §312's
pipeline and reproduces the published number, and this rung is derived from it. **That file is not modified.**

## Construction

Three arms, each a full rerun of §312's published **norm-selection** pipeline:

| arm | change |
|---|---|
| **BASELINE** | none — the published frontier |
| **a5 collapsed** | `CV` ← ten copies of `Y.mean(0)`; `LW` ← `{}` |
| **a6 collapsed** | the same, applied to a6 — a like-for-like control from the same motif band `ATTM` |

The collapse changes **fitted values only, never control flow**: the hook still computes `cur['lab']` from the input probe, so every
downstream dictionary sees exactly what it saw before. a6 is the control rather than a tail dictionary (a10–a17) so that the comparison
is within one structural family; §2834's census independently ranks a5 **1 of 36** for constancy, so a genuine difference is expected
rather than assumed.

## Predictions, each with its worked-example line

- **pred_a — REPRODUCTION GATE, carried over verbatim from §2125 rung 30.** `|L2_F(baseline) − 2.6735| ≤ .05`. *Worked example:* the
  same pipeline on the same data reproduces to ≈ **.00–.02**; anything larger means the derivation perturbed the construction and
  **nothing else in this rung is readable.** Registered first for that reason.
- **pred_b — collapsing a5 is free.** `cost_a5 = L2_F(a5) − L2_F(baseline) ≤ +.02` nats (POSITIVE = WORSE). *Worked example:* if the
  ten rows are near-identical, replacing them by their mean changes almost nothing, ≈ **+.00 to +.01**; if the class structure is
  load-bearing inside the frontier, ≈ **+.1 or more**, and §2835's real-model result does not transfer to the construction.
- **pred_c — the control layer is not free.** `cost_ctrl ≥ +.05` nats **and** `cost_ctrl ≥ 5 × cost_a5`. *Worked example:* if a6's
  dictionary genuinely uses its classes, ≈ **+.1 to +.4**; if it reads ≈ **+.00** then collapsing *any* motif dictionary is free, pred_b
  says nothing about a5 specifically, and the whole reading collapses. This is the clause that stops the rung from proving a triviality.
- **pred_d — a5's rows are near-identical, structurally.** `1 − min pairwise cosine` among a5's ten `CV` rows ≤ **.05**, and strictly
  smaller than a6's. *Worked example:* §2835's |cos| .9999996 in the real model predicts ≈ **.00–.02** here; a dictionary using its
  classes gives ≈ **.3+**. Measured on the BASELINE fit, so it is independent of whether the collapse itself is cheap.
- **pred_e — the parameter saving is stated exactly**, not gestured at: `10·D + |LINK|·D² − D` per collapsed layer, D=1152.

## Nulls

- `b_null_a5_class_structure_is_load_bearing` (`cost_a5 ≥ +.05`): **§2835's constant does NOT survive composition into the frontier** —
  the real-model result would then be about attn5 in isolation and not about the program, which is exactly the composition failure
  §2125 found for Fisher selection. This is the outcome that would most limit the arc and it is registered to be recognisable.
- `c_null_collapsing_anything_is_free` (`cost_ctrl ≤ 2 × cost_a5`): the control refutes the specificity claim.

## Price

**3 full frontier pipeline runs, ≤ 800 GPU-seconds** (§2125's parent ran two arms in 204 s each), 0 backwards, 0 fitted parameters
beyond the pipeline's own fits. The parent script is **not forward-instrumented**, so the receipt reports `gpu_forwards: 0` with
`forwards_instrumented: false` and `pipeline_runs: 3` beside it; the ledger's `Price:` line states this explicitly rather than implying
a measured forward count. Receipt: `frontier_a5_constant_collapse_results.json`, read with `price` in the same command the ledger
section is written from, in the canonical `Price: N GPU forwards, X GPU-seconds` / `Results: <file>.json` form (§2853, §2858).
