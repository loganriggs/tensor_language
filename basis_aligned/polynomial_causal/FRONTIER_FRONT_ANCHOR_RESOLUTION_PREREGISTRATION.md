# Frontier: quantify the refit compensation in the front MLP stage — preregistration

Registered 2026-09-04T10:42Z (the exact string `date -u` returned in its own tool call immediately before this write). Before the run.
Immutable; the rung's frozen-hash check refuses to execute if this file changes.

## SIGN CONVENTION

Frontier L2 is **CE ADDED ABOVE THE REAL MODEL, so LOWER IS BETTER** (§2135; §312: "+2.6735 beating +2.84/+2.93"). A cost is
`L2(arm) − L2(baseline)`, **POSITIVE = WORSE**. §2128/§2129/§2133/§2134 RETRACTED; **§2125 STANDS** — norm-2304 at 2.6735.

## Why

§2895 shrank the front MLP tables and found improvements on the fresh window (best **−0.1648 at `a_scale` .5**) — and its `tb := 0`
anchor **missed by 4.9484 nats**: **+5.6298** frozen against §2877's **+0.6814** refit-time. The `A := 0` anchor **held** at .0215.

**That asymmetry is the interesting part.** §2877 zeroed `tb` **during fitting**, so the quadratic residual `A` was then fitted against
a residual computed with no token table and **absorbed the lookup's job**. Zeroing `tb` on a frozen stack gives `A` no chance to
compensate. The gap between the two is the size of that compensation — and at ≈ 4.95 nats it would be **an order of magnitude larger
than any interaction this ledger has measured** (§2880 +3.2104, §2888 +0.3023, §2894 +0.0610).

Three runs against **one** baseline:

| run | arms |
|---|---|
| 1 — frozen stack | baseline; `tb := 0`; `A := 0`; `a_scale = .5` |
| 2 — refit-time | `tb := 0` applied **inside** the fitters — §2877's exact operation |
| 3 — refit-time | `A := 0` applied **inside** the fitters — §2877's exact operation |

## Predictions, each with its worked-example line

- **pred_a — REPRODUCTION GATE, verbatim from §2125 rung 30.** `|L2_F(baseline) − 2.6735| ≤ .05`. *Worked example:* every rung in this
  family reads +2.6735/+2.6736; past .05 and **nothing else here is readable.**
- **pred_b — the refit-time `tb := 0` arm reproduces §2877.** `|cost − 0.6814| ≤ .05`. *Worked example:* it is §2877's exact operation,
  so ≈ **.000**. A miss ≥ .10 (`b_null_the_tb_anchor_fails_at_refit_time`) means the refit path itself is not reproducible and the whole
  front-table line stays closed.
- **pred_c — the refit-time `A := 0` arm reproduces §2877.** `|cost − 0.7536| ≤ .05`. §2895 already measured **+0.7321** for the
  *frozen* version, so this arm additionally tells us how little the residual's removal depends on refitting.
- **pred_d — the token table's refit compensation is large.** `|cost(frozen tb:=0) − cost(refit tb:=0)| ≥ **3.0** nats`. *Worked
  example:* §2895 vs §2877 implies ≈ **4.95**; if it comes back ≤ **1.0** (`d_null_the_compensation_is_not_the_cause`) then refitting is
  **not** what separated the two numbers and §2895's failed anchor needs a different explanation.
- **pred_e — the residual's refit compensation is small.** `|cost(frozen A:=0) − cost(refit A:=0)| ≤ **0.10**`. *Worked example:*
  §2895's frozen +0.7321 against §2877's refit +0.7536 implies ≈ **.02**. Predicting **both** a large and a small compensation, on the
  two halves of the same stage, is what makes this an asymmetry claim rather than a general statement that refitting matters.

## Nulls

- `b_null_the_tb_anchor_fails_at_refit_time` / `c_null_the_A_anchor_fails_at_refit_time` (≥ .10): the refit path is not reproducible.
- `d_null_the_compensation_is_not_the_cause` (≤ 1.0): §2895's 4.95-nat miss has some other explanation.

**Adoption note:** this rung is diagnostic. It does **not** adopt §2895's −0.1648, and the receipt records
`adoption_gate_all_of_a_b_c` so that a later rung can see whether the anchors were sound without re-deriving it.

## Price

**3 full frontier pipeline runs, ≤ 700 GPU-seconds** (this family measures 104–166 s per run), 0 backwards, 0 fitted parameters beyond
the pipeline's own. The parent `ops/frontier_fisher8.py` is **unmodified**. It is **not forward-instrumented**, so the receipt reports
`gpu_forwards: 0` with `forwards_instrumented: false` and `pipeline_runs: 3`, and the ledger's `Price:` line says so — the count is
absent, not zero. Receipt: `frontier_front_anchor_resolution_results.json`, read with `price` in the same command the ledger section is
written from, in the canonical `Price: N GPU forwards, X GPU-seconds` / `Results: <file>.json` form (§2853, §2858), under a filename no
other section cites (§2876).
