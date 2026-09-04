# Frontier: attention 5's price cliff, asked of the frontier properly — preregistration

Registered 2026-09-04T09:46Z (the exact string `date -u` returned in the tool call immediately preceding this write; five headers this
session were stamped one minute ahead of the value read, so the string is pasted rather than recalled). Before the run. Immutable; the
rung's frozen-hash check refuses to execute if this file changes.

## SIGN CONVENTION, and why this measures a *share*

Frontier L2 is **CE ADDED ABOVE THE REAL MODEL, so LOWER IS BETTER** (§2135; §312: "+2.6735 beating +2.84/+2.93").
§2128/§2129/§2133/§2134 RETRACTED; **§2125 STANDS** — norm-2304 at 2.6735.

The motif heads are an **approximation** (scaled value copies) standing in for real attention, so dropping a layer from `ML` **restores
that layer to real** and should **lower** L2. The quantity is an error share, not a deletion cost:

    error_share(layer) = L2_F(baseline) − L2_F(that layer restored to real),   POSITIVE = the approximation at that layer costs that much.

## Why

**"attn5's write = the price cliff" has been one of the three largest standing gaps in the explained fraction for weeks**, and the
explained fraction (5.348% / 10.923% / 4.727 nat / 0 of 68) has not moved all session. §2874 tried to ask this of the frontier through
the `attnd` dictionaries; **§2879 had to withdraw that** — those dictionaries are not in the evaluated configuration.

Attention 2–9 enters the §312 frontier through the **38 motif heads**, and the motif-head layer list `ML` is passed **directly into
`evalM`**. It is installed by construction, which is exactly what §2879's standing rule demands, and pred_d re-checks it as a measured
predicate rather than a code reading.

**The control choice is the load-bearing design decision here.** a2 is the control because §2834's real-model census makes it the
**second-largest** component in the band — zero-ablation damage **0.349** nats against a5's **2.211** and a3's 0.141 — so it is the
layer most likely to rival a5. Beating a weak control would prove nothing; that is the §2820 lesson, applied to a layer instead of a
head.

| arm | change |
|---|---|
| BASELINE | `ML = [2…9]` — the published frontier |
| attn5 real | `ML = [2,3,4,6,7,8,9]` |
| attn2 real (control) | `ML = [3,4,5,6,7,8,9]` |

Derived from `ops/frontier_fisher8.py` (§2125 rung 30), which is **unmodified**; the derived file retargets the parent's single `OUT`,
under a filename no other section cites (§2876).

## Predictions, each with its worked-example line

- **pred_a — REPRODUCTION GATE, verbatim from §2125 rung 30.** `|L2_F(baseline) − 2.6735| ≤ .05`. *Worked example:* every rung in this
  family reads +2.6735/+2.6736; past .05 and **nothing else here is readable.**
- **pred_b — attn5 carries a large error share.** `share_a5 ≥ +.15` nats. *Worked example:* §2830 put attn5 3rd of 36 in document CE
  and **20.4× disproportionate per unit written**, so if that transfers to the frontier the motif approximation at layer 5 should be
  one of its worst, ≈ **+.2 to +.6**; if the motif heads happen to approximate layer 5 well, ≈ **.00** and
  `b_null_attn5_is_not_a_frontier_error_source` fires — which would mean the price cliff is a fact about the *model* that the
  *construction* has already absorbed, and would finally close that standing gap in the honest direction.
- **pred_c — attn5 is specific against the control layer.** `share_a5 ≥ 3 × share_a2`. *Worked example:* if a5 is the band's error
  source, ≈ **5–20×**; if every motif layer contributes alike, ≈ **1×** and `c_null_attn5_is_not_special_in_the_band` fires — the
  frontier's motif error would then be broad rather than concentrated, which changes what a fix must look like.
- **pred_d — both arms are connected.** `|share_a5| ≥ .005` **and** `|share_a2| ≥ .005`. *Worked example:* §2879's rule as a measured
  predicate — a disconnected manipulation reads exactly **.0000**, as `fit_attnd` did three times. If either reads .0000, that arm
  never reached the evaluated config and its share must not be reported.
- **pred_e — restoring a layer never harms.** `share ≥ −.01` for both arms. *Worked example:* replacing an approximation by the real
  component should not make the frontier worse; ≈ **+**. A materially negative share would mean the motif approximation at that layer
  is *better than real attention* for this construction, which is possible in a fitted stack and would be a genuine surprise worth its
  own section rather than a bug.

## Nulls

- `b_null_attn5_is_not_a_frontier_error_source` (share ≤ .03).
- `c_null_attn5_is_not_special_in_the_band` (ratio ≤ 1.5).

Either fires and the "price cliff" stops being a frontier-side target; both are worth the price because the gap has been open for weeks
on model-side evidence alone.

## Price

**3 full frontier pipeline runs, ≤ 900 GPU-seconds** (this family measures 279–283 s for three arms), 0 backwards, 0 fitted parameters
beyond the pipeline's own. The parent is **not forward-instrumented**, so the receipt reports `gpu_forwards: 0` with
`forwards_instrumented: false` and `pipeline_runs: 3` beside it, and the ledger's `Price:` line says so — the count is absent, not zero.
Receipt: `frontier_attn5_error_share_results.json`, read with `price` in the same command the ledger section is written from, in the
canonical `Price: N GPU forwards, X GPU-seconds` / `Results: <file>.json` form (§2853, §2858).
