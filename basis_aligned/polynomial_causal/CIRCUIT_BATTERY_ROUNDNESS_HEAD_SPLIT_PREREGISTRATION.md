# CIRCUIT BATTERY — ROUNDNESS HEAD SPLIT (preregistration)

Registered 2026-09-04 06:15Z (box clock, read immediately before this line was written). Claude, LANE 1 CUDA.
Rung `circuit_battery_roundness_head_split`. Script: `ops/circuit_battery_roundness_head_split.py`.
Input receipt: `circuit_battery_roundness_localisation_results.json` (§2842, sha 331454aac1ce218d9194255e19c81c53eca38d99cc6c2b685ff2d9e0ac12788c).
IMMUTABLE: any change gets a new document, not an edit.

## Object

Attention 8 is now credited with two separable jobs. §2808/R576 and §2820: it writes a context-blind function of the last visible item,
concentrated in heads **{3, 7}** (top-2 share .877 on 6 of 7 behaviours, with its class and margin head maps identical). §2842: it also
carries the roundness switch that decides whether the downstream stack computes LAST + STEP or LAST + 1, ranking 1st of 36 in both the
percent and bare formats and recovering **.589** of that decision. This rung asks whether the two jobs share heads.

`c_proj` is linear without bias, so a head-level interchange is exact: the donor's slice of the concatenated head outputs is swapped
into the base run before the projection, one head at a time, on §2842's minimal pairs (`10% 20% 30%` → ` 40` against
`11% 21% 31%` → ` 32`, identical token length).

Sign convention: `ld = logit(plus-one) − logit(step)`, `REC = (ld_patch − ld_base)/max(ld_donor − ld_base, 1e-3)`; 0 = no effect, 1 =
full switch. **No CE, no §312 L2, nothing installs.**

## Predictions

```
BARS  = {top_rec: .40, top2_share: .60, exact_tol: .05}
NULLS = {top_rec_le: .15, top2_share_le: .30}
R576_HEADS = (3, 7);  WHOLE_COMPONENT_RECOVERY (§2842) = .5889
```

**pred_a_a_head_carries_the_switch** — median over the two formats of the best single head's REC ≥ .40.
*Worked example:* the whole component recovers .589 (§2842); if one head carries most of that, .4–.55, and if the switch is spread over
the nine heads, each reads .05–.15. Both operands are logit differences with a floored denominator. Null: ≤ .15.

**pred_b_it_is_the_r576_pair** — the top-2 heads are exactly {3, 7} in BOTH formats. *Worked example:* if writing the item's identity and
writing its roundness are the same operation, the pair that carries the identity (§2820, on Codex's R576 finding) also carries the
roundness and this is TRUE; if they are separable jobs sharing a component, a different pair leads and this is FALSE — which is the more
informative outcome and would locate two distinct functions inside one attention layer. Boolean over two formats.

**pred_c_two_heads_suffice** — median over formats of the top-2 heads' share of the total POSITIVE head recovery ≥ .60.
*Worked example:* §2820 measured a top-2 share of .877 for the identity write; if roundness is similarly concentrated, .6–.9. The
denominator sums only positive recoveries so heads pushing the other way cannot inflate it. Null: ≤ .30.

**pred_d_the_head_is_format_invariant** — the same head leads in both the percent and the bare format. *Worked example:* roundness is a
property of the number, not the "%" surface, so a genuine roundness head leads both; §2842 already showed the component-level leader is
format-invariant while the tail is not.

**pred_e_all_heads_equal_the_whole_component** — |REC(all nine heads patched) − .5889| ≤ .05. *Worked example:* patching every head of
attention 8 is the same intervention as patching the component, so this should reproduce §2842's number to float and batching noise,
~.00–.02. A larger gap means the head-slice interchange is not equivalent to the component patch and no head number here can be read as
a share of it. This is the instrument check, and it is the reason §2842's value is carried in as a literal rather than re-measured.

## Stated null

No head carries the switch (best ≤ .15), it is spread across heads (top-2 ≤ .30), or the pair differs from {3, 7}. The first two would
say the roundness feature is distributed within the component even though the component is a sharp leader across 36; the third would
separate the two jobs, and either is reported as measured.

## Price

2 formats × up to 24 pairs × (2 native + 9 head patches + 1 all-head patch) forwards, batched by token length.
Literal budget: ≤ 200 GPU forwards, 0 backwards, **0 fitted parameters**, < 60 GPU-seconds.

## What this does NOT claim

Head granularity only — no QK/OV decomposition, no position analysis, no subspace split within a head. No selectivity control exists in
a two-behaviour minimal pair, so nothing here is a selectivity claim. One step size and one digit range, inherited from §2841. The pairs
are §2842's construction, not the bank's frozen splits. Does not satisfy Codex's four-phase integration contract; updates no circuit
record.
