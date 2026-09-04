# Hourly strategic review — 2026-09-04 12:35Z (Claude, lane 1)

## Where the program stands, and the thing I should have checked earlier

**Explained fraction (strict ledger): 5.348% / 10.923% / 4.727 nat / 0 of 68 — UNCHANGED.**

SIGN CONVENTION (§2135): frontier L2 is **CE ADDED ABOVE THE REAL MODEL, LOWER IS BETTER** (§312: +2.6735 beating +2.84/+2.93); a cfgE
"gap" is damage, a cfgE "gain" is gap reduction. §2128/§2129/§2133/§2134 RETRACTED; **§2125 STANDS**.

This lane has spent the last three hours producing frontier improvements: **T+C adopted at +2.3522** (from +2.6735), with §2896's tail
scalar reproduced **eight** times and §2902's CP scalar anchored at .0001. I have flagged each for the adoption ledger and entered none.

**This review's finding is about the metric rather than the work.** I looked up what the fourth component counts: the ledger records
*"0 of 68 stands; this is a distinct effect-variance metric, proposed as coverage credit"* — **it is a certificate count, and it has been
exactly zero for the whole campaign.**

That reframes what the scaling programme is. **It improves the number and certifies nothing.** Rescaling three fitted objects makes the
construction better without explaining any component or bounding any error a priori. It is a genuine *manipulability* result — the
program is editable and we found a better setting — but it is not an explanation result and it cannot move three of the four components
of the fraction. **I should have established that before, not after, spending three hours on it**; the numbers are real and the strategic
accounting was loose.

## What the scaling programme did establish, kept in proportion

- **A unifying mechanism** (§2890, broadened §2902): components are chosen by a **local** criterion and scored **end-to-end**; ridge
  fitting and norm selection both leave the same slack. §2900 showed it survives refitting; §2905 showed it is **directional per block**
  (motif wants *more*, everything else *less*).
- **Two windows separate two diagnoses**: tail and CP improve more in sample than fresh (objective mismatch); the front tables improve
  fresh and worsen in sample (overfitting, §2895) — which is why TF was **not** adopted despite being the lowest fresh number on the
  table (§2904).
- **One-scalar sufficiency** (§2903): per-layer, prefix and joint-grid tuning all failed to beat a single number for the tail block.
- **Interaction structure**, measured rather than assumed: T·C nearly independent (+0.0149), motif 91% absorbed by T+C (§2906),
  one-way compensation of 4.9479 vs 0.0215 in the front stage (§2897).

## Candidates, pruned by information gain / falsifiability / GPU cost / redundancy

1. **Unblock the certificate line.** `0 of 68` is the only component of the fraction at literal zero, and a single success moves it to
   1. The 10:28Z mathematical review ranked balanced truncation with the **Glover bound** (`‖G−G_r‖_∞ ≤ 2 Σ_{i>r} σ_i`) second and named
   its blocker: **the fitted matrices are never written to disk**. That is mundane and removable. **RANK 1 — executed.**
2. **Finish the scaling sweep** (`frontier_joint_three_scalar` running, `frontier_remaining_block_scale` queued). Closes the programme
   cleanly either way; `d_null` on the latter would say the corrections are **one two-parameter object**. **RANK 2 — in flight.**
3. **A genuine end-to-end refit** (Gauss–Newton/KL; cf. arXiv:2405.12241). Largest prize, largest implementation cost, and §2900 bounds
   it: refitting a block into a bigger job already moves it most of the way, so the marginal gain over scaling may be small.
   **RANK 3.**
4. **Möbius attribution over all six blocks** — partly delivered by §2904's three-block decomposition; the remaining value is lower than
   when the mathematical review ranked it. **RANK 4.**
5. **The m16 remainder — still blocked on scoping.** `m16` is not in `cfgF` and after §2879 I will not guess. **RANK 5.**

Pruned: further tail-scale work (§2903 closed it); anything on the CLOSED list; anything scored only by per-layer reconstruction MSE
(§2890 is the direct evidence against it); circuit-battery refinements (§2871/§2872).

## Executed: rank 1

`ops/frontier_stack_dump.py`, preregistration `FRONTIER_STACK_DUMP_PREREGISTRATION.md` (12:34Z), **enqueued**. It fits the published
stack once and **persists it** via `ops/frontier_fitcache.save_stack`, after which Gramians, Hankel spectra and per-matrix spectra are
computable on CPU at **zero GPU cost**.

**pred_b** requires the dump to round-trip with **max deviation 0.0**, verified **recursively** over tuples and dicts — `S` maps a key
to a tuple containing a dict of tensors, so a top-level check would be structurally blind to a changed link map, which is exactly how
`ops/fastload.py` shipped broken at 06:24. **pred_e** is the one that tests *usability*: the reload arm **replaces every fitted entry
with the one read back from disk** before evaluating, and must reproduce L2_F within .001. That arm was rewritten during construction —
the first version would have evaluated the in-memory stack and tested nothing.

**A tool re-valued, and I would rather say so than quietly reuse it:** `frontier_fitcache.py` was written at 11:06Z to save GPU time and
**I downgraded it myself** at 12:07Z when the GPU turned out to be 68% idle. Its real value is a different one — getting the fitted
objects onto disk — and that is what makes the certificate line cheap.

Price: 1 pipeline run, ≤ 400 GPU-seconds, plus a one-off disk write recorded in the receipt.

## Standing

Explained fraction **unchanged**. Two adoptions flagged for the adoption ledger, none entered by this lane. **attn5's price cliff closed
frontier-side** (§2885/§2889); tail dictionaries decomposed (§2881) and partly harvested (§2896); **m16 still blocked on scoping**.
