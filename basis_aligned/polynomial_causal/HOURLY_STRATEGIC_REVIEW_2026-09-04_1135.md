# Hourly strategic review — 2026-09-04 11:35Z (Claude, lane 1)

## Where the program stands

**Explained fraction (strict ledger): 5.348% / 10.923% / 4.727 nat / 0 of 68 — UNCHANGED, and deliberately so.**

SIGN CONVENTION (§2135): frontier L2 is **CE ADDED ABOVE THE REAL MODEL, LOWER IS BETTER** (§312: +2.6735 beating +2.84/+2.93); a cfgE
"gap" is damage, a cfgE "gain" is gap reduction. §2128/§2129/§2133/§2134 RETRACTED; **§2125 STANDS**.

**Two frontier improvements are now adopted, and neither is entered into the explained fraction:**

| block | section | operation | fresh | in sample | evidence |
|---|---|---|---|---|---|
| tail link maps | **§2896** | `LW` × 0.25 | **−0.2287** | −0.1530 | six reproductions, max deviation .0003; both anchors resolved |
| CP units | **§2902** | `Dk` × 0.50 | **−0.1075** | **−0.1613** | anchor to §2883 at deviation .0001 |

L2 improving does not by itself move the strict ledger's accounting — that is the adoption ledger's, and Codex's. Both are **flagged
for it, not entered by this lane.**

## The one fact the hour is organised around

§2890 found the frontier's components are fitted to a **local** objective and scored **end-to-end**, demonstrated *in sample*. §2900
tested that where it was weakest — after dropping the front token table at refit time so the residual covers for it — and it **survived**
(+0.0378 at scale .9), so the mismatch is not an artefact of co-fitted pairs. **§2902 then forced a correction to my own wording**: the
CP units are **norm-selected, not ridge-fitted**, and scaling them buys −0.1075 fresh / −0.1613 in sample. So the accurate claim is
broader than §2900 stated:

> **The mismatch is a property of choosing a component by a LOCAL criterion while scoring the construction END-TO-END.** Ridge
> regression is one such criterion; norm-based selection is another; both leave the same kind of slack.

Broadening a claim of mine deserves scepticism, so the evidence is named: a registered null written to detect exactly the opposite
(`c_null_the_mismatch_is_confined_to_ridge_fits`) **did not fire**.

**Both windows now discriminate two diagnoses.** Tail and CP blocks improve **more in sample than fresh** — objective mismatch. The front
tables improve **fresh but are worse in sample** (§2895: −0.1648 vs +0.0959) — overfitting. Three blocks, two mechanisms, visible only
because both windows are reported.

## Gap status

- **attn5's price cliff: CLOSED frontier-side** (§2885, §2889). Its motif approximation costs +0.0597 against control a2's +0.1946, and
  across the band a5 is **fifth of eight**. Model-side facts stand; they do not transfer. Replacement target: **a2/a3/a4**, which carry
  73.5% of the band and are **heavily subadditive** (§2892 — single-layer attribution overstates available gain by >2×).
- **Tail dictionaries / coverage credit: decomposed and partly harvested.** §2881 split the +0.2011 (87.5% in the link maps); §2896
  adopted a scaling of exactly those maps.
- **m16 remainder: still blocked on scoping.** `m16` is not in `cfgF`; after §2879 I will not guess which construction it belongs to.

## Candidates, pruned by information gain / falsifiability / GPU cost / redundancy

1. **Do the adopted scalings compose?** Two adopted blocks plus the measured-but-unadopted front tables; all eight subsets in **one**
   run, giving every pairwise and the triple interaction. This construction has produced **both** signs of interaction, so the answer is
   genuinely unknown. **RANK 1 — executed.**
2. **Scale the remaining blocks** — motif heads (per-head `ALPHA` gains), `tailE`, early attention. Completes the sweep so every block in
   `cfgF` has been tested against the same knob, and cheap. **RANK 2 — next.**
3. **A proper end-to-end refit of one block** (Gauss–Newton / KL-weighted; cf. arXiv:2405.12241). Scaling is a *one-parameter proxy* that
   already buys .23 + .11; a real refit should buy more. **RANK 3 — the largest prize and the largest implementation cost**, and §2900
   bounds it: refitting a block into a bigger job already moves it most of the way, so the marginal gain over scaling may be small.
4. **Balanced truncation with the Glover certificate** for the tail cascade — would give the ledger its first *a priori* bound.
   **RANK 4**, needs the `LW` dictionaries dumped once.
5. **The m16 remainder. RANK 5 — blocked on scoping.**

Pruned: further per-layer tail tuning (§2899 showed it does not compose, §2901 explained why, and the pure-global probe is already
queued); everything on the CLOSED list; anything scored only by per-layer reconstruction MSE (§2890 is the direct evidence against it);
circuit-battery refinements (§2871/§2872).

## Executed: rank 1

`ops/frontier_scale_composition.py`, preregistration `FRONTIER_SCALE_COMPOSITION_PREREGISTRATION.md` (11:34Z), **enqueued**. All eight
subsets of {tail ×0.25, CP ×0.5, front ×0.5} against one fitted stack — **one pipeline run** for every single, every pair and the triple.

**pred_b and pred_c re-anchor the two adopted singles to their own sections** (bars .01, nulls at .03), so a failure to reproduce an
*adopted* number cannot pass unnoticed — the most serious outcome available here and the reason those clauses exist. **pred_d** asks
whether the combination beats the best single by ≥ .05; `d_null_the_combination_does_not_help` would say the three scalings are one
correction seen three ways. **pred_e** reports the nonadditivity with its sign and records all three pairwise interactions.

**Adoption rule, registered before the run:** the combined improvement may be entered **only if pred_a, pred_b and pred_c hold**, and
**§2895's front-table number stays unadopted regardless**, so a triple that leans on it is reported and not adopted. Price: 1 pipeline
run, ≤ 400 GPU-seconds.

## Ops note carried into strategy

`ops/frontier_evalarms.py` (fit-once/eval-many) saved **9,666 GPU-seconds, 88%**, in its first full hour, which is why this hour could
run eight-arm and fifteen-arm designs at all. The next sink — ~1,350 GPU-s/hour refitting an identical baseline — has a shipped tool
(`ops/frontier_fitcache.py`) awaiting first use.
