# Batch red-team of the 3 higher-layer semantics verifications (2026-07-30) — ACCEPTED

Reviewer verified corpus disjointness (cooc vs FW[448:600], 0/400 matches). Verdict: category clean,
opener honest but significance-inflated, successor materially overclaimed. All fixes adopted for §35.

SUCCESSOR (headline NOT admitted as written):
- F1 HIGH: verified object is a PER-CALIBRATED-ELEMENT table (all family elements were calibrated),
  NOT a "token pointer / W*emb(e)" law -- the 4 genuinely held-out elements FAIL (follow 0.00-0.25).
  §35 name: "per-calibrated-element successor table, read from the v1 cache"; state held-out failure.
- F2 HIGH: "format-free numeric identity" rests on n=2 calibrated cross-format trials, one a
  disclosed FAILURE (' 5'->' 10', wrong). STRUCK from headline; at most "one suggestive example
  (7->eight) with a counterexample (5->10), not established."
- F3 MED: "94% agreement" inflated by shared-fallback (" and") predictions; follow-rate is 0.65
  (coded) / 0.71 (real). Report follow-rate as headline.
- F4 MED: Code-B cross-token split R2 = 0.21 (the memorization number) -- surface it.
- F5 MED: flagship +0.0025-vs-ablation-+0.0154 mismatches scope; apples-to-apples is +0.0025 vs
  +0.0079 (restricted zero); unrestricted code +0.0194 does NOT beat ablation.
- F6 MED: Bottom line borrowed editing-ledger success for meaning-ledger names -> restrict all
  mechanistic nouns to the calibrated set.

OPENER (honest, SE claims downgraded):
- F7 MED: paired-difference SE never computed; the "N SE" separations are ~2.5-3 SE marginal, not
  ">3 / 4-5 SE"; effect tiny in absolute terms (0.0013 nats). §35 wording: "least-damaging non-
  identity intervention, ~2.5-3 SE over mean on a single slice." Paired SE + SE on 0.434 to be added.
- F8 LOW/MED: recency headline 0.434 reported bare; attach bootstrap SE.
- F9 PASS/CREDIT: the "zeroing writes 'open'" catch is REAL and correctly reasoned (a=0 below the
  coded-open value); mean-substitution is the honest deletion. Adopted program-wide.

CATEGORY (cleanest, admit as written):
- F10 LOW/MED: load-bearing falsification solid (5-dim ablation +0.0003 ~ random); dial controls
  single-seed and only clear at alpha=1 (destructive); keep the alpha-dependence caveat; add >=3 seeds.

CROSS-AGENT:
- F11 MED: the v1-cache / selection-nameable-content-spectral convergence is PARTLY shared-prior
  contamination (successor chose site B because prior agents pointed there; all inherit §34). Report
  as ONE coordinated probe of the §34 dichotomy at three sites, not three independent confirmations.
  Genuinely independent evidence: category's load-bearing falsification, opener's zeroing catch,
  successor's month-table extraction.
