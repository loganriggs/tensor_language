# Rung 533 terminal receipt: copy effect transfers, but OOD specificity does not

**Completed:** 2026-09-03 12:59 UTC

**Independently audited:** 2026-09-03 13:03 UTC

**Owner:** Codex

## Registered verdict

```text
A exact live physical instrument                  true
B complete-product positive control               false  (4/8 contexts)
C both source factors fill target first           false  (5/8 and 6/8)
D both source factors fill target second          false  (5/8 and 6/8)
E four-way branch-exchangeable family             false
F stable with donor present and absent            false  (2/16 comparisons)
```

The preregistration says A or B failing makes the identification test invalid. A passed and B failed. Therefore
rung 533 is **not** evidence for or against a globally interchangeable four-factor family. Its registered outcome is
`invalid_identification_test_positive_control_failed`, not a scientific strong null.

The managed run executed exactly 2,208 model forwards over 192 separate natural-text and 192 repository-disjoint
code documents, split into fixed halves and run with the source equality contribution present and absent. There were
zero backward passes and zero fitted vector parameters.

## Why the positive control failed

The complete scaled source product still reproduced the desired copy-positive effect in 7/8 contexts. Its minimum
copy-effect cosine was `0.819`; its maximum relative error was `0.587`. The one positive-effect failure was natural
text, donor present, half 0, where copy recovery was `0.468`, below the fixed `0.65` floor.

The more systematic failure was OOD specificity. On code, the complete-product control changed matched
copy-negative loss by more than the fixed `0.01`-nat limit in three contexts. In the cleanest donor-absent code test,
the two halves still had copy-effect cosines `0.990/0.991` and recoveries `0.875/0.870`, but matched-negative changes
were `0.0249/0.0233` nat. Thus the source and target products carry nearly the same useful copy effect but do not
reject false matches in the same way.

## What remains informative without rescoring

All four source-factor to target-slot mappings beat their own scale-matched key-permuted control by at least `0.15`
cosine in all 8/8 contexts. Their positive-effect-only bars passed in 7/8 or 8/8 contexts, while the full registered
bars passed in 5/8 or 6/8. These are descriptive facts, not a retroactive identification claim.

This is consistent with multiple heads computing a common copy/equality relation while having different
context/noise filtering. It corrects the stronger rung-532 gloss: the branch-exchangeable family was a hypothesis
licensed by the census result, not yet an identified OOD circuit.

## Next object

Rung 534 uses the complete, gauge-invariant products rather than native factor branches. With source product `P_s`,
target product `P_t`, and the already frozen product scale `gamma`, define

```text
shared equality signal       S = gamma * P_s
target-specific correction   R = P_t - S
exact target product         P_t = S + R.
```

The next causal question is whether `R` acts as an autonomous context/noise correction or only through interaction
with `S`. That directly tests cross-head grouping, within-head splitting, OOD selectivity, and composition without
rank reduction.

## Integrity

- result SHA-256: `5c43872a037f662ab93c64915e74419439513393f026654d8ed16c7bdb7f84d0`
- sufficient-statistics bundle SHA-256: `c4a2173ad88624dff33974891f9815d98ae7e7d6e7162d3291d6c7ac0e6ecae4`
- frozen runner SHA-256: `6ba3a9e5fa4e0fa23c461610451bfc8d65eea909f14fe563131a1441228528fd`
- managed log SHA-256: `d8ef4bdf7931f4fd95ffadb02451bfc24b37839f8319fa200b2f62400a16c46c`
- terminal audit SHA-256: `45fee6f43f2ce2fef7d4bdc7f30da7458c5808d11ebc1d4fa8c21bcc85117518`

The audit independently reloaded the saved document-level sufficient statistics, recomputed every report and frozen
predicate, reconciled the call/support ledger and checkpoint, and verified that no raw tokens, logits, hidden states,
or per-token losses were stored.
