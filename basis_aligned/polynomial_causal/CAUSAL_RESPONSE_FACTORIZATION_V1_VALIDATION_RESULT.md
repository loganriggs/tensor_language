# Causal-response factorization v1 — validation result (Amendment 16 executed)

**Date:** 2026-08-30 16:32 UTC  
**Terminal receipt:** `causal_response_factorization_v1_validation_terminal/receipt.json`
(SHA-256 `dc69ab5371f413d3b62789736c03a56a436dd0c9ae52d79aa0b3dc8235fc9dd2`)  
**Table:** `causal_response_factorization_v1_validation_table.json` (27 programs × 2 designs ×
4 budgets + unconditional arm; 0 failed panels; nothing dropped or selected)  
**Analyzer:** `causal_response_factorization_v1_validation_analysis.json`
(SHA-256 `6f398234ceb91006fe8d24b91796c96f2b96ce8c374e1b177a8af57870572c50`)  
**Verdict under the prospectively registered branch rule (explanation_1405 §15.2):
held-out prediction fails broadly → causal-response factorization v1 is rejected as a
transporting program.** Details, scope, and the two integrity caveats follow.

## 1. Two integrity caveats, stated first

1. **Self-reviewed, not independently audited.** Every earlier transaction in this
   chain carried an independent GO audit. No auditor was available on this instance.
   The validation authority records `independent_audit: null` and the focused suite
   result (63 tests across six files) that replaced it. Anyone citing this table must
   say "self-reviewed".
2. **FIT parent bound by content identity.** The published parent binding compares
   receipt-bound artifacts by device/inode/mtime as well as SHA-256. After the repository
   was re-materialized from git on a fresh instance, SHA-256 and byte counts replay
   exactly but inode/mtime cannot. `causal_response_factorization_v1_parent_rebinding.py`
   repeats every published check with the comparator narrowed to `(path, sha256, bytes)`
   and returns the identical binding body: `binding_sha256 = 2c17df26…`, byte-for-byte
   the value recorded in the training terminal. The recorded-versus-observed physical
   identity of all three artifacts is in the authority under
   `self_review.fit_parent_physical_identity_deviation`. The published
   `parent_binding.py` was not modified.

## 2. Unconditional arm (mean training code; zero validation-response access)

Median over the three seeds; NRMSE divides by the training response RMS 0.21465.

| rank pair | P | C | pooled NRMSE | signed corr | worst owner pair (m16→m16) |
|---|---:|---:|---:|---:|---:|
| (1,0) | 100 | 1 | 0.9929 | 0.182 | 3.21 |
| (2,0) | 200 | 2 | 0.9880 | 0.206 | 3.18 |
| (4,0) | 400 | 4 | 0.9889 | 0.202 | 3.19 |
| (4,1) | 755 | 10 | 0.9909 | 0.193 | 3.18 |
| (8,0) | 800 | 8 | 1.0007 | 0.152 | 3.18 |
| (8,2) | 1510 | 20 | 1.0005 | 0.167 | 3.17 |
| (16,0) | 1600 | 16 | 0.9990 | 0.165 | 3.23 |
| (16,4) | 3020 | 40 | 0.9942 | 0.188 | 3.21 |
| (32,0) | 3200 | 32 | 0.9971 | 0.181 | 3.25 |

Every candidate sits at NRMSE 0.988–1.001. For (1,0) seed 1 the unconditional pooled
MSE is 0.04542 against a training RMS² of 0.04607: the shared factors with a mean code
remove about 1.4 % of the response energy on new documents. Per-phase NRMSE is 1.08
(full) and 0.90 (residual); per source owner the failure is concentrated at m16 (2.71)
with a16 nearly exact (0.07) — the same shape the training analysis found, now on
untouched documents. The 65.17 % pooled training reconstruction reported in
explanation_1405 §7 was carried entirely by the per-document codes, not by the factors.

## 3. Calibrated arm, outcome-blind block design (fixed scored population)

The outcome-blind design anchors the same arms for every candidate at a budget, so its
scored population is identical across candidates — this is the fair cross-candidate
view. Pooled NRMSE / worst owner-pair NRMSE, seed medians:

| rank pair | m=2 | m=4 | m=8 | m=16 |
|---|---:|---:|---:|---:|
| (1,0) | 0.970 / 3.22 | 0.981 / 3.22 | 0.943 / 3.21 | **0.902 / 3.17** |
| (2,0) | 4.00 / 13.6 | 3.91 / 13.2 | 5.02 / 16.8 | 2.43 / 8.33 |
| (4,0) | 1.97 / 7.01 | 1.77 / 6.17 | 1.42 / 4.86 | 1.73 / 5.92 |
| (4,1) | 41.3 / 110 | 41.6 / 110 | 22.1 / 57.2 | 1.85 / 5.87 |
| (8,0) | 6.06 / 21.1 | 6.11 / 21.0 | 4.38 / 14.5 | 1.43 / 4.58 |
| (8,2) | 13.1 / 34.2 | 13.4 / 33.9 | 7.72 / 17.7 | 3.36 / 9.35 |
| (16,0) | 38.5 / 204 | 24.1 / 109 | 26.9 / 152 | 4.51 / 20.1 |
| (16,4) | 160 / 637 | 145 / 559 | 207 / 828 | 49.1 / 202 |
| (32,0) | 49.4 / 259 | 42.8 / 224 | 25.5 / 130 | 4.67 / 22.5 |

Only rank (1,0) ever beats NRMSE 1, and only by 10 % at sixteen intervention forwards
per new document. Every richer program extrapolates catastrophically from arbitrary
arms; codes inferred from 98–784 cells are not the codes the training fit used. The
block-balanced Pareto frontier under this design is `{(1,0)}` alone at every budget.

## 4. Calibrated arm, training-only block D-optimal design

D-optimal selection picks the loudest arms first, and for every **global-only** program
at m=16 it selects all twelve m16 arms (both phases of sources 29–34). Those cells are
then excluded from scoring, so the m16→∗ row is absent: 30 of 36 owner pairs scored,
108 empty pairs in the table. The scorer's `worst_owner_pair_nrmse` silently skips
empty pairs; the analyzer recomputes it over scored pairs and marks such panels
`complete_owner_coverage: false`, ineligible for the block-balanced frontier. The
apparent m=16 pooled NRMSE of 0.27–0.30 for global-only programs is scored on an
easier population and is **not comparable** with any other cell.

At m=8 every panel keeps all 36 pairs. Seed medians, pooled NRMSE / worst pair / corr:

| rank pair | m=8 D-optimal |
|---|---|
| (1,0) | 0.627 / 3.37 / 0.01 |
| (2,0) | 0.545 / 2.59 / 0.16 |
| (4,0) | 0.559 / 2.70 / 0.14 |
| (4,1) | 0.799 / 2.97 / 0.37 |
| (8,0) | 0.637 / 3.52 / 0.21 |
| (8,2) | 0.761 / 2.77 / 0.43 |
| (16,0) | 0.533 / 2.47 / 0.31 |
| (16,4) | 0.739 / 2.60 / 0.49 |
| (32,0) | 0.533 / 2.46 / 0.35 |

Frontier at m=8: `{(1,0), (2,0), (16,0), (32,0)}`. The worst owner pair is m16→m16 or
m16→a16 in every panel of every candidate; the fully covered shared/private programs at
m=16 reach 0.59–0.69 pooled with worst pairs 2.44–3.15. Support fails the 90 % gate only
at m=2 for (16,4) (24.6 %) and (32,0) (62.3 %): 98 cells cannot identify 40- or 32-dim
codes.

## 5. Scope

- This is response tomography on the 114 internal validation documents. It is not
  EVAL, OOD transport, semantic extraction, selective removal, a terminal circuit, or
  whole-model ledger credit. The strict ledger is unchanged (5.348 % certified removable
  storage, 10.923 % named deletion CE, 0/68 complete terminal circuits).
- The table carries no null baseline (zero prediction, per-cell training mean). NRMSE 1
  is the training-RMS reference by construction; the unconditional MSE ≈ training RMS²
  says the validation energy is at that scale. A registered null-baseline transaction
  (an Amendment 17) would sharpen the numbers but cannot change the verdict: the
  unconditional arm is ≥ 0.988 for all nine pairs.
- The hierarchy test in the preregistration is untestable: the frozen library contains
  no independent-only (K₀ = 0) candidate.

## 6. Decision and what is now primary

Under §15.2's prospectively defined pattern — *if held-out prediction fails broadly,
reject causal-response factorization v1 rather than repairing it indefinitely* — v1 is
rejected. The failure is broad (nine of nine pairs, both arms) and concentrated
(m16→∗), which is the case §15's alternate entry point was written for: an empirical
**controllability/observability quotient** — choose early directions by what downstream
readers can distinguish, merge states with the same measured future consequences, and
factor only the quotient. That becomes the primary direction. Gauge certification and
composition tests (§15.3–4) are not run: there is no survivor to certify.
