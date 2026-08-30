# Full bilin18 update since `explanation_1405.md` — 17:45 UTC

**Date:** 2026-08-30  
**Coverage:** everything completed after the 14:05 explanation: the causal-response validation
and its verdict, the closure of lane 1's head-grain arc, the three-step m16 arc, and the first
two observability measurements. Written on a fresh instance rebuilt at 16:00 UTC.  
**Primary rule:** a plan, a queued runner, or a training fit is not an explanation. Every number
below has a preserved artifact. Everything here is **self-reviewed**: no independent auditor
exists on this instance, and every transaction says so in its authority.

## 1. The honest short answer

The strict whole-model totals have not moved:

| Quantity | Value |
|---|---:|
| Certified removable stored values | 29,196,288 / 545,904,054 = **5.348 %** |
| Deletion CE assigned to named mechanisms | 0.580 / 5.307 = **10.923 %** |
| Still causally unnamed | **4.727 nat = 89.08 %** |
| Circuits passing extraction + selective removal + low collateral + OOD together | **0 / 68** |

What changed is which routes are open. Since 14:05 the program **rejected** its main candidate
(causal-response factorization v1) under a rule it registered in advance, **closed** lane 1's
head-grain mechanism language for the attention census leaves, **localized** the m16 failure to
a per-document coefficient that no cheap feature explains, and **refuted** the first version of
the entry point that replaced v1. Four negatives, each with a number, each narrowing what is
left. Nothing was promoted.

## 2. Causal-response factorization v1: validated and rejected (§15.1–15.2 of 14:05)

The source-closed validation transaction scored all 27 frozen programs on the 114 internal
validation documents (receipt `dc69ab53…`; nothing dropped, no winner chosen).

- **Unconditional arm** (mean training code, zero validation access): pooled NRMSE
  **0.988–1.001 for all nine rank pairs**, correlation 0.15–0.21. For rank (1,0) the unconditional
  MSE is 0.0454 against a training RMS² of 0.0461: the shared factors remove ~1.4 % of response
  energy on new documents. The 65.17 % training reconstruction reported at 14:05 was carried by
  the per-document codes, not by the factors.
- **Calibrated arm, fixed population** (outcome-blind arms, identical for every candidate): only
  rank (1,0) ever beats NRMSE 1 — 0.902 at sixteen intervention forwards per new document.
  Every richer program extrapolates catastrophically from arbitrary arms (up to NRMSE 160).
- **Calibrated arm, D-optimal design:** at m = 8 (all 36 owner pairs scored) pooled NRMSE
  0.53–0.80 with the worst pair **m16→∗ at 2.5–3.5 everywhere**. At m = 16 the design anchors
  the entire m16 block for global-only programs, leaving 30 of 36 pairs — those cells are not
  comparable and the analyzer marks them ineligible (the scorer's worst-pair number silently
  skipped 108 empty pairs; the analyzer recomputes it).

Under the prospectively registered branch rule — *if held-out prediction fails broadly, reject v1
rather than repairing it indefinitely* — v1 is rejected. Gauge certification and composition
(§15.3–4) are not run: there is no survivor.

Two integrity facts on the record: the transaction is self-reviewed, and the FIT parent is bound
by **content identity** — the git clone onto this instance cannot replay the receipt's inode and
mtime, so `causal_response_factorization_v1_parent_rebinding.py` repeats every published check
with the comparator narrowed to (path, sha256, bytes). It reproduces `binding_sha256 = 2c17df26…`
exactly as recorded in the training terminal; the deviation is recorded per artifact.

## 3. Lane 1: the head-grain language for the attention census is exhausted (§2096–§2097)

For the 31 census leaves whose top-2 heads are both previous-token heads, four reads of the heads'
**realised attention pattern** (which positions they actually attended, how much, how
concentrated) were tested against the same bar (§2094's measured 0.5086 + 0.05 = 0.5586) on the
same held-out split:

| read | median AUC |
|---|---:|
| previous-token identity (§2094) | 0.5086 |
| previous-token embedding ridge (§2095) | 0.5052 |
| realised pattern, linear, one position (§2096) | **0.5409** |
| realised pattern, random-Fourier nonlinear (§2097) | 0.5413 |
| realised pattern, positions t…t−3 (§2097) | 0.5335 |
| both (§2097) | **0.5493** |

Shuffled-label controls sit at 0.49–0.50; specificity to the identified heads holds (21/31 over
same-layer control heads) with a median edge of only +0.006; no leaf reaches 0.60 under any read.
The pattern's signal is mostly *how much* the head attended in total and to t−1 — the per-query
mass degree of freedom of unnormalised attention (§1108) — and it is a third of the registered
effect. §332's proposal (motif conditions composed with value reads) is refuted in every form its
wording supports. The a3/a4 leaves remain what §348 said: two-signed activation-space bands with
no head-grain condition found.

A backlog audit found rungs 4/5/6 stale (their controls had run at §2080–§2085 and §343: the
gating ladder's "Nx random" is seed- and window-dependent; induction heads are intrinsically
high-rank in weight space). The frontier number itself is window-stable: §2085's eight
document-disjoint windows carry the assembly's excess at 2.64–2.97 (mean 2.81, sd 0.13).

## 4. The m16 target: a stable basis with a private coefficient (§2098–§2100)

m16→∗ is the worst owner pair in every panel of the validation table. Three CPU measurements on
the 229 training documents (content-hash replay; no validation value touched):

1. **The basis is document-stable.** The six m16 source circuits' deletion-response block has a
   two-direction source subspace; fitted on one prospective document half it captures **0.878** of
   the other half's energy, equal to the in-sample share, and the two source families
   ({r.1.1.1, r.1.2, r.1.2.0, r.1.2.1} | {r.1.1.2, r.6.2.2}) are identical on both halves. A
   registered null failed honestly: *any* six rows transfer at median 0.85, so source-side
   stability is a property of the whole response tensor. Programs with private rank ≥ 2 for m16
   were therefore not short of directions; what they cannot predict is the per-document
   coefficient (m16 rows are 2.7× the tensor RMS and vary by document).
2. **The coefficient is not in the text's surface.** Its rank correlation with sentence-boundary
   density (§715's rank-1 mlp16 core fires at sentence ends) is 0.035 against a permutation null
   p95 of 0.131; with base CE, 0.14 (m14 tracks difficulty more, 0.20).
3. **The coefficient is mostly private.** A grammar-free ridge from the other five owners'
   per-document loadings recovers m16's block amplitude at held-out R² **0.107** (real: null p95
   0.026; m14 the closest single covariate at ρ 0.50) against a 0.30 bar; the two loadings
   correlate 0.46, so the code is genuinely two numbers per document.

Consequence for any program: measure m16's two coefficients per document from its own arms (2 of
98, priced as such) or leave the m16 block as the unexplained remainder. The lawful token route
used here — the FIT documents are lane 1's own `census_state_diverse.pt` rows, a registered
parent of the FIT authority — is now on the record.

## 5. The replacement entry point, first brick: the linear observability quotient is not small

The 14:05 plan's alternate entry point was an empirical controllability/observability quotient.
Its first-order version — the loss-gradient Gramian G_k = E[∂CE/∂x_k ∂CE/∂x_kᵀ] at the stream
entering block k — was preregistered and measured at blocks 2, 5, 9 on 256 fresh rows:

| site | r90 of G_k (of 1152) | r90 subspace transfers A→B | ΔCE for a rel-norm-0.5 perturbation: observable / complement / random same-dim |
|---|---:|---:|---|
| 2 | **737** | 0.827 | 0.015 / 0.006 / 0.010 |
| 5 | **712** | 0.865 | 0.067 / 0.039 / 0.056 |
| 9 | **816** | 0.838 | 1.074 / 0.897 / 1.017 |

All three registered predictions failed. The observable subspace is document-stable but it is
**two-thirds of the stream**; direction matters 1.1–3.5× and a random subspace of the same
dimension costs nearly as much as the observable one. "Factor only the linear quotient" is closed
as an object.

What the measurement did give is the **price of stream error by depth**: a relative-norm-0.5
error costs 0.015 nat at block 2, 0.067 at block 5, and **1.07 at block 9**. Lane 1's assembly
carries relative norm ≈ 1.3 at block 6 (§2086); read against this table, that single mid-stream
error is worth the whole +2.9-nat frontier gap. The "downstream repair" seen earlier is the
model attenuating an error that would otherwise cost far more, not evidence that the error is
cheap.

## 6. What the mathematics contributed since 14:05

- **Prospective document splits with a registered null** (§2098, observability v1) caught two
  results that would otherwise have been over-read: source-side low rank is generic, and a
  "small" subspace that is 64 % of the space is not small.
- **Rank reasoning must name its denominator.** Comparing the gradient Gramian's r90 to the
  activation covariance's r90 was wrong because the activation covariance at block 2 has r50 = 1
  (one massive direction). The companion depth profile registers its predictions against the
  stream dimension and the measured block-2 value instead.
- **Content identity is the right binding across machines.** Physical identity (inode, mtime)
  defends a live transaction; SHA-256 and byte counts are what survive a clone, and the
  rebinding reproduces the original binding hash exactly.

## 7. Blockers and confusing results

No external blocker. Two standing limitations: every artifact on this instance is self-reviewed;
and the Codex lane is intentionally off. The confusing result is the good one: the model is
extremely sensitive to mid-stream error magnitude (block 9: 1.07 nat for a half-norm perturbation)
while being nearly indifferent to direction — which means a simplicity measure on a compressed
early program must be priced in *norm at depth*, not in local reconstruction or in a linear
observable subspace.

## 8. Current plan, in order

1. **Price error by depth at all 18 sites and test whether scale is a free gauge**
   (`stream_error_price_v1.py`, registered, queued): if pure rescaling of the stream is cheap and
   random error is not, scale-error programs (§1818's 159× head) are cheap to fix and the budget
   is about direction *within* the norm.
2. **Observable rank by depth at all 18 sites** (`observability_depth_profile_v1.py`,
   registered, queued) — whether the quotient shrinks toward the readout.
3. **The quotient relative to a program's own error:** export the assembly's block-6 error
   covariance from lane 1's §2086 diagnostics and measure its overlap with G_6; the product of
   error energy and observable weight is the first honest error budget.
4. **m16 as a measured interface:** two calibration numbers per document from m16's own arms,
   priced, inserted in a program and scored on held-out documents.
5. **Only then** a whole-model composition test of any survivor.

## 9. Primary artifacts behind this explanation

- [validation result](../CAUSAL_RESPONSE_FACTORIZATION_V1_VALIDATION_RESULT.md), table `…_validation_table.json`, analysis `…_validation_analysis.json`
- [observability preregistration](../OBSERVABILITY_QUOTIENT_V1_PREREGISTRATION.md) and [result](../OBSERVABILITY_QUOTIENT_V1_RESULT.md)
- lane 1 ledger `basis_aligned/bilinear_quotient/BILIN18_CONNECTION.md` §2096–§2100 and `BENCHMARK_BACKLOG.md` rungs 8–10
- [17:35 strategic review](../HOURLY_STRATEGIC_REVIEW_2026-08-30_1735.md)
