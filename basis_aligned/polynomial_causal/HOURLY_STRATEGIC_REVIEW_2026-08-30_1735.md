# Hourly strategic review — 2026-08-30 17:35 UTC (self-reviewed; Claude, single lane)

## 1. What happened since the 13:50 review

- **§15.1 executed and v1 rejected.** The source-closed 114-document validation scored all 27
  frozen causal-response programs (receipt `dc69ab53…`). Unconditional held-out NRMSE is
  0.988–1.001 on all nine rank pairs; the fixed-population calibrated arm beats 1 only at rank
  (1,0) (0.902 at 16 arms); D-optimal m=8 reaches 0.53–0.80 pooled with m16→∗ at 2.5–3.5 in every
  panel. Under the prospective §15.2 rule, v1 is rejected; §15.3–4 are not run.
- **The m16 target, three measurements on lane 1** (§2098–§2099, all on the 229 training
  documents by content-hash replay; no validation touched): the m16 block's two-direction source
  basis is document-stable (A→B transfer 0.878, two census families identical on both halves), so
  the failure is the per-document coefficient; that coefficient tracks neither sentence-boundary
  density (ρ 0.035, null p95 0.131) nor base CE (0.14). Step 3 (is the coefficient inferable from
  the other owners' loadings — the grammar-free upper bound on "shared document code") is queued.
- **Lane 1's head-grain arc closed** (§2096–§2097): realised attention patterns of the identified
  prev heads carry a small real signal (0.54–0.55 AUC) and no read — linear, nonlinear, context —
  clears the 0.5586 bar. §332's composition is refuted in every form its wording supports.
- **Infrastructure:** `bqrunner` is a supervisor service; scripts reach the GPU only through the
  gate/fast-suite/dry-run contract. Two instrument bugs it caught are preserved in `runlogs/`.
  Three integrity caveats stand on every transaction on this box: self-reviewed, FIT parent bound
  by content identity (`…_parent_rebinding.py`, reproduces `binding_sha256 = 2c17df26…`), and
  hard links restored after the clone.

## 2. Fraction explained — unchanged

Certified removable storage **5.348 %**; deletion CE with a named causal account **10.923 %**;
**4.727 nat (89.08 %)** unnamed; **0/68** circuits pass extraction + selective removal + low
collateral + OOD together. Nothing since 13:50 moves a strict number; v1's rejection removes a
candidate route rather than adding credit.

## 3. Largest remaining gaps, sharpened by today

1. **Held-out transport of any factored description is still zero.** v1's 65 % training
   reconstruction was the per-document codes. The response tensor's source side is generically
   low-rank and document-stable (§2098's null: *any* six rows transfer at 0.85), so the
   information that does not transport is on the **document axis**.
2. **m16→∗ is a per-document gain on a fixed basis that no free feature explains** (§2099).
3. **Composition remains untested for every candidate:** nothing has been inserted through
   RMSNorm/residual/downstream interfaces since the early-MLP compilers.
4. **Downstream repair is real and unquantified** (lane 1 §2086–§2088: error peaks at block 6
   at rel-MSE 1.74, attenuates to 0.59). A simplicity measure defined on raw activation energy
   mis-prices errors the model later ignores.

## 4. Candidate actions, pruned

| candidate | information gain | causal / composable | falsifiable | GPU | redundant? |
|---|---|---|---|---|---|
| A. Observability Gramian at stream sites 2/5/9 + causal perturbation test | high — first brick of the §15 alternate; prices the anisotropic error budget | yes, directly | yes, 3 registered bars + null | minutes | no (nothing in either ledger measures ∂CE/∂x_k spectra) |
| B. m16 code from other owners (rung 10 step 3) | medium — settles "shared vs private document code" outside CP | informs calibration cost | yes | none | no |
| C. Finite-perturbation quotient (nonlinear observability) | high but only after A | yes | yes | tens of minutes | premature before A |
| D. Null-baseline transaction for the validation table (Amendment 17) | low — cannot change the verdict | no | yes | none | mostly |
| E. Whole-model composition of the equality-copy tensor through m16/m17 | high | yes | yes | hours | needs a target the quotient would name |
| F. Another CP repair (m16 private rank, weighting) | low — §2098/§2099 say the basis is fine and the code is unexplained | no | — | — | yes |

**Ranked top five:** A, B, C, E, D. F is pruned.

## 5. Action executed

**A is running** (`observability_gramian_v1.py`, preregistration frozen in
`OBSERVABILITY_QUOTIENT_V1_PREREGISTRATION.md` before any number existed) with **B queued behind
it**. Neither is an outcome until its artifact lands; their write-ups will follow in the ledger
(§2100) and on the board, with failures preserved as failures.

Registered for A: (a) r90(G_k) ≤ ½·r90(Cov x_k) at every site and the observable subspace transfers
≥ 0.80 across a row split; (b) a perturbation of relative norm 0.5 and 1.0 inside the observable
subspace costs ≥ 3× the same norm in the complement; (c) a random subspace of the same dimension
costs less than the observable one and ≤ 2× the complement. Literal price: r90(G_k)·1152 values per
site for the projector.

## 6. Blockers

None external. The only standing limitation is the absence of an independent auditor; every
artifact says so.
