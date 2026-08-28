# Hourly strategic review: the remaining compiler gap is a program-class gap

Date: 2026-08-28 07:30 UTC

Status: latest-receipt consolidation plus CPU interface proof. This document grants no
row, final-role, model, validation-selection, or promotion authority.

## Honest whole-model balance sheet

- Structural inventory covers 36/36 attention/MLP sites, but named behavioral
  explanation remains $32.1\%\pm6.4\%$ and named causal recovery remains $10.923\%$.
- Strict admitted executable recovery remains 0% of the distinct $+0.8976$ current-ship
  CE gap. The compiler experiments below are discovery-only and use a different table
  baseline, so their nats must not be converted into strict ship recovery.
- Under the final-CE objective, the best all-36 `table[token] + current-state low-rank`
  program reaches $+0.6006$ nat at rank 8 and $+0.6390$ at rank 32. Rank 128 reaches
  $+0.6387$: essentially the same fidelity for four times rank-32 and sixteen times
  rank-8 cost. It later destabilizes to $-0.0914$.
- The one-at-a-time sum is $+1.7460$ nat. Thus the best current joint class realizes
  only $36.6\%$ of that diagnostic sum and leaves $1.1070$ nat, or $63.4\%$, as a
  composition/class shortfall. This is not an additive decomposition of model CE; it
  is the correct within-experiment comparison.
- The rank-sweep control did not reproduce S1748 at rank-8 initialization because all
  ranks were fitted in a rank-128 prefix context. The preregistered claim that this
  could only disadvantage rank 8 had the sign wrong: it improved rank 8. Therefore the
  rank-8 starting comparison is not clean. The decisive class evidence is the
  rank-32/rank-128 tie plus diminishing returns, not the confounded rank-8 start.

## Largest remaining gaps

1. **Wrong whole-program primitive.** More rank and more coordinate passes are closed:
   a full-rank position-local map cannot move information between positions or express
   a bilinear product. The remaining error belongs to missing typed interactions.
2. **Attention interface.** A current-position map recovers only 16.4% of the attention
   output-write stake. Frozen lags $(1,2,4,8,16,32,64)$ reach 70.1% and plateau, leaving
   29.9% that requires content-dependent routing rather than a larger lag bank.
3. **MLP polynomial interface.** In the middle MLP band, adding exact native bilinear
   products to the linear carrier recovers the entire measured band shortfall, while
   512 selected products recover only 29.4%. Existing local magnitude selection is not
   the needed causal ordering.
4. **Semantic/edit interface.** MLP0 has an exact 64-D physical code, but its
   downstream-minimal dimension, nonlinear sufficiency, selective-removal behavior,
   and OOD transport are not yet measured. The Fisher quotient is preregistered but
   lacks a legal production execution context.
5. **Admission and OOD.** The canonical final source/publisher remains incomplete, both
   compiler evaluation roles are already spent, the generic Hankel splice was grossly
   OOD, and the SNR semantic ordering failed replication. No current frontier point is
   certified on a genuinely shifted corpus.

## Ranked next actions

### 1. Factorial hybrid-class oracle, then freeze the winning grammar

Under the same final-CE objective and table baseline, compare: current program; native
attention with compiled MLPs; native MLPs with compiled attention; both native as the
control. This is diagnostic, not a proposed simple endpoint. It distinguishes whether
the $1.107$-nat class shortfall is dominated by losing the squared-attention contraction,
the bilinear MLP contraction, or their interaction. The prior `mlp_only/attn_only`
experiment used different local families/objectives, so it does not answer this final-CE
question. Freeze rank 8, checkpoint selection, roles, and the four-arm interaction
contrast before execution. This has the highest information gain per GPU minute and
directly chooses between actions 2 and 3.

### 2. Preserve the attention tensor contraction; compress only its typed projections

Represent attention as the actual program

$$
q,k,q',k'=\operatorname{RoPE}(\operatorname{RMSNorm}(W x)),\qquad
P_{ts}=\frac{\langle q_t,k_s\rangle\langle q'_t,k'_s\rangle}{d_h^2},
$$

followed by the existing value-bus mixture and $W_O\sum_sP_{ts}v_s$. Replace the six
linear projections with activation/downstream-weighted factors, but keep RMSNorm,
RoPE, the squared routing product, causal mixing, residual addition, and value-bus
interface exact. Separate receipts already show all-layer QK at rank 256 costs only
$+0.060$ held-out CE (27.8% raw QK bits); routing and value projections individually
reach roughly 95% at ranks 384--512. The missing experiment is their *joint*, deployed
composition with the MLP compiler. Reject if its joint loss materially exceeds the
sum of registered projection losses or fails the hybrid oracle.

### 3. Downstream-weighted selection of native bilinear MLP products

For MLP site $l$ retain the typed program

$$
y_l=b_l+D_l\big((L_lx_l)\odot(R_lx_l)\big),
$$

but select/share hidden products by suffix consequence, not
$\operatorname{std}(h_j)\lVert D_{:,j}\rVert$. Estimate each product's final-CE or
Fisher contribution in deployed context, then compare grouped top-$K$ or sparse-group
paths at $K=128,512,1024$ against the frozen local-magnitude selection. Cheapest
falsifier is the middle band at $K=512$: it must beat the existing 29.4% recovery at
equal product count and transfer to an untouched role. Generic CP rank is pruned—the
randomized arithmetic-rank audit already gives full 1152 output rank at every tested
site, and exact product compression is bounded to about $4\times$.

### 4. Execute the MLP0 downstream predictive quotient once lifecycle authority exists

Estimate $O=\mathbb E[J^TFJ]$, combine it with code covariance $C$, and test the
spectrum of $C^{1/2}OC^{1/2}$ against nonlinear edit KL, mixtures, gauge replay,
selective removal, and OOD transport. It is lower priority for immediate whole-model
fidelity than the class oracle, but highest for turning the 64-D code into a semantic,
editable interface. A Fisher knee alone receives no credit.

### 5. Gauge-canonical cost plus prequential/OOD consequence harness

Price physical contractions and canonical factor orbits rather than arbitrary gauge
coordinates. At matched suffix distortion, compare executable FLOPs, stored bits,
data-doubling stability, untouched-corpus CE, extraction accuracy, and collateral
damage under removal. This is the validation layer that can decide whether a smaller
description is useful. It follows, rather than precedes, a viable typed program class.

## Pruned actions

- More rank in the current class: rank 32 and 128 tie.
- More bottom-up/coordinate iterations: two extra passes changed recovery by 0.0000.
- Wider fixed-lag banks: lags through 64 add only 2.0 points beyond $(1,2,4,8)$.
- Raw PCA, unweighted SVD, or local MSE as simplicity: they do not predict composition.
- A single shared dictionary across every residual bond: already lossy; gauge work found
  the embedding-pinned residual bond and shared value bus leave little free rotation.
- Generic prefix/continuation Hankel completion: tested splices were 3.54--3.61 CE OOD.
- Untyped sparse synthesis: without tensor-interface types it would search spent roles
  and rediscover noncomposable local fits.

## Safe action executed in the CPU interval

`predictive_quotient_v1_interface_proof.py` now implements a one-use sealed fake
transaction for the exact proposed MLP0 graph boundary. It creates
`predicted.detach().requires_grad_(True)`, proves exact numerical identity, proves that
both the physical projected write and parent read consume that leaf, proves suffix
connectivity and producer-graph disconnection, preserves parameter values and existing
gradients, returns only a tensor-free receipt, and revokes all stored aliases on success
or failure. Five focused tests include malicious bypass, replay, wrong-shape,
non-graph, and nonfinite-failure cases. Combined quotient tests pass 21/21.

This closes the cheapest CPU uncertainty while the GPU lane resolves class capacity.
It does not authorize production quotient execution; the selected canonical program
and legal quotient run context are still missing.

## 07:44 cost-axis correction

The S1751/S1752 scripts price only trainable low-rank factors. Their 36 covered-token
tables contain 224.737M active values, versus 0.664M rank-8 factors. Rank-8 efficiency
therefore changes from 0.905 factor-only nat/Mreal to 0.002665 conditional
table-plus-factor nat/Mreal. The current post-forward hooks additionally allocate
2.084B dense table values, execute every native module, and fall back to native outputs
off support. These are program-class probes, not zero-native-call compressed models.
Full correction and machine-readable counts are in
`COMPILER_COST_CORRECTION_2026-08-28.md` and
`program_cost_audit_2026-08-28.json`.
