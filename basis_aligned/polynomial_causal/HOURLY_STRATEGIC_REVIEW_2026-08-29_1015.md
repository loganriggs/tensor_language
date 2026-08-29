# Hourly strategic review — 2026-08-29 10:15 UTC

## What actually changed

Two things completed. Only one is a model measurement.

1. **S1915 is a measured statistical diagnostic.** With eight independent seeds, the
   permutation-null means at 8 versus 64 permutations differ by only
   `0.020/0.007/0.008` across the three discovery roles. Across-seed spreads fall from
   `0.218/0.334/0.188` to `0.061/0.123/0.096`. The reduction is not exactly the frozen
   factor-of-two bar on every role, but the cached token-margin advantage over the
   live-model margin survives at the higher precision. Thus S1914's ranking is robust;
   it is still a reliability diagnostic, not a causal circuit or removal result.
2. **The E4 physical dispatcher core is implemented, not measured.** It binds all
   eight frozen copy candidates to exact physical heads and computes

   $$
   w_{a}=w_{\mathrm{native}}-sum_{h\in H_a}w_h+
   \sum_{h\in H_a}\mu_h(p).
   $$

   Here $H_a$ is candidate $a$'s head set and $\mu_h(p)$ is that head's fit-role mean
   write at position $p$. L8H3+L8H4 is dispatched in one layer transaction, while the
   four-head and late-pair candidates intervene sequentially at their registered
   layers on their own live counterfactual states. The shared first-value attention bus
   is preserved. The combined dispatcher, adapter, scientific-contract, and streaming-
   statistics suite passes `35/35` CPU tests.

The dispatcher opened no rows, checkpoint, model, or outcome. It therefore does not
complete E4.1, E4.2, or E4.3.

## Honest fraction explained

The strict balance sheet is unchanged:

- structural interception: **36/36 sites**, which is access rather than explanation;
- removal-certified storage: **29,196,288 / 545,904,054 = 5.3482453%**;
- named causal CE: **0.57968 / 5.30682 = 10.9233025%**;
- unexplained named-CE residual: **4.72714 nat = 89.0766975%**; and
- extraction/removal/OOD terminal actions: **0/68**.

Family F is a receipt-complete negative under its local-fit gate. E1 is closed by two
large prediction failures. E2's universal/typed and large shared-private bases are
negative, although tight rank 64/128 sharing has a bounded compression signal. E3's
rank-64 destination oracle, direct map, and composed map are all inadequate. E4 has
good rows and engineering prerequisites but zero behavioral model outcomes.

## Largest missing interfaces and confusing evidence

1. **E4 execution lifecycle.** Missing are the fit-role per-position head-mean
   collector and receipt, exact outer/native/per-layer call and support ledgers,
   source-closed production owner, explicit authority keeping every MLP native, and
   the create-only selection/result/manifest/receipt chain. The prior checkpoint test
   is deliberately nonauthorizing. Its separately accumulated bfloat16 head sum differs
   from the unpartitioned native contraction by relative `0.002627--0.002667`; the
   dispatcher therefore uses the exact unpartitioned native write as its base and
   reports the subtraction discrepancy rather than calling the head sum bit-identical.
2. **Native-Down anomaly.** Native Down has worse local NRMSE but better suffix KL
   (`0.05772` versus refitted `0.08476`). This could be causal alignment or
   compensation. The corrected consequence-Gram design must use an independent test
   sketch, a frozen response metric, and common-background interventions before it is
   executable.
3. **Whole-program composition.** The closed linear stream map loses
   `1.08978--1.27276` nat and self-refitting loses about `5.5` nat. A local fit does not
   currently predict recursive composition.
4. **Low-dimensional state.** At E3 rank 64, even projecting the true destination
   response has error `0.2709`; direct and chained prediction score `0.4861/0.4520`.
5. **Reliability versus explanation.** Cached token confidence is now a robust and
   effectively free ranking signal, but native fallback explains neither rejected
   tokens nor the mechanism and earns no strict removal credit.

## Ranked top five after pruning

### 1. Finish the E4 production owner and run the eight-candidate copy screen

This has the best combination of causal relevance, falsifiability, OOD transfer, and
short path to a first terminal action. Rows, labels, head formulas, candidate grammar,
dispatcher core, and simultaneous document bootstrap exist. Complete the fit-mean and
receipt lifecycle before using the now-free GPU. If no candidate passes selection,
close copy localization without opening final/OOD.

### 2. Build the corrected native-Down consequence assay

Use JVP range sketch $Y=J\Omega_{\rm fit}$, QR, and VJPs for $B=Q^\top J$, then test
the missed range on a disjoint $\Omega_{\rm test}$. Freeze a response metric $W$ and
analyze $J^\top WJ$ so 1,152-dimensional states do not dominate log-odds merely by
units. Add a common native background before comparing native, refit, permuted, and
random physical columns. This directly targets the most informative Family-F anomaly.

### 3. Replace the full-logit hybrid API with a streaming telescope pilot

The exact 37-arm identity is sound, but materializing
`[37, documents, 256, vocabulary]` is infeasible. Stream one microbatch and adjacent
pair at a time, accumulating exact scalar CE increments and per-token logit-infinity
bounds. Bind every cut and call ledger. Label the first result as a certificate for the
currently hooked program, not an autonomous program, because the current build retains
native calls and the native first-value bus.

### 4. Calibrate the cached-token selective compiler

S1915 makes this statistically credible and cheap. Use document-level labeled errors
and the simultaneous risk bound already implemented. It can establish useful selective
prediction at a literal price, but native deferral must be counted and it cannot earn
whole-model removal credit.

### 5. Test one native-free nonlinear fallback on uncovered current tokens

This is the main route to closing the largest practical prediction gap after E1. It
must be nonlinear or attention-conditioned, operate only where tables are not exact,
and be evaluated recursively. It is expensive, so it follows the more discriminating
E4 and native-Down assays.

Tight q64/q128 shared-private allocation is sixth: it is cheap and may improve storage,
but E2 makes it more redundant and less causally informative than the five moves above.

## Pruning and conditional mathematics

- Do not revive global rank-512 linear closure, large shared bases, local-MSE Down
  refits, generic HOSVD/CP, or normalized-energy allocation.
- The proposed 64-mask E4 tensor-train/Nerode experiment remains conditional on an E4
  passer. With aggregate CE/log-odds outputs it is a finite behavioral response model,
  not by itself an executable extracted circuit. Use all 64 masks with a frozen split
  and restrict the first fit to ranks 1/2 unless identifiability assumptions are added.
- The corrected consequence Gram may reveal a causally sufficient product subspace,
  but 16 fit probes cap the visible rank at 16; only an independent test sketch can
  measure missed range.
- MDL, information bottlenecks, and canonical gauges remain comparison/canonicalization
  tools after a causal program exists, not mechanisms for creating one.

## Safe action executed this hour

`terminal_copy_attention_dispatcher.py` and its tests were implemented and committed
with the concurrent S1915 sweep. The launch contract now requires the dispatcher as a
separate binding. This removes the candidate-to-physical-write interface blocker.

The exact next blocker is the fit-role head-mean authority/collector plus production
call/support ledger and receipt lifecycle. The GPU is free after S1915, but launching
E4 before those objects exist would spend the prospective roles without a replayable
transaction, so no model run was started.
