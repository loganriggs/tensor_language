# Hourly strategic review — 2026-08-30 12:30 UTC

## Bottom line

The project has finally crossed the data-interface bottleneck: the exact 229-document
FIT-only signed causal-response tensor can be loaded lawfully without exposing the 114
internal-validation documents or EVAL. The GPU is free. The next scientific question
is therefore no longer “can we access the data?” but “does a shared-parent plus
owner-private tensor program predict these causal interactions at a useful price?”

No new fraction of bilin18 is explained in this review. The strict ledger remains:

- certified removable storage: 29,196,288 / 545,904,054 = **5.348245316%**;
- named deletion cross-entropy: 0.57968 / 5.30682 = **10.923302467%**;
- unexplained deletion cross-entropy: 4.72714 nat = **89.076697533%**; and
- circuits meeting the complete prediction/extraction/removal/OOD standard: **0/68**.

Those percentages measure different things. A parameter can be compressed without
its behavioral role being understood, and a named behavioral effect can be known
without a compact executable implementation. The conservative whole-model statement
is therefore still: about eleven percent of the measured deletion-CE gap has a named
causal explanation, while almost eighty-nine percent does not.

## Concrete work completed this hour

A resumable source-closed factor-grid lifecycle was implemented and hardened. The
production protocol is fixed to 17 rank pairs, three seeds, 2,000 Adam steps, learning
rate 0.03, and CUDA float32 optimization with canonical CPU-float64 replay. It cannot
accept caller-selected data, paths, ranks, seeds, device, or fitter.

The important computation is

$$
\widehat R_{pstd}
=\sum_{k=1}^{K_0}a_{pk}b_{sk}c_{tk}h_{dk}
+\sum_{g}\mathbf 1[s\in g]
\sum_{j=1}^{K_g}a^{(g)}_{pj}b^{(g)}_{sj}c^{(g)}_{tj}h^{(g)}_{dj}.
$$

Here $R_{pstd}$ is the measured signed effect of source intervention $s$ on target
$t$, in phase $p$, for document $d$. $K_0$ counts interaction patterns shared by all
six source owners. $K_g$ counts additional patterns private to owner $g$. The
document code $h_d$ tells us how strongly each pattern is present in one document.
This is a genuine multilinear tensor program; it contains no data-dependent top-k
router.

The runner records one immutable cell for each $(K_0,K_g,\text{seed})$. Before a cell
becomes visible, it replays factor shapes, exact owner topology, document codes,
canonical prediction, masked MSE, health, literal prices, phase errors, all 36
source-owner/target-owner errors, and the worst normalized owner-pair error. Only two
registered optimizer-nonfinite outcomes count as scientific failures. I/O, integrity,
protocol, or GPU-resource errors abort instead of masquerading as negative results.

The source-isolated evidence is four passing tests:

1. six fitted cells publish and resume exactly;
2. a planted numerical failure becomes a preserved failure cell;
3. changing a stored document code is caught by semantic replay; and
4. the synthetic API cannot be used to invoke a caller-selected published run.

This validates the experiment machinery, not the bilin18 hypothesis. Exact source
`fb9b14ab` is now in independent audit. No production fit starts without its GO.

## What “simplicity” means in this experiment

The two controlling prices remain separate:

$$
P=K_0(2+49+49)+\sum_gK_g(2+|S_g|+49)
$$

stored values that persist across documents, and

$$
C=K_0+\sum_gK_g
$$

numbers inferred per document. We also report execution cost $4802C$ multiply-adds
per prediction. At later calibration budget $m\in\{2,4,8,16\}$ physical source arms,
the program observes $49m$ cells and the frozen normal-equation cost proxy is

$$
49m\,C(C+1)+C^3.
$$

The original matched dense SVD control is mathematically rank zero at every structured
price because one dense basis vector already costs 4,802 persistent values and the
largest structured candidate costs 3,200. We preserve that negative fact. Amendment
12 adds total amortized storage $T=P+229C$ as a third, explicitly noncontrolling view;
it never replaces the $(P,C)$ order. The 4,802-value observation-wise mean is also
reported honestly at its larger persistent price.

These quantities are only candidate definitions of simplicity. They earn project
value only if lower price predicts held-out responses, composes across RMSNorm and
residual interfaces, yields stable/named atoms, and improves selective extraction or
removal and OOD transport. Local training MSE alone cannot validate them.

## Largest remaining gaps and confusing facts

1. **No real factor result yet.** The lawful tensor exists, but the 51 numerical fits
   await an independent grid-lifecycle GO. This is the exact current blocker.
2. **The strict dense comparator is degenerate.** It is rank zero under the original
   price, so the structure claim must rest on held-out global-only/private-only/joint
   dominance, separately labelled amortized controls, and downstream consequences.
3. **One geometry result is seed-fragile.** The a8 grouping generalized to learned
   directions pooled over seeds, but one of five seeds was null. A one-seed semantic
   story is not reliable.
4. **The downstream gate branch did not certify a circuit.** Its apparent roughly
   sixfold advantage collapsed on a fresh window; oracle gain mostly tracked base text
   CE, while the random denominator tracked assembly excess. More gate-width sweeps
   are pruned.
5. **Composition remains missing.** Even a held-out response program would explain a
   library of measured source-to-target interactions, not yet the MLP0/MLP1/MLP2
   computation through RMSNorm, residual addition, and downstream logits.
6. **No terminal edit exists.** No learned factor has yet predicted a new intervention,
   enabled selective removal with low collateral damage, or transported OOD.

## Candidate actions considered and pruned

Another local activation SAE, raw HOSVD, wider downstream probe, or new hand-picked
gate has low expected information gain: each duplicates completed work or optimizes a
noncomposable local metric. Global full-model tensor factorization is too expensive and
too gauge-confounded before the small causal interface is understood. A top-k router
is not treated as a tensor program. The expired 2026-08-29 eight-hour entry-point plan
is historical evidence, not an active deadline.

## Ranked next five actions

1. **Complete the exact independent audit and, on GO, run the 51 FIT-only cells.**
   Highest information gain, directly tests tensor/shared-owner structure, fully
   falsifiable, already paid data, roughly 3.9 GPU-hours, and crash-resumable.
2. **Analyze the complete training grid without selecting a winner.** Report seed
   health/range, phase and owner-pair failures, the $(P,C)$ Pareto set, global-only vs
   private-only vs joint comparisons, strict rank-zero control, mean, and separately
   labelled amortized dense ranks. This is cheap CPU work and prevents pooled MSE from
   hiding a failed interface.
3. **Freeze all healthy nondominated programs, then open the existing 114-document
   validation role through a new audited adapter.** Score unconditional transport and
   calibration at 2/4/8/16 physical arms. This is the first real test that the program
   predicts unseen document interactions rather than memorizing training codes.
4. **Apply structural certificates only to validation survivors.** Use tree-cut ranks,
   quotient-Jacobian gauge/nullity, sparse document use, and stability. These math
   tools can distinguish a minimal hierarchy from a merely low-MSE factorization, but
   are wasted on candidates that do not transport.
5. **Convert surviving atoms into causal whole-model tests.** Predict fresh amplitude
   and direction interventions, compose across early MLP/RMSNorm/residual interfaces,
   extract the named behavior, remove it with unrelated-target controls, and evaluate
   a second domain. Only this step can promote a response factor to a terminal circuit.

The highest-priority safe action was executed as far as currently authorized: the
runner produced concrete synthetic receipts, survived tamper/resume tests, was
published, and entered independent audit. The real 51-cell run remains deliberately
blocked until that audit supplies exact GO.

## 12:56 UTC launch update — six real cells preserved, moving-HEAD lifecycle repaired

The independent audit returned exact GO at `ace91e82`: 99 synthetic tests passed and
the three interrupted-resume attacks all failed closed. After a pre-`main()` import
path failure that exposed no data, the production grid launched with the repository
root on `PYTHONPATH` and completed six real FIT-only cells:

| rank | median training MSE | MSE range | median initialization improvement | median worst owner-pair NRMSE |
|---:|---:|---:|---:|---:|
| 1 | 0.041553247 | 0.041553234–0.041553258 | 9.870% | 2.9802 |
| 2 | 0.039748801 | 0.039748798–0.039750116 | 13.987% | 2.8994 |

All three seeds at each rank are healthy and final-MSE seed spread is very small. The
large worst-owner-pair errors warn that these tiny global-only programs are not uniform
interface explanations. They are not a frontier or hierarchy result because 45 cells
remain absent.

The process was deliberately interrupted when a lifecycle problem became clear. Its
logical source identity included the repository's current `HEAD`; unrelated agent
commits would therefore make unchanged audited source bytes fail final replay and make
completed cells non-resumable. Continuing would knowingly spend GPU time on a
nonterminal transaction. The six cells and precise failure receipt are preserved in
`causal_response_factorization_v1_grid_results_interrupted_moving_head_20260830T1256Z`.

Amendment 13 replaces moving `HEAD` with a stable identity: audited source commit,
independent-audit artifact hash, and exact audited path hashes. Current bytes must
still match the audit; the historical commit and audit blob must still be published;
any source drift fails. Unrelated commits no longer change the identity. Exact source
`4d7cb379` is in a fresh independent audit and the canonical namespace is pristine.
