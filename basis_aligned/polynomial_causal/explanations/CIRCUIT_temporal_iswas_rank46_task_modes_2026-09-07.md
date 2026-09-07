# Temporal / is–was shared graph with task-typed response programs

## Status

This is an identified two-task causal interface, not yet a fully extracted whole-model
program.  A single physical source graph of 46 native components drives eight measured
response sites.  Inside that graph, temporal `will`/`had` and copular `is`/`was` use
different four-dimensional response programs.  The programs transfer in both interchange
directions and are selective, but their worst 10%-source-noise behavioral recovery is
`0.79708`, just below the frozen `0.80` adoption bar.  The eight-dimensional pooled
executor remains the noise-robust release.

## Physical support and interface

The greedy source graph contains 46 components:

`L0H3, L0H8, MLP0, L1H1, L1H7, MLP1, L2H2, L2H3, L2H6, L2H7,
L2H8, MLP2, L3H0, L3H4, L3H5, L3H6, MLP3, L4H0, L4H1, L4H3,
L4H4, L4H5, L4H6, L4H7, MLP4, L5H0, L5H1, L5H6, L5H7, L5H8,
MLP5, L6H1, L6H3, MLP6, L7H0, L7H7, L7H8, MLP7, L8H1, MLP8,
L9H1, L9H4, L9H7, MLP9, L10H5, MLP10`.

The measured response sites are `MLP1`, `MLP3`, `MLP4`, `MLP6`, `L8H1`,
`L9H1`, `L9H4`, and `L11H3`.  Greedy deletion removed `L1H3` and `L3H7`.
Removing the last candidate, `L10H5`, reduced the minimum bidirectional behavioral
projection to `0.799723`, so 46 is the frozen greedy boundary under the exact `0.80`
criterion.  This is empirical minimality under the registered deletion order, not a
global minimum proof.

## Causal behavior

The pooled eight-dimensional response projector is the robust executor.  Across three
independent 10%-cue-source-noise seeds and both interchange directions its minimum
behavioral projection is `0.800048`, minimum coordinate projection is `0.930818`,
maximum direction gap is `0.008122`, and no control top-1 token flips.

Task-conditioned modes split that shared physical interface.  Rank three is insufficient
(`0.799066` temporal and `0.779499` is–was).  Rank four passes clean forward execution
(`0.801335`, `0.804071`) and reverse execution (`0.800677`, `0.800228`).  Installing the
other task's rank-four program instead gives only `0.274508` temporal and `0.379984`
is–was recovery.  This cross-task causal failure is the main evidence that the same native
modules contain distinct task programs rather than one generic response axis.

With 10% source noise, rank-four coordinate recovery remains almost exact: the minimum
mean signed projection is `0.988757`, the worst site residual is `0.043865`, the maximum
direction gap is `0.007841`, and controls have zero top-1 flips.  Behavioral recovery,
however, ranges down to `0.797080`.  Therefore rank four is licensed as a clean,
bidirectional task-conditioned program, but not as the robust executor.  This near-threshold
miss is treated as a real null; the threshold is not relaxed post hoc.

## What the weights say

Exact MLP quadratic weight contractions and complete attention OV contractions agree with
the causal task split.  Across the four MLP response sites, the largest temporal/is–was
quadratic principal cosine is below `0.176`.  Across the four attention sites, the largest
complete-OV overlap is `0.626`; no site has a shared weight-function direction above the
registered `0.80` bar.  Thus the tasks reuse native locations and the source graph, but the
weight functions used inside those locations are task typed.

This is precisely where tensor translation helps: for a response basis `Q`, an attention
mode maps to explicit value-input and residual-output factors through `W_V^T Q` and
`W_O Q`; an MLP output covector `a = W_down^T Q` maps to the gauge-invariant quadratic
reader `S(a) = 1/2[L^T diag(a)R + R^T diag(a)L]`.  Ranking upstream writers and downstream
readers by their contractions with these task-specific objects can group cross-module pieces
that implement the same variable and split native modules that implement different ones.

## DAS interpretation

The constrained-DAS result is a target-mismatch and family-memorization warning, not proof
that optimization is intrinsically worse than difference in means (DIM).  Unregularized
constrained DAS had held-out full-vocabulary error `0.4485`; centered full-vocabulary KL
reduced it to `0.2445`, near DIM's `0.2385`, while tangent noise alone stayed at `0.4486`.
A better aligned objective reached `0.233426` on sealed A2, slightly better than DIM, but
traded away a registered target-sufficiency bar.  Complete-family selection then rejected
the learned rotations and retained the pooled step-zero/DIM-like estimator.

The failure mode is under-observation.  A scalar answer/complement loss only constrains
the contractions it observes; the optimizer can rotate inside their joint nullspace and
specialize to the fitted construction.  Complement inertness is necessary but not sufficient.
KL helps because it observes the full output distribution.  Noise can select a flatter point
inside an already feasible set, but it does not add missing task or reader constraints.

Accordingly, the next DAS-worthy object is a multi-environment finite causal operator with
whole construction and downstream-reader blocks held out.  DIM must remain an eligible
checkpoint; target extraction is a hard feasibility constraint; complement and KL choose
among feasible checkpoints.  A learned rotation graduates only if it beats DIM without
refitting on sealed task families and predicts through the exact weight-derived readers.

## Evidence ledger

- Minimal support: `temporal_five_mlp_rank47_pooled_greedy_rank46_deletion_v1_result.json`
  plus the hash-bound count correction, and
  `temporal_five_mlp_rank46_pooled_greedy_rank45_deletion_v1_result.json`.
- Robust pooled executor:
  `temporal_five_mlp_rank46_pooled_bidirectional_source_noise_ood_v1_result.json`.
- Task split and rank boundary:
  `temporal_iswas_rank46_task_typed_mode_causal_factorial_v1_result.json` and
  `temporal_iswas_rank46_task_mode_complete_rank_ladder_v1_result.json`.
- Reverse and noisy rank-four validation:
  `temporal_iswas_rank46_task_rank4_reverse_ood_v1_result.json` and
  `temporal_iswas_rank46_task_rank4_bidirectional_source_noise_v1_result.json`.
- Exact weight evidence: `temporal_iswas_mlp_quadratic_reader_overlap_v1_result.json`
  and `temporal_iswas_attention_ov_task_usage_v1_result.json`.

## Remaining gates

1. Test the frozen task-rank-four programs on a genuinely new capability-qualified lexical
   and construction bank, without refitting.
2. Produce the exact task-mode writer/reader atlas through upstream and downstream weights,
   then causally patch the highest-ranked edges and their complement.
3. Fit multi-environment DAS only against that richer operator and use an unopened family
   for model selection; do not reopen single-task scalar complement tuning.
4. Test joint composition when temporal and is–was commands are installed together.
5. Price an extracted executor only after these identification gates pass.
