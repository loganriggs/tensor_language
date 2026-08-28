# Hybrid tensor-class oracle preregistration

Date: 2026-08-28 07:36 UTC

Status: frozen discovery-only protocol; not queued and not executed. It grants no new
row, final-role, selection, or promotion authority.

## Question

S1751 shows rank 32 and 128 tie within the all-36
`table[token] + current_state @ rank-r` family. Determine whether the residual program
class gap is caused primarily by deleting the native squared-attention contraction,
deleting the native bilinear MLP contraction, or superadditive interaction between the
two deletions.

This does not propose native halves as a simple endpoint. It is a causal class oracle
which chooses which typed contraction must be simplified next.

## Arms

All arms use the same frozen model, fit rows, coverage mask, rank 8, bottom-up
initialization, final-CE optimization, and checkpoint policy.

1. `both_compiled`: compile all 18 attention and all 18 MLP output writes.
2. `attention_native`: attention remains exactly native; compile all 18 MLP writes.
3. `mlp_native`: MLP remains exactly native; compile all 18 attention writes.
4. `both_native`: exact live-model control, no fitted parameter.

Each compiled site is `table[token] + (xA)B`. Tables are frozen from the fit rows;
rank-8 factors alone train on final CE. Uncovered tokens remain native. This diagnostic
prices both factor reals and active covered-table reals separately; native parameters
are reported rather than called free.

## Roles and optimization

- fit/training: `fineweb_n96_skip80.pt`;
- checkpoint selection: `fineweb_n192_skip7000.pt` every 30 steps;
- untouched report: `fineweb_n192_skip11000.pt` at the selected step;
- 180 Adam steps, batch 4, learning rate $10^{-3}$;
- score covered target positions 64:256 only;
- all three fitted arms use the identical sampled fit-row sequence.

Both evaluation roles are already spent for this program arc. Results are discovery
only and cannot move strict executable or OOD accounting.

## Frozen attribution

For held-out CE $c_A$ and live CE $c_0$, define harm $h_A=c_A-c_0$. Then

$$
G_{\rm attn}=h_{CC}-h_{OC},\qquad
G_{\rm mlp}=h_{CC}-h_{CO},\qquad
I=h_{CC}-h_{OC}-h_{CO},
$$

where $OC$ is `attention_native` and $CO$ is `mlp_native`. Positive $I$ means the two
compiled halves amplify one another's errors; negative $I$ means redundancy.

## Registered predictions and decision

1. `pred_a_attention_dominates`: $G_{\rm attn}>G_{\rm mlp}$. If false, prioritize the
   bilinear MLP grammar rather than attention.
2. `pred_b_each_native_half_helps`: both hybrid arms have lower held-out CE than
   `both_compiled`. Failure identifies a compensatory compiled half and forbids naive
   independent replacement.
3. `pred_c_superadditive_harm`: $I\ge0.10$ nat. Failure says missing primitives add
   approximately independently or redundantly.
4. `pred_d_controls`: `both_native` reproduces live CE, `both_compiled` initialization
   reproduces S1748 within 0.002 nat recovery, its selected recovery reproduces S1750's
   peak within 0.03 nat, table-only/live/coverage controls hold, and every fitted site
   sees all 24,576 fit positions.

The winner is the larger of $G_{\rm attn}$ and $G_{\rm mlp}$, with a 0.01-nat tie
band. This decision is descriptive and independent of predictions 2--4. If either
hybrid is worse than `both_compiled`, the next program must be trained jointly and the
simple winner rule is suspended.
