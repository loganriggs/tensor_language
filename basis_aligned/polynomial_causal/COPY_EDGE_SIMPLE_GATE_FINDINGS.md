# Two cheap gates for the exact copy edge

Status: exploratory fit32/eval96 result on an exposed cache.

## Question and programs

The exact L8 H3/H4 copy edge writes the token code found after the nearest earlier
occurrence of the current token.  The shared $\lambda_8v_1$ code already preserves
95.9% of that edge's causal effect if we retain the native attention scalar.  This
experiment tested two cheap replacements for that scalar.

### Static matcher gate

Older work identified L2 H5 and L3 H8 as repeat matchers.  We ran only their fixed
query/key weight pipelines on normalized token embeddings, not on contextual residual
states.  For destination $p$ and its nearest earlier equal-token position $j$, the
one-number feature was

$$
s(p)=-a^{\mathrm{static}}_{2,5}(p,j)
     -a^{\mathrm{static}}_{3,8}(p,j).
$$

Two affine functions $\alpha_hs+\beta_h$ predicted the L8 H3/H4 scalar pair.  This
costs four fitted scalars, but it also retains 1,179,648 query/key projection-slice
values plus the embedding table.

### Distance-bin gate

This gate used only repeat distance $d=p-j$.  It stored two constants in each of four
bins: 1--8, 9--32, 33--64, and 65--128.  Its gate price is eight scalars and five
boundaries; no embedding table or live contextual state is needed.

Both gates used the already compiled shared payload and exact successor source
$j+1$.  Baseline arms were hash-pinned and reused from the preceding experiment;
only the three new gate arms were forwarded.  Total runtime was 30.6 seconds.

## Results

| Gate | Copy $\Delta$CE | Recovery vs deletion | Copy top-1 | All-scored $\Delta$CE |
|---|---:|---:|---:|---:|
| native scalar + shared payload | +0.00537 | 95.9% | 88.79% | -0.00003 |
| unconditional two constants | +0.09618 | 27.1% | 86.14% | +0.00532 |
| static matcher affine gate | +0.08083 | **38.7%** | 86.55% | +0.00415 |
| one-position-shifted matcher score | +0.11491 | 12.9% | 85.67% | +0.00595 |
| four distance bins | +0.09235 | **30.0%** | 86.21% | +0.00491 |
| delete exact edge | +0.13189 | 0.0% | 85.12% | +0.00720 |

The matcher gate's repeat-negative $\Delta$CE was `-0.00542` nat and its nonrepeat
effect was `-0.00066`; it did not create broad collateral.  Its causal improvement is
associated with the correctly aligned score because shifting the score by one
position removes most of the gain.

But the association is weak.  On held-out eligible positions, the affine matcher
predictor explains only `0.68%` of H3 scalar variance and `3.46%` of H4 scalar
variance.  Its 38.7% causal recovery misses the frozen 70% bar and improves over
unconditional constants by only 11.6 percentage points, below the required 20.

Distance bins explain negative held-out variance (`-3.07%`, `-1.77%`) and recover
only 30.0%.  They are essentially an expensive way to reproduce the unconditional
mean.  Both distance gates and direct reuse of the old static matcher are pruned as
faithful replacements.

## What the failure means

The old matcher score answers “is this token a repeat?”  That is useful but does not
answer the harder L8 question: “given the whole prefix, is what followed that repeat
a good next-token prediction now?”  Distance alone also does not answer it.

This narrows the mechanism rather than returning us to an unconstrained probe.  The
native L8 scalar is a product of two normalized bilinear forms:

$$
a_h(p,k)=
\frac{q_h(p)^\top k_h(k)}{128}
\frac{q'_h(p)^\top k'_h(k)}{128},
$$

where all four vectors are linear projections of contextual residual states followed
by head-wise RMS normalization and rotary position transforms.  The next high-return
move is therefore to compress this exact polynomial gate itself: find the smallest
shared/canonical subspaces of the two dot products that retain causal recovery, then
interpret those modes.  This directly exploits the tensor architecture and produces
a rank-versus-CE curve.  A generic lexical matcher or distance feature is no longer
the priority.

## Artifacts

- `COPY_EDGE_SIMPLE_GATE_PREREGISTRATION.md`
- `run_copy_edge_simple_gates.py`
- `copy_edge_simple_gate_results.json`
- `COPY_EDGE_CONSTANT_SCALAR_FINDINGS.md`

