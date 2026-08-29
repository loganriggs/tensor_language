# What remains in the exact copy edge?

Status: exploratory result on an exposed cache.  The fit and evaluation documents
are disjoint, but this is not fresh-data confirmation.

## Short answer

We can now separate the L8 copy edge into two parts:

1. **What to write:** an almost static token code.  Replacing the live/contextual
   value by the shared $\lambda_8v_1$ code preserves **95.9%** of the edge's causal
   effect on natural copy positions.
2. **How strongly to write it:** a context-dependent scalar.  One weak pair of
   constants averaged over all repeats preserves only **27.1%**.  A stronger pair
   averaged over repeats where copying is correct preserves **81.4%** on disjoint
   documents, but it is less faithful on the full distribution.

So the payload is nearly compiled.  The main unsolved local computation is a gate:
**when is an earlier occurrence's successor actually a good continuation here?**

## The computation, with every object defined

Let $p$ be the current token position.  Search backward up to 128 tokens for the
nearest earlier position $j(p)$ containing the same token.  The candidate token to
copy is stored at source position

$$
k(p)=j(p)+1.
$$

This source rule uses observed input token IDs only.  It does not inspect the future
target.

For attention head $h$, the native contribution of that one source edge is

$$
w_h(p)=a_h(p,k(p))P_h
\left((1-\lambda_8)v_{8,h}(k(p))+\lambda_8v_{1,h}(k(p))\right).
$$

The terms are:

- $a_h(p,k)$: the scalar attention-pattern value produced from L8 query/key
  computations for destination $p$ and source $k$;
- $v_{8,h}(k)$: the fresh L8 value vector at source $k$, which depends on the current
  contextual residual stream;
- $v_{1,h}(k)$: the value vector first created at L0 and carried as the shared value
  bus through later layers;
- $\lambda_8$: the model's learned scalar mixing coefficient at L8;
- $P_h$: the fixed columns of L8's output projection belonging to head $h$.

The exact H3/H4 edge is $w_3(p)+w_4(p)$.  The proposed small replacement is

$$
\widetilde w(p)=
\sum_{h\in\{3,4\}}c_hP_h\left(\lambda_8v_{1,h}(k(p))\right),
$$

where $c_3,c_4$ are just two stored numbers.  Physically, the experiment subtracts
the native one-edge write and adds this replacement.  Every other source edge, head,
layer, residual term, and MLP remains native.

## Data separation

- Fit: cached documents 1--32.
- Evaluation: cached documents 33--128.
- Evaluation sample: 96 documents, 1,472 natural copy-positive positions, 6,559
  repeat-negative positions, and 10,401 nonrepeat scored positions.
- Fit coefficients are frozen before any evaluation forward pass.

All rows belong to an already exposed selection cache, so the exercise is exploratory.
The preregistration and runner hashes stored in the result match the files on disk.

## What constants were learned?

Head order is H3, H4.

| Fit rule | $c_3$ | $c_4$ |
|---|---:|---:|
| mean over all input-eligible repeats | -0.01779 | +0.02377 |
| mean over fit repeats where copying is correct | -0.05832 | +0.07375 |
| older synthetic-repeat constants | -0.11900 | +0.19000 |

The native scalar distribution explains the difference.  Across all fit repeats,
the medians were only `-0.00254` and `+0.00452`.  On copy-positive repeats the means
grew by about threefold.  The model generally writes weakly on the undifferentiated
repeat population and much more strongly on contexts compatible with copying.

## Results

Recovery is measured against exact edge deletion:

$$
R=1-\frac{\Delta\mathrm{CE}_{\mathrm{replacement}}}
{\Delta\mathrm{CE}_{\mathrm{edge\ deletion}}}.
$$

Here $R=100\%$ means native-equivalent copy-positive CE, while $R=0$ means the
replacement is no better than deleting the edge.

| Arm | Copy $\Delta$CE | Copy recovery | Repeat-negative $\Delta$CE | All scored $\Delta$CE |
|---|---:|---:|---:|---:|
| delete exact edge | +0.13189 | 0.0% | -0.00905 | +0.00720 |
| native scalar + shared $\lambda v_1$ payload | +0.00537 | **95.9%** | -0.00137 | -0.00003 |
| all-repeat constants + native mixed value | +0.09384 | 28.9% | -0.00526 | +0.00545 |
| all-repeat constants + shared payload | +0.09618 | 27.1% | -0.00541 | +0.00532 |
| positive-fit constants + shared payload | +0.02455 | **81.4%** | +0.00457 | +0.00334 |
| historical constants + shared payload | -0.07209 | 154.7% | +0.03232 | +0.00546 |
| wrong source + all-repeat constants | +0.13253 | -0.5% | -0.00848 | +0.00729 |

Nonrepeat propagated effects were below `0.0008` nat in magnitude for every constant
arm.  The wrong-source control recovered nothing, confirming that the successor
source $j+1$, not merely adding a generic head vector, is essential.

The preregistered primary arm was the all-repeat, target-free fit.  It failed the 70%
recovery gate, and its scalar-only version failed the 85% gate.  The experiment
therefore rejects **one unconditional constant per head as a faithful replacement**.

The positive-fit arm is still useful.  Although its coefficients were estimated by
selecting positive examples in the fit split, the frozen evaluation program does not
read evaluation targets; it applies the same two constants at every repeat.  It
recovers 81.4% of copy CE with small collateral, showing that a very small payload
program is viable if paired with a better input-side gate.  Copy-position top-1
accuracy moved from `88.86%` native to `87.84%` for this arm, so it is not yet a
top-1-faithful substitute.

The historical constants make the purpose tradeoff especially clear.  They improve
copy-position CE by `0.07209` nat and top-1 accuracy to `91.03%`, but damage
repeat-negative CE by `0.03232` nat and worsen aggregate CE.  They are a plausible
aggressive **copy circuit extractor**, not a faithful whole-model replacement.

## What this changes about the plan

The value-side decomposition is no longer the bottleneck.  More SAE work on the L8
payload is low priority because a fixed shared code already preserves 95.9% of the
causal effect.

The next high-return target is the scalar gate.  Candidate replacements should be
judged by whether they distinguish copy-positive from repeat-negative contexts, not
by local attention-scalar MSE alone.  In priority order:

1. Test whether the already known weights-computed matcher score, with one affine
   calibration per head, predicts the L8 edge scalar and preserves CE on the disjoint
   split.  This reuses an older successful stand-in rather than inventing a new probe.
2. As a cheaper falsifier, test a small distance-binned table.  If repeat distance
   explains the strength, 8--16 scalars may suffice; if not, stop that branch quickly.
3. If neither works, fit a sparse, low-degree gate from a small set of input-side
   state variables and charge every variable, multiplication, and stored coefficient.
4. Confirm the winning gate on fresh natural text and then compose it with the older
   upstream matcher reduction.  Only that composition would be a standalone local
   copy program.

The mathematical object is now a conditional key--value lookup: a discrete
equality/successor selector, a low-complexity contextual gate, and a fixed linear
token-code writer.  This is much closer to a small executable program than a list of
heads, while the failed unconditional constant tells us exactly which conditional
part remains unexplained.

## Artifacts

- Preregistration: `COPY_EDGE_CONSTANT_SCALAR_PREREGISTRATION.md`
- Runner: `run_copy_edge_constant_scalar.py`
- Numerical result: `copy_edge_constant_scalar_results.json`
- Exact edge baseline: `COPY_SOURCE_EDGE_DISCOVERY_FINDINGS.md`

