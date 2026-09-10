# Returning to the bilinear products: what the math now says

We still have not recovered an is/was circuit satisfying all four requested
properties. The latest experiments explain why several shortcuts fail, but
that is weaker progress than discovering a reusable computation. I am closing
the response-approximation series and returning to the actual products in the
bilinear weights, following the original handoff and pilot.

The important new practical result is a tested way to interchange or remove
any chosen set of bilinear products exactly, while letting the rest of the
model respond normally. I also derived and tested a weight-tensor comparison
that avoids constructing a huge tensor. Neither tool has yet identified a
shared feature in the trained model.

## What happened in the latest experiment

Earlier tests found that directly folding MLP4's output weights into attention
layer9's value weights missed most of that partial path's effect. Both attention
and MLP computations between the layers mattered. Approximating their response
by the derivative at the receiving input also failed.

The latest test retained the actual changed normalization and simplified only
the products inside those intervening modules. For a product of two factors,
we kept the original product and each factor's separate change, but omitted the
product of the two changes. Attention uses five projected factors; we applied
the corresponding first-order rule there too. Normalization rescales vectors
by their root-mean-square magnitude; those rescalings were recomputed exactly
as in the changed native execution.

The test still failed: **33–51% relative error in the full output-effect vector**,
against a10% preliminary screen threshold. All eight combinations of task,
panel and swap direction missed. Here the effect vector is the change in all
50,304 centered vocabulary logits after the intervention. This measures the
whole output change, not just the is-minus-was score. The instrument controls
passed. The managed GPU run used48forwards on864sequence instances in2.82s.

This rejects that approximation on the existing partial value path. It does
not rule out a simpler exact operation, a different circuit boundary, or a
nonlinear decomposition. No new unseen task-family evidence was established.

## The bilinear layer already supplies an explicit product program

Let n be the normalized1152-dimensional input to MLP4. It has4608 hidden
products. For product j, let l_j and r_j be its two learned input weight rows,
and w_j its1152-dimensional output weight column. Then

    phi_j(n) = (l_j · n)(r_j · n)
    m(n) = sum_j w_j phi_j(n) + bias.

Each term measures two input quantities, multiplies them, and writes the result
in a particular direction. This is the weight-tensor viewpoint used in
[Pearce and colleagues' bilinear-MLP paper](https://arxiv.org/abs/2410.08417).
It gives explicit candidate operations; naming their semantic meaning still
requires evidence from inputs and interventions.

For a proposed subset S, a paired source interchange is

    delta_m_S = sum_(j in S) w_j [phi_j(n_donor) - phi_j(n_base)].

Add this to the receiving MLP output, then run the complete downstream model.
This edits just the chosen product contributions. It does not approximate the
later attention or MLP response. Setting the same columns of Down to zero
instead removes those products on every input; the output bias remains.
A swap requires a donor; a static removal does not. Their success is separate.

The implemented checks verify full-set interchange, disjoint subset addition,
and agreement between output subtraction and static weight removal. Numerical
errors are below3e-14 on the small FP64 controls. These are algebra checks,
not trained-model successes.

## Sharing must survive cancellation and changes of parameter scale

Two tasks using the same native module does not tell us whether they reuse an
operation. A common product subset would be a proposal for sharing. Different
subsets could explain how the module splits between tasks. Neither conclusion
follows from overlap in large activations alone.

For example, suppose three product writes on an input are +100, -100 and +1.
Their sum is +1. Ranking squared sizes selects the first two, although the
third alone reproduces the output. This exact counterexample passes in the
new control file. Even ranking each write's signed alignment with the total
returns100,-100,1 and does not automatically select the simpler support.
Thus neither energy ranking nor its signed correction identifies a circuit.

The signed score is still a useful diagnostic:

    score_j = average [delta_phi_j * (w_j · delta_m)].

It obeys sum_j score_j = average ||delta_m||². Negative entries reveal
cancellation in the local write. They are not negative behavioral importance.
Both the edit and score stay unchanged when l_j and r_j are rescaled by a and b,
while w_j is divided by ab. Product labels also permute consistently. This
prevents arbitrary parameter scale from determining the conclusion, but does
not establish uniqueness under every possible tensor factorization.

## Compare tensor terms without building the full tensor

On ordinary tied-input execution, define

    T_j = w_j outer (l_j r_j^T + r_j l_j^T)/2.

Contracting its two input indices with n reproduces that product's write.
The inner product of two such coefficient tensors is

    G_ij = (w_i · w_j)/2 *
           [(l_i · l_j)(r_i · r_j) + (l_i · r_j)(r_i · l_j)].

G is a Gram matrix: it stores pairwise coefficient inner products. For any
coefficient vector c, ||sum_j c_j T_j||² = c^T G c. This detects cancellation
and similarity of whole product tensors, using matrix products rather than
constructing every1152-by1152-by1152 coefficient. The small dense-tensor oracle
agrees within2.85e-14. A full4608-square Gram would itself contain21,233,664
entries, about162MiB in FP64; it is not free, and was not allocated on the
trained model in this work. Subset comparisons can use smaller blocks.

This is coefficient geometry, not a behavioral distance. Normalization,
reachable inputs and downstream readers still matter. Symmetrizing the two
inputs also describes ordinary tied-input execution; it does not erase the
Left/Right distinction if independently editing those operands is part of the
claimed interface. The previous operand-intervention failure remains binding.

## What this contributes to the four properties

| Property | Contribution now | Still required |
|---|---|---|
| OOD prediction | Explicit fixed product formula, no response fit | Freeze a proposed support and test new constructions |
| Extraction | Executable subset with a defined normalized-input interface | Identify its input computation and account for native background |
| Removal/interchange | Exact implementations and algebra controls | Show selective native behavioral effects and unrelated-task preservation |
| Composition/reuse | Shared products can be evaluated once; joint edits are explicit | Demonstrate two consumers reuse them and predict joint native effects |

Additive source writes do not imply additive final behavior. In the toy suffix
F(z)=z², edits u and v have a joint effect exceeding the sum of separate effects
by2uv. The executor must evaluate their combined input through the suffix.
This is why the next causal test must keep the downstream computation active.

The next scientific decision is which product support, if any, has a stable
common role across has/had and is/was. A simple energy-only selection is not
licensed by these calculations. No native subset or budget has been selected
from the results, and no successor GPU job is queued yet. The tested subset
tool is the concrete continuation already completed. All545,902,902 native
parameters remain charged, with zero demonstrated structural savings.

## Files and evidence

- [Latest native result](../../BILIN18_MLP4_NORM_PRESERVING_RESPONSE_V1_RESULT.json)
- [Exact subset and tensor-Gram code](../../bilinear_product_subsets.py)
- [Executed algebra controls](../../BILINEAR_PRODUCT_SUBSETS_V1_CONTROLS.json)
- [Hourly review and direction decision](../../HOURLY_STRATEGIC_REVIEW_2026-09-10_0314.md)
- [Original handoff and updated success criterion](../bilinear_circuit_reconstruction_codex_handoff.md)
- [Original pilot report](../bilinear_reconstruction_pilot_report.md)
