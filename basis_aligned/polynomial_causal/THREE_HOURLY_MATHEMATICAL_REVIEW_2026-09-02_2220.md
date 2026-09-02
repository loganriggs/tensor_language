# Three-hour mathematical review — 2026-09-02 22:20 UTC

## Exact object

At MLP10, a token state is `z[b,t,i]` with residual index `i=1..1152`. The bilinear MLP has hidden index
`h=1..4608` and output index `o=1..1152`:

`y[b,t,o] = sum_h D[o,h] (sum_i L[h,i] z[b,t,i]) (sum_j R[h,j] z[b,t,j]) + bias[o]`.

Equivalently its ordered weight tensor is

`T[o,i,j] = sum_h D[o,h] L[h,i] R[h,j]`.

This local map has polynomial degree two. The complete network is not a polynomial: every block includes RMS
normalization, which divides by the square root of a quadratic form, and the final logits use a tanh soft cap.
Attention's unnormalized numerator is a tensor contraction, but the full suffix measured here includes those
normalizations and therefore must be tested by finite execution rather than polynomial coefficient comparison alone.

The exact MLP10 input split supplies 22 named sources `s`: embedding, attention0--10, and MLP0--9. Projecting each
source through `L` and `R` gives `l_s[b,t,h]` and `r_s[b,t,h]`. The 253 unordered exact terms are

`Y_(s,s) = D(l_s * r_s)` and

`Y_(s,u) = D(l_s * r_u + l_u * r_s)` for `s != u`.

Rung509 measures a finite causal response tensor `C[a,p,c]`, with score implementation `a=1..4`, exact term
`p=1..253`, and downstream coordinate `c=1..34` (four copy contexts plus 30 circuit member-minus-control effects).
Its proposed eight-atom assignment is `G[a,p,k]`, `k=1..8`, with `sum_k G=1`. A physical atom removes
`sum_p G[a,p,k] deltaY[a,p]` before exactly recomputing layers11--17.

Allowed inputs are the frozen natural documents500:748 for discovery and752:1000 for confirmation; documents748:752
are unused. Outputs to preserve are the signed finite cross-entropy effects in the34 coordinates, their document-half
direction, all-copy selectivity, and predictable joint removals. Approximation is measured in discovery-scaled
response mean squared error only as a screen; identification requires finite held-out interventions. Maximum price is
145,328 model forwards, six CPU fits, zero model backwards, and zero deployed parameter changes.

## Symmetries and what existing uniqueness theorems do not give us

The local ordered tensor is a CP decomposition with 4,608 triplets `(D[:,h],L[h,:],R[h,:])`. Kruskal's classical
condition gives essential uniqueness when the three factor k-ranks sum to at least `2r+2` [Kruskal 1977](https://doi.org/10.1016/0024-3795(77)90069-6).
Here each factor has only1,152 rows, so the sum of k-ranks is at most3,456, far below9,218 for `r=4,608`.
The theorem cannot identify the native bilinear units. In addition, evaluating both input slots on the same `z`
exposes only the input-symmetrized tensor; swapping Left and Right per term is a real gauge, and scale can move among
the three factors while preserving their product.

Tensor-power guarantees for latent-variable models require specially whitened, usually orthogonally decomposable
observable moments [Anandkumar et al. 2014](https://www.jmlr.org/papers/v15/anandkumar14b.html). Our finite loss
responses are dependent outputs of one nonlinear suffix, not conditionally independent views or orthogonal moments.
The latent-class identifiability arguments of [Allman, Matias, and Rhodes 2009](https://arxiv.org/abs/0809.5032)
likewise require conditional-independence structure that our34 coordinates do not have. These are useful boundary
theorems, not direct solvers.

The rung509 soft mixture has an additional algebraic problem. With freely learned response atoms `W`,
`C approximately equals G W` admits changes of latent basis that preserve the product while changing the purported
source groups. The Left/Right softmax removes scale from each row of `G`, but it does not make the factorization unique.
Restart agreement is empirical, not a theorem.

## Executed falsifier: stable optimization can recover the wrong atoms

I generated an exact synthetic `4x253x34` response tensor from the registered eight-atom model, then ran all six
registered fits. Runtime was10.81 seconds. Seven atoms passed the proposed restart/half stability gates; minimum
restart response cosine was`.990` and minimum assignment cosine was`.99996`. Nevertheless, after optimal atom
matching, only two of eight mean learned response atoms matched their planted counterparts (cosines`.9995/.9985`);
the others ranged from`-.115` to`.246`, and assignment mean-squared error was`.0613`.

This is the dangerous case: the algorithm is highly repeatable because its optimization bias is repeatable, while
the latent explanation is wrong. Therefore the existing restart and split gates cannot establish stable
identification. Running the model with that instrument would risk Goodharting exactly the user's simplicity warning.

## Archetypal/convex-hull constraint

Archetypal analysis constrains each atom to a convex combination of observed data points [Cutler and Breiman 1994](https://digicoll.lib.berkeley.edu/record/85980/files/379.pdf).
The Archetypal SAE paper applies this idea to dictionary atoms and reports improved plausibility and stability
[Fel et al. 2025](https://arxiv.org/abs/2502.12892). This is principled for our response tensor only under an explicit
geometric assumption: each true causal variable must appear approximately as an extreme observed exact-term response.
It does not follow merely because convex constraints are convenient.

There is a relevant theorem-level special case. Separable nonnegative factorization is recoverable when each latent
component has an observed anchor [Arora, Ge, Kannan, and Moitra 2012](https://arxiv.org/abs/1111.0952). Our responses
are signed, so that theorem does not directly apply, but its anchor condition explains what the convex hull buys:
atoms become tied to actual measured terms instead of floating under arbitrary latent changes of basis.

A 2026 reanalysis reports that Archetypal SAE stability can be an artifact of shared initialization and metric design
[Brzozowski and Chung 2026](https://arxiv.org/abs/2606.02061). That directly supports our requirement for genuinely
independent initializations and a planted recovery test; endpoint agreement alone is not evidence.

## Decision and executable consequence

Do not run the current free-response dictionary. Before any CUDA outcome:

1. constrain each response atom to the convex hull of the measured discovery responses;
2. require each atom to have a distinct observed anchor carrying at least90% of its convex weight;
3. add a synthetic separable-response test with known anchors and require recovery of both assignments and atoms,
   not merely restart agreement; and
4. keep every registered held-out finite response, physical removal, composition, and source-changing test.

This is not adoption of Archetypal SAE as a generic answer. It is a falsifiable restriction: if real data lack stable
anchors, the archetypal dictionary is not identifiable and the registered next route becomes the observable
downstream predictive-state quotient, with no latent atom claim. This repair has higher information value than
spending up to145,328 forwards on an instrument already falsified by a ground-truth toy.
