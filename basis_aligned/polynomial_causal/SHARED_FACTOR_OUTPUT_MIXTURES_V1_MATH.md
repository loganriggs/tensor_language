# Output-basis search and the scope of a shared reader

11 September 2026. These weight-only calculations address two obstacles to
discovering arithmetic graphs: a useful product may be hidden by output mixing,
and a fitted graph may omit computations that read the same input feature.

**The output-mixture search found gains, but no nearly isolated product at the
12 fixed readers. Lifting one reader to the complete native layer reveals two
strong leading branches and a broad remainder.** Neither result establishes
the absence of reusable computation or the identity of a behavioral circuit.

## Exact output-mixture search

Write the fitted layer as $\sum_g c_g(x^\top Q_gx)$. For unit $u$, projection of
a symmetric quadratic matrix onto products using $u$ is

$$
S_u(Q)=u(Qu)^\top+(Qu)u^\top-(u^\top Qu)uu^\top.
$$

This is the existing fixed-reader orthogonal projection. For a mixture
$Q(\alpha)=\sum_g\alpha_gQ_g$, define

$$
G_{gh}=\langle Q_g,Q_h\rangle_F,\qquad
H_{gh}=2(Q_gu)^\top(Q_hu)-(u^\top Q_gu)(u^\top Q_hu).
$$

The largest generalized eigenvalue of $H\alpha=\lambda G\alpha$ gives the
greatest fraction of mixture energy representable by one product using $u$.
Thus this part is solved exactly within the specified span, without nonlinear
optimization. Normalize individual $Q_g$ before reporting mixture coefficients;
that prevents arbitrary group scales from deciding whether a mixture uses two groups.

The [native-sized calculation](SHARED_FACTOR_OUTPUT_MIXTURES_V1.json) compares
each reader's existing consumers with all 64 groups. Its dense projection-Gram,
generalized eigen-residual, and implicit coefficient checks agree within 2.34e-15.
None of 12 full-span optima reaches the registered 95% bar (B/C miss).
Maxima range from 25.14% to 81.19%. Reader 8 increases from 11.74% in its local
consumers to 56.46% in the full span, so grouping can substantially hide a relation.
The strongest full-span result, reader 6, only improves its best individual group
by 0.53 percentage points. The large gain and near-product properties do not coincide.

This does not exclude changing the reader, changing the quadratic span, or using
several product nodes. Nor does rotating the output basis alone establish cheaper
execution: the remaining groups and input/output adapters must still be priced.
No graph-topology optimization was run in this calculation.

## Lift the frozen reader to the entire native layer

The surrogate captures only about 12% of the native coefficient tensor, so the
next calculation removes that restricted-span assumption. Hold parent 1's reader
fixed. Using the actual native $L,R,D$ and full-unembedding metric whitener $W$,
the entire projected tensor has the form

$$
D_u^{\mathrm{native}}(x)=(u^\top x)M_ux,
$$

$$
M_u=WD\left[\operatorname{diag}(Ru)L+
\operatorname{diag}(Lu)R\right]
-\left[WD((Lu)\odot(Ru))\right]u^\top.
$$

This uses the existing `native_partner` function; no reader or text fitting.
The metric-corrected SVD described in the canonical-branches note gives the best
rank curve at that fixed reader. Direct input-deletion execution agrees within
9.1e-14. [Receipt](SHARED_PARENT1_NATIVE_LIFT_V1.json).

| Native projected branch count | Fraction of this projected tensor's energy |
|---|---:|
| 1 | 21.16% |
| 2 | 37.02% |
| 4 | 41.90% |
| 16 | 52.47% |
| 64 | 65.56% |
| 256 | 86.27% |

Reaching 95% requires 463 branches; 99% requires 722. These ranks concern the
orthogonal projection at one frozen reader, not a lower bound on arbitrary DAGs.
That entire projection contains 0.3534% of the full layer's coefficient energy.

The two old output directions capture 33.22% of the native projection (50% bar
missed), versus 37.02% for the optimal two directions. They contain about 89.15%
of each of the first two native output singular vectors. This supports a useful
leading output subspace, with substantial remaining computation. However, the
complete frozen node function has cosine only 0.332 with the native projection,
missing the registered 0.8 bar: output agreement is not agreement of input partners.

## Graph-node deletion is a different intervention

The last miss has a concrete scope confound. Setting one named shared node to
zero affects its declared consumers. Projecting the input perpendicular to $u$
affects every computation that reads any component of $u$.

An [executed scope comparison](SHARED_PARENT1_SCOPE_ALIGNMENT_V1.json) projects
the entire fitted graph using the same `native_partner` formula and checks against
its actual input-deletion executor (6.8e-14 error). Fifteen of its 64 groups have
at least 1% of their own quadratic energy touching this direction, although only
groups 16 and 25 explicitly consume the named node. This is algebraic overlap,
not fifteen identified semantic consumers.

The full-graph/native projection cosine rises to 0.679, still below 0.8. The
declared-node/full-graph projection cosine is 0.481, and the latter has 1.73 times
the former's energy (the registered twofold bar also misses). Scope explains part
of the disagreement; approximation and organization remain unresolved.

The queued behavioral screen retains its frozen **named-node branch removal**.
These analyses neither change that intervention nor replace its bars. A broader
native input-direction intervention would require its own declared meaning,
including how normalization and background computation are handled.
