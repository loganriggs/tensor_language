# Three-hour mathematical review — 2026-09-06 14:30 UTC

## Decision

Keep the circuit program on the exact block8H1 -> {block9H1/H4, block11H3} path. The newest insertion experiment upgrades the two readers from a removal-mediation description to a writer-free executable response program. Do not return to cross-layer concatenated DAS: a subspace/complement decomposition is only defined at one live component boundary (or independently per boundary).

## Exact response-program accounting

Let `m(x)` be the answer-minus-foil margin of the native suffix. Let `w` denote the exact block8H1 cue intervention, and let `r9` and `r11` be the captured changed-minus-base output responses at block9H1/H4 and block11H3. Define

\[
W=m(w)-m(0),\qquad R_S=m(\operatorname{install}(S))-m(0)
\]

for a causally ordered set of installed response tensors `S`. The preregistered writer-free experiment measures normalized means `R_S/W`. Its results are

| family | `R9/W` | `R11/W` | `R9,11/W` | other heads / `W` |
|---|---:|---:|---:|---:|
| A1 | 0.443034 | 0.515301 | 0.958468 | -0.000711 |
| A2 | 0.484605 | 0.523091 | 1.008440 | -0.027343 |

The two-factor interaction in normalized margin space is

\[
I_{9,11}=R_{\{9,11\}}-R_{\{9\}}-R_{\{11\}}.
\]

It is only `+0.000133` of the writer effect in A1 and `+0.000744` in A2. Thus the two installed responses are not merely jointly sufficient; their downstream behavioral contributions are additive to within 0.08% of the writer effect on both families. The remaining writer residual is `+0.041532` in A1 and `-0.008440` in A2. This does not prove the response tensors themselves are linearly independent, but it does give an exact, fit-free behavioral accounting at the registered endpoints.

This insertion result and the earlier removal result answer different counterfactuals. Simultaneously clamping both subject readers leaves 3.46%/0.85% of the writer effect, while installing their responses recovers 95.85%/100.84%. Their agreement rules out the common failure mode where a path is necessary only because clamping damages unrelated computation or sufficient only because addition steers a high-gain suffix.

## What the DAS objective must mean

At a single live component with exact donor-base displacement `d`, let `Q` be an orthogonal projector. The coherent interventions are

\[
z_Q=z+Qd,\qquad z_{\perp}=z+(I-Q)d.
\]

They satisfy tensor closure `z_Q+z_perp=2z+d`, and `Q=I` must reproduce the exact component patch. A constrained objective may then minimize

\[
L(Q)=E[(\Delta m_Q-\Delta m_d)^2]+\lambda E[(\Delta m_\perp)^2],
\]

preferably after normalizing by the exact component effect. The first term tests sufficiency; the second tests complement inertness. Neither term alone identifies a unique direction when the suffix exposes only a low-dimensional readout. Under a nonlinear suffix, `Delta m_Q + Delta m_perp = Delta m_d` is a diagnostic approximation, not an algebraic theorem; exact equality is guaranteed for component tensors, not for logits.

The red-team result resolves the apparent contradiction with difference-in-means. Cross-layer cached additions violated the live-boundary definition above, so their optimization target was not a partition of the exact intervention. In the closure-valid block11H3 experiment, constrained DAS reduced its own fit objective 24.6-fold relative to difference-in-means and improved A2 transfer (`0.840` exact-effect fraction versus `0.753`) while lowering the complement (`0.157` versus `0.246`). Optimization therefore found a better solution to the intended objective. Difference-in-means remains a strong zero-fit baseline because it has lower variance and selects the empirical displacement centroid, but it does not dominate the corrected constrained objective.

## Complexity and identifiability

For `n` response branches, exhaustive interaction accounting costs `2^n` installed subsets. Here `n=2`, so the complete Mobius decomposition is only four endpoints and has no estimation ambiguity. General greedy selection is appropriate when whole-head/module screening finds many branches: at each step add the component with the largest held-out gain under the exact live intervention, then validate the final set jointly. DAS should follow localization and operate separately inside each fixed component; a single joint rank-one axis spanning causally ordered layers is not available in one forward because later live displacements depend on earlier interventions.

The H3 optimized direction is reproducible (`|cos|` approximately 1 across independent deterministic fits), but that is an empirical property of this restricted single-component problem. The multi-layer direction battery showed lower restart agreement, so equal behavioral loss is not enough for identifiability. Required trust checks remain: full-rank closure, held-out and reverse-orientation transfer, complement inertness, random-direction control, restart cosine, and donor-side necessity.

## Next executable consequence

The verified response program is still row-matched: it replays captured `r9` and `r11`. The next mathematical bottleneck is compression, not another localization sweep. Fit or derive separate predictors for the two reader response tensors from the already identified writer/cue state, and compare:

1. each predicted response against its captured tensor under a held-out Frobenius and causal-margin norm;
2. their joint installed output against the fit-free 0.958/1.008 response-program ceiling;
3. a shared low-rank predictor against two independent predictors; and
4. the literal parameter/multiply price against cached response replay.

The causal-margin criterion must be primary because a large tensor residual can be behaviorally inert, as the direction studies already show. Freeze the predictor family and ranks before model execution. A failure to predict the tensors while retaining causal margin licenses a quotient-space program; a failure in causal margin means the captured response program has not yet become predictive.

Next mathematical review due around **2026-09-06 17:30 UTC**.
