# Separate linear spaces with adaptive ranks improve the same-cost graph

21 September 2026, 02:56 UTC.

The error-composition test did not find consistent reinforcement between the two approximation errors. The larger, more consistent issue was the error introduced by linear compression itself. We then changed its structure: give the two input roles separate output spaces and allocate their ranks under the same coefficient budget. This improves native full-swap fidelity without changing the512-product interaction.

## Error composition: no single cancellation story

For native effect $y$, exact-linear program effect $\hat y_0$, and compressed-linear effect $\hat y_1$, define

$$
e_0=\hat y_0-y,\qquad d=\hat y_1-\hat y_0.
$$

Then

$$
\|\hat y_1-y\|^2=\|e_0\|^2+\|d\|^2+2\langle e_0,d\rangle.
$$

For the separate-role256 program's whole-contribution swaps, fractions of total squared error are:

| Domain | Baseline error | Incremental error | Cross term |
|---|---:|---:|---:|
| FineWeb | 60.03% | 46.43% | −6.46% |
| Code | 62.27% | 29.00% | +8.73% |

The signed fractions sum to100%. Errors partly cancel on FineWeb and reinforce modestly on code. The registered prediction of at least10% positive cross contribution in both domains fails; the incremental-error prediction and additive-identity checks pass. Calibration showed a small negative cross term as well.

These are errors of final native logit effects. After nonlinearities, the increment is a change between two predictions, not a standalone linear-write effect. This audit does not prove that joint refitting cannot help, but it does not justify treating reinforcement as the common primary cause.

## Shared versus private linear features

A common output basis uses

$$
\hat J(n,m)=W(P_n^\top n+P_m^\top m),
$$

costing $3dr$ coefficients for input/output width $d$. Separate spaces use

$$
\hat J(n,m)=W_nP_n^\top n+W_mP_m^\top m,
$$

costing $2d(r_n+r_m)$. With $d=1152$, common rank256 and private ranks summing to384 both cost884,736 linear coefficients. The bilinear graph is fixed, so all candidates below have2,654,208 total weights and512products.

Allocating the private ranks by retained calibration-role singular energy selects **344midpoint features and40source features**. This is optimal within the fixed independent linear-map SVDs and that coefficient budget, not for native-model behavior or arbitrary arithmetic circuits.

## Native diagnostic results

| Linear structure | FineWeb whole-swap error | Code whole-swap error | FineWeb / code source error | FineWeb / code CE added |
|---|---:|---:|---:|---:|
| Shared256 | 30.06% | 25.45% | 26.13% / 21.35% | 0.00911 / 0.04116 |
| Private192 +192 | 31.84% | 26.67% | 25.96% / 21.06% | 0.01027 / 0.05237 |
| Private344 +40 | 28.45% | 24.15% | 26.43% / 21.65% | 0.00873 / 0.03850 |

The adaptive private candidate passes the registered comparison: whole-swap error improves in both domains and source error stays within5% of the shared256 program. Baseline replay and protected context effects are identical. Equal private ranks perform worse, so separating spaces alone is not the explanation; allocation matters.

This candidate still does not preserve the larger exact-linear program's whole-swap performance, and context-only error remains55.37% FineWeb/39.55% code. These are repeatedly used diagnostic panels, not fresh confirmation or semantic identification.

## Frozen successor comparison

We froze the new private512-product graph and the earlier corrected1,024-product graph before constructing another document panel:

| Frozen graph | Products | Weight coefficients | Other serialized entries |
|---|---:|---:|---:|
| New adaptive/private graph | 512 | 2,654,208 | 2,688 |
| Earlier grouped/corrected graph | 1,024 | 2,671,616 | 2,176 |

Other entries include means and stored zero vectors. This is approximately matched weight storage with half as many variable products; different linear arithmetic means it is not a runtime-speed claim. The frozen artifact and hash are saved. Fresh-document evaluation is pending, and code replacement loss must remain visible alongside any intervention gain.

Evidence: `MIDPOINT_ERROR_COMPOSITION_NATIVE_V1.json`, `MIDPOINT_PRIVATE_LINEAR_BUDGET_V1.json`, `MIDPOINT_PRIVATE_LINEAR_NATIVE_V1.json`, and `MIDPOINT_PRIVATE_FROZEN_V1.json` under `direct_tensor_match`. The complete circuit goal remains open.
