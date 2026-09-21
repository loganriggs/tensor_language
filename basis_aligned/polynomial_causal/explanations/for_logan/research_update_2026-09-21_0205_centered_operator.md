# Matching the context-dependent operator directly

21 September 2026, 02:05 UTC.

The [recombination test](research_update_2026-09-21_0200_input_recombination.md) found that the smaller graph's context-dependent interaction was less accurate than its aggregate effect. A fixed-dictionary experiment now shows that the fitting objective accounts for part of that gap. Refitting against the original centered operator improves interaction error by approximately 10–11%, while slightly worsening paired reconstruction. The correction has a substantial linear-map cost, so this is not yet an adopted improvement.

## Exact weight-based objective

Keep the existing 512 learned product directions $a_j,b_j$. Let $M_n$ and $M_m$ be centered calibration covariance matrices. For independent centered inputs with those covariances, the product Gram matrix is exactly

$$
K_{ij}=(a_i^\top M_n a_j)(b_i^\top M_m b_j).
$$

The original folded operator is

$$
F(n,m)=\sum_k c_k[(\ell_k^\top n)(r_k^\top m)+(r_k^\top n)(\ell_k^\top m)].
$$

The product–teacher cross matrix is therefore

$$
H_{jv}=\sum_k c_{vk}\left[
(a_j^\top M_n\ell_k)(b_j^\top M_m r_k)
+(a_j^\top M_n r_k)(b_j^\top M_m\ell_k)
\right].
$$

This uses all original 4,608 channels and their symmetric counterparts. It does not fit the previous approximate graph as a teacher. Conditional output fitting solves the corresponding normal equations; we tested unregularized fitting and six strengths of ridge regularization toward the previous output map, after normalizing product columns.

The metric uses independent centered marginals. It is not the joint distribution of native pairs. Covariances supply data information, while the target comes directly from trained weights. Dense Cartesian-product toys verify both contractions to below $4\times10^{-16}$ relative error.

## Preserve the existing single-input behavior

We add only a centered bilinear correction:

$$
\widehat F_{\mathrm{new}}(n,m)=\widehat F_{\mathrm{old}}(n,m)
+\sum_j d_j[a_j^\top(n-\bar n)][b_j^\top(m-\bar m)].
$$

The correction vanishes when either input equals its calibration mean, preserving the old graph's constant and single-input terms in the expansion around those means. This does not claim that the old terms are exact native terms.

| Diagnostic | Existing 512-product graph | Unregularized centered refit |
|---|---:|---:|
| Paired calibration reconstruction error | 20.00% | 20.36% |
| Source-only swap-effect error, three shifts | 19.50–20.01% | 18.05–18.50% |
| Context-dependent interaction error | 34.43–35.77% | 30.86–32.03% |

The proposed screen—at least 10% relative interaction-error reduction on all three shifts, with paired error no more than 5% worse—passes. Large ridge penalties reduce the improvement. These are reused calibration diagnostics; no independent generalization or native final-logit result follows.

## Graph cost: products can be shared, linear maps still count

Let $s_n=P_n^\top n$, $s_m=P_m^\top m$, and let $a=T_n^\top s_n$, $b=T_m^\top s_m$ be the 512 product inputs. Define their means $\alpha,\beta$. In row-vector notation,

$$
[(a-\alpha)\odot(b-\beta)]D
=(a\odot b)D-(a\odot\beta)D-(\alpha\odot b)D+(\alpha\odot\beta)D.
$$

Thus we can fold $D$ into the existing product output map, add two linear maps from the shared 256-dimensional input features, and adjust the constant. The same 512 products suffice; we need not evaluate a second centered product dictionary.

An executable toy verifies this compilation to relative error $1.95\times10^{-16}$. A direct dense implementation at native dimensions stores:

| Part | Weight coefficients |
|---|---:|
| Shared input projections and product inputs | 851,968 |
| Combined product output map | 589,824 |
| Two linear compensation maps | 589,824 |
| Total, excluding bias | 2,031,616 |

This is larger than the existing 512-product program's 1,291,264 coefficients. Compressing or sharing the correction's linear maps remains necessary before claiming a favorable cost–fidelity tradeoff. No whole-model speedup or native adoption is claimed.

The result narrows the uncertainty: the current product dictionary can represent a better approximation of the context-dependent operator than its existing output coefficients do. It does not resolve the remaining factor-capacity limits, stable feature identities, selective semantic manipulation or composition across model sections.

Evidence: `MIDPOINT_CENTERED_OPERATOR_REFIT_V1.json`, `MIDPOINT_CENTERED_CORRECTION_COMPILE_V1.json`, and their executable Python scripts in `direct_tensor_match`.
