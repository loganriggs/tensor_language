# Gauge transport triangle v1 execution preregistration

**Frozen before model execution:** 2026-08-29 08:28 UTC

## Purpose and scope

This run is the preliminary finite-intervention screen for E3.2. It asks whether a
rank-64 state at block 8 predicts a sealed response at block 14, both directly and by
composing independently fitted maps through block 11:

$$
\widehat{\Delta c}_{14}^{\mathrm{direct}}
  = \Delta c_8 T_{8\to14},
\qquad
\widehat{\Delta c}_{14}^{\mathrm{chain}}
  = \Delta c_8 T_{8\to11}T_{11\to14}.
$$

The chain never reads the true block-11 response on evaluation rows. The evaluation
intervention is a finite, position-matched donor-minus-target residual edit, not an
infinitesimal Jacobian probe. A negative result is an E3.2 outcome for this pointwise
rank-64 grammar. A positive preliminary screen is not an interface license; it only
authorizes the already specified 20-null, behavior, alternate-background, and full
price extension in `PRICED_GAUGE_TRANSPORT_SPEC.md`.

## Immutable inputs

- Checkpoint: bilin18 weights SHA256
  `680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3`.
- Rows: the receipt-last v2 unique-document artifact with 96 basis, 96 response-fit,
  and 192 evaluation rows from 384 distinct documents.
- Row tensor file SHA256:
  `102b79726b7132a6438b4080272fee1774499ac4fc83c4aa025fa86439b4074d`.
- Row receipt file SHA256:
  `3f92d8b3aa5e89e6059a010338521bffa0cf440e0815d9d67e1b65aa58a8e102`.
- Sites 8, 11, and 14; raw post-block residual semantics; live downstream RMSNorm.
- Rank 64 bases inside independently fitted rank-256 local token-deviation supports.
- Four antithetic isotropic fit draws, relative ridge $10^{-3}$, and no intercept.
- Sixteen response rows calibrate one amplitude into median suffix KL `[0.01, 0.20]`;
  64 fit the maps and 16 provide a non-selecting row-group validation diagnostic.

No basis, rank, ridge, amplitude, map, or threshold may use evaluation outcomes.

## Frozen measurements and decisions

The output error for an arm is

$$
E_{\mathrm{out}}
=\frac{\sum_i \mathrm{KL}(p_i^{\mathrm{early}}\Vert
                           p_i^{\mathrm{transported}})}
        {\sum_i \mathrm{KL}(p_i^{\mathrm{early}}\Vert
                           p_i^{\mathrm{baseline}})}.
$$

The screen records centered raw-logit relative RMSE and coordinate response $R^2$.
It passes only if all existing `screen_decisions` are true:

1. full block-14 oracle: $E_{\mathrm{out}}\le10^{-3}$ and raw-logit error
   $\le10^{-3}$;
2. projected rank-64 block-14 oracle: $E_{\mathrm{out}}\le0.25$;
3. direct map: coordinate $R^2\ge0.75$ and $E_{\mathrm{out}}\le0.25$;
4. chain: $E_{\mathrm{out}}\le0.35$, no more than `0.10` worse than direct, and
   coordinate $R^2$ at least 75% of direct.

Native/reference replay, zero patch, CP gauge rewrite, position shuffle, expanded
physical-map gauge rewrite, and price drift are fail-closed canaries. Scale-calibration
failure is a recorded terminal negative before response fitting, not permission to
change the grid.

## Source closure and terminal lifecycle

An execution authority created only after the runner, reducer, specification, and
tests are committed and pushed binds their SHA256 hashes, the checkpoint, the row
artifacts, and a fresh terminal namespace. The runner refuses dirty or unpushed bound
sources. Result and optional state are create-only; their semantic replay is checked;
the receipt is written last. Any exception before the receipt creates a mutually
exclusive failure artifact containing the traceback and hashes of partial terminals.

Terminal files are:

- `gauge_transport_triangle_results.json`
- `gauge_transport_triangle_state.pt`
- `gauge_transport_triangle_v1_execution_receipt.json`
- `gauge_transport_triangle_v1_execution_failure.json`
