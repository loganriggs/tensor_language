# Finite gauge-transport triangle result — 2026-08-29

## Outcome

The preregistered pointwise rank-64 L8 $\rightarrow$ L11 $\rightarrow$ L14
finite-intervention interface fails its preliminary screen. This is a receipt-backed
negative E3.2 outcome on 384 globally unique FineWeb documents.

The evaluation uses a finite position-matched donor-minus-target edit at block 8. The
direct predictor is

$$
\widehat{\Delta c}_{14}=\Delta c_8T_{8\to14},
$$

and the composed predictor is

$$
\widehat{\Delta c}_{14}
=\Delta c_8T_{8\to11}T_{11\to14}.
$$

The composed arm never observes the true evaluation-time block-11 response.

## Measurements

| Arm | Output KL error $E_{\mathrm{out}}$ | Coordinate response $R^2$ | Frozen gate |
|---|---:|---:|---|
| Full block-14 response oracle | $1.50\times10^{-11}$ | 1.000 | pass |
| True response projected into rank-64 $U_{14}$ | 0.2709 | 1.000 | **fail**: must be $\le0.25$ |
| Direct $T_{8\to14}$ | 0.4861 | 0.4028 | **fail** |
| Chain $T_{8\to11}T_{11\to14}$ | 0.4520 | 0.4024 | **fail** |

The full oracle's centered raw-logit relative error is
$4.37\times10^{-6}$, confirming that the intervention and downstream-resume harness
is accurate. The position-shuffle control has response NRE `1.2645`, well above its
`0.25` detection floor. CP gauge, physical-map gauge, exact replay, price-drift, and
zero-patch controls all pass.

The scale calibration selected multiplier 10, with median suffix KL `0.03983` inside
the frozen `[0.01, 0.20]` band. Thus the negative is not caused by an imperceptibly
small or saturated edit.

## Interpretation

The earliest failure is **destination-basis insufficiency**. Even when given the true
physical block-14 response, its rank-64 projection has $E_{\mathrm{out}}=0.2709$.
No better fit of a map into the same basis can repair information that $U_{14}$ discards.
The direct and chain maps fail more strongly, retaining only about 40% coordinate
response variance on the sealed donor family.

The chain is slightly better in output KL than the direct map (`0.4520` versus
`0.4861`) and has almost identical coordinate $R^2$. Therefore the experiment does
not show a special extra catastrophe from composing the two learned maps; it shows
that this rank-64 pointwise state grammar is already inadequate before composition
can become a useful API.

This prunes:

- the frozen rank-64 local token-deviation basis as a reusable L14 state interface;
- direct or chained pointwise linear transport in that basis;
- E3.3 selective state editing using this failed interface;
- the expensive 20-null, behavior, and alternate-background extensions, which were
  conditional on passing this preliminary screen.

It does **not** prune higher rank, temporal/position-mixing kernels, behavior-specific
states, explicit nonlinear interaction terms, or balanced causal ports fitted to a
particular component boundary. Those are different grammars and must be separately
priced and preregistered.

## Lifecycle record

- V1 failed before rows/model load because the repository root was absent from the
  import path. The failure is preserved.
- V2 made only that import-path correction and completed the full 42.9-second
  scientific run. Receipt publication then failed because JSON converts integer
  dictionary keys to strings; complete result and state hashes were preserved.
- V3 performed no scientific rerun. Its CPU-only receipt recovery bound the exact v2
  result, state, failure, and authority hashes; checked the result decisions and state
  shapes/finiteness; and wrote receipt last.

Authoritative artifacts:

- `gauge_transport_triangle_results.json`
- `gauge_transport_triangle_state.pt`
- `gauge_transport_triangle_v3_receipt.json`
- `gauge_transport_triangle_v1_execution_failure.json`
- `gauge_transport_triangle_v2_recovery_failure.json`
