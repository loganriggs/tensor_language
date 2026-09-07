# Selected projector × attention-15 dependency factorial

## Decision

The learned four-head projector has only been evaluated while every computed layer-15 attention
head output is replaced by a cached donor output.  The next causal question is therefore whether
the projector still transfers the temporal answer margin when attention 15 is allowed to react to
the changed upstream state.  This is required before interpreting the weight-ranked `L15H5`
Q/K/V interfaces as a causal reader.

The experiment is eligible only when the queued multi-construction parent is mechanically valid
and finds a moved checkpoint that is target-feasible on both v15 constructions.  If that gate
fails, the already registered separate-axis/projective-bisector test has higher information value:
there is no construction-stable selected projector whose downstream dependency should be named.

## Exact intervention game

For each parent evaluation parity, use the selected rank-one bases trained on the opposite parity,
without fitting or choosing any parameter.  Cross two bits:

| Cell | Four upstream projectors | Attention 15 |
|---|---|---|
| `00` | off | live/native |
| `01` | off | complete donor clamp |
| `10` | on | live/native |
| `11` | on | complete donor clamp |

The upstream sites remain `L8H1`, `L9H1`, `L9H4`, and `L11H3`, installed in ascending-layer order
with the parent's absolute unit-dose projector rule.  The panels are v15 A1/A2/P/C and v16
A1/A2/P.  V16 C remains excluded because it lacks the full-sequence alignment contract.

For the rowwise answer-minus-foil margin response $r_{ab}$ relative to one native base, compute

\[
u_{\mathrm{live}}=r_{10}-r_{00},\qquad
a_{\mathrm{alone}}=r_{01}-r_{00},\qquad
i=r_{11}-r_{10}-r_{01}+r_{00}.
\]

The exact closure is

\[
r_{11}=r_{00}+u_{\mathrm{live}}+a_{\mathrm{alone}}+i.
\]

The same cells also give the conditional upstream effect $r_{11}-r_{01}$ and conditional
attention-15 effect $r_{11}-r_{10}$.  Full-vocabulary KL and top-one flips are nonlinear and stay
as four separate cell reports; no additive KL attribution is allowed.

## Opposing outcomes

If arm `10` reaches the registered target and selectivity bars, the upstream projector has a real
route through the native suffix.  If its effect is also at least 75% of arm `11` and the interaction
is small, complete attention 15 was mostly a background clamp rather than a required gate.  That
licenses a separate reset/rescue experiment at the weight-predicted L15H5 interfaces; the factorial
itself does not distinguish L15H5 from the other eight live layer-15 heads.

If arm `11` replays the parent but arm `10` fails, the current projector depends on the artificial
attention-15 background.  The upstream coordinate may remain a valid controlled intervention, but
the native-reader chain is rejected.  A large interaction identifies conditional dependence even
when arm `10` retains some target effect.

## Reusable implementation

`ops/two_by_two_dependency_contract.py` performs the exact rowwise linear accounting, validates
the zero arm and closure, and uses the same signed-projection convention as the temporal runners.
It intentionally refuses nonlinear control objects.  The eventual GPU runner must bind the parent
result hash after it lands and replay arm `11` against the stored parent report before opening any
reader intervention.
