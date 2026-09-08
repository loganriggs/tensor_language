# Temporal/is-was L11H3 writer route-closure ladder result — 2026-09-08

## Verdict

The nested controlled-mediator ladder passes its validity, source-replay, and selectivity gates
and terminates `l11h3_bypass_dominant`. Neither value spreading nor writer-induced L11H3 routing
explains the missing selected is-was writer effect.

Result artifact:
`basis_aligned/bilinear_quotient/circuits/followups/temporal_iswas_l11h3_writer_route_closure_ladder_v1_result.json`

SHA-256: `a5401d739775526fa1802c09d7cf633d6ae1253625a32671aabbb822a0754e6c`.

## Nested mediator scores

| role | population/phase | source value | full-prefix value | full L11H3 head |
|---|---|---:|---:|---:|
| is-was | original HOLDOUT | `.1650` | `.1650` | `.1548` |
| is-was | OOD FIT | `.2031` | `.2031` | `.1964` |
| is-was | OOD HOLDOUT | `.2025` | `.2025` | `.1966` |
| temporal | original HOLDOUT | `.7993` | `.7991` | `.8006` |
| temporal | OOD FIT | `.7841` | `.7834` | `.7856` |
| temporal | OOD HOLDOUT | `.7829` | `.7822` | `.7847` |

Entries are signed recovery against the selected-writer effect. Every arm/cell has direction
agreement `1.0`. Is-was cosine remains `.9762-.9885`, so L11H3 carries a small aligned component,
not random noise; relative residual remains `.798-.846`. Temporal remains a strong, stable
positive control with cosine >=`.9978` and residual <=`.223`.

The source and full-prefix value tensors are exactly equal for is-was in both populations; their
relative L2 difference is zero. Thus the compact `L7H7+L9H4` intervention does not create another
L11H3 value contribution elsewhere in the causal prefix. The full observed head differs from the
value-only tensor (relative L2 `.378` OOD, `.533` original), proving induced routing/head changes
exist, but installing them slightly reduces rather than rescues behavioral mediation.

Source-report replay, later-to-earlier causal zero, and all collateral are exact or within the
frozen bars. The result is a stable causal split: the two writers have an L11H3 source branch for
is-was, but roughly 80% of their is-was command effect travels through another downstream route.

## Next circuit action

Run full native module-output mediation across downstream blocks before splitting passing
attention modules into heads. Capture the exact writer-induced outputs of each attention module
and MLP, install one at a time into a native run, and score recovery against the selected writer
effect. Use temporal L11 attention as the positive control, original FIT/HOLDOUT for the cheap
screen, and promote only stable modules to OOD/head-level confirmation. After localization,
translate the induced mediator tensors through exact downstream weights to name which heads/MLPs
read the bypass variable. This directly implements full module/head narrowing and weight-tensor
reader tracing rather than fitting another DAS subspace.
