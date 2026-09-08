# P7 within-module split reuse analysis — 2026-09-08 12:07 UTC

## Decision and circuit target

The queued fixed-P7 leave-one-module-out audit decides which native modules are operationally
necessary inside the exact OOD-composable `identity + P7-response` interface. The next step must
split only passing native boundaries. This advances cross-boundary grouping/within-module
splitting, computational specification, OOD identification, and extraction; it is not a rank or
activation-reconstruction objective.

## Attention A11 branch

The input to A11's linear output projection `c_proj` is the concatenation of nine 128-dimensional
head outputs. For native and selected-writer executions, capture tensors
`z0,z1 in R^[batch,token,9,128]`. Replacing a subset `S` of head slices by donor slices gives

```text
z_S[h] = z1[h] if h in S else z0[h]
A11_S = c_proj(z_S).
```

Because `c_proj` is affine, the all-nine arm exactly equals the previously used complete A11
selected-writer output (up to deployed precision), including its unchanged bias. The valid
`temporal_iswas_h4_reader_factor_factorial_v1` runner already supplies the audited pre-`c_proj`
slice hook pattern. Reuse it inside the identity background while continuing to clamp the six
selected P7 MLP outputs. Run all nine singleton and all nine leave-one-head-out arms: the fixed
endpoint attribution

```text
0.5 * (singleton recovery + full recovery - leave-one-out recovery)
```

distinguishes a distributed A11 response from a necessary head without claiming exact Shapley
values. Full-nine must replay complete-A11 P7+identity before any head result is scored. Prior
L11H3 evidence predicts H3 will lead, but that earlier experiment tested a different background;
it cannot substitute for this within-P7 OOD result.

## MLP branch

For every P7 MLP that is necessary on both frozen OOD phases, reuse the exact pre-`Down` identity

```text
delta h = delta L * R0 + L0 * delta R + delta L * delta R.
```

The three factors are invariant under reciprocal Left/Right coordinate scaling and paired hidden
coordinate permutation. Capture them under native and selected-writer executions, install them
before the native `Down`, and require the all-three arm to replay that MLP's complete response in
the same P7+identity background. Earlier MLP8 evidence explicitly forbids dropping the interaction
term from magnitude or analogy. Static weight norms remain incidence priors, not selectors.

## Branch rule and cost discipline

- If queued prediction C passes, run the A11 9-singleton + 9-LOO split first because it directly
  tests a necessary attention boundary and has one exact full-replay control.
- If C fails, do not spend a head sweep on A11.
- For prediction-D MLPs, freeze the necessary MLP list from the immutable receipt, then execute
  exact three-factor arms. If many MLPs pass, screen each factor singleton plus full-three first;
  reserve all eight subsets for promoted factors.
- Do not fit DAS, select on OOD, lower the `.03` module-necessity bar, or replace physical causal
  effects with reconstruction, variance, or tensor-norm scores.

The queued module audit remains the sole authority for branch selection. This note changes no
bar or outcome and exists to remove repeated implementation/review latency once that receipt lands.
