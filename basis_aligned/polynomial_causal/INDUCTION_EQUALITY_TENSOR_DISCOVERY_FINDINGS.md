# Induction equality-tensor discovery findings

**Executed:** 2026-08-30 04:36 UTC

**Runtime:** 38.05 seconds

**Receipt:** `induction_equality_tensor_discovery.json`

**SHA-256:** `0b826952d227c6f2c9e8b0fadf19aeb28edcd4153a52e4b67777a587733e184b`

## Result

The fixed equality-and-successor tensor across `L5H5`, `L7H3`, `L8H3`, and `L8H4`
passed every preregistered SELECT gate. It is eligible for a one-shot fresh natural
FINAL and code-OOD transaction.

The support tensor is

$$
M_{qk}=\langle e_{t_q},e_{t_{k-1}}\rangle\mathbf 1[1\le k\le q].
$$

It retains every attention edge whose key immediately follows an earlier occurrence
of the query token. All matches are summed; there is no argmax, nearest-match, TopK,
or `has_match` router.

## Selective removal

Deleting only these equality-fetch edges caused target CE damage `+0.51225` nat, with
95% document interval `[+0.30746,+0.76051]`. Target-minus-matched-negative specificity
was `+0.55251`, interval `[+0.34037,+0.80667]`. The matched-negative point effect was
beneficial rather than harmful, about `-0.0400` nat.

Off-target damage was only `+0.006264` nat, interval `[+0.002986,+0.009824]`, passing
both the 0.01-nat absolute limit and the 10%-of-target relative limit.

## Extraction and null

Deleting the four full heads raised positive-cell CE from `0.67478` to `1.11713`.
Restoring only their equality-fetch tensors brought it to `0.68597`, for extraction
recovery `0.97397` with 95% interval `[0.94789,0.99479]`.

The same-shape fixed cyclic-vocabulary permutation recovered `-0.00216`, interval
`[-0.01084,+0.00617]`. Thus the result depends on the correct token identity relation,
not merely adding a similarly sized edge set.

## Execution and composition

All four replacements were applied sequentially on the live residual stream: later
heads consumed the residual consequences of earlier replacements. Analytical replay
matched native logits bit-for-bit. Every candidate arm made zero calls to the native
attention and Q/K/Q2/K2/V/O modules at layers 5, 7, and 8; native made exactly 48 calls
to each registered site/submodule.

The earlier four-head mean-replacement screen had target damage `+0.4487` but failed
collateral at `+0.0244`. This result identifies the reason: whole-head interventions
erase other services. Restricting the intervention to the equality-fetch tensor keeps
the induction service and removes most unrelated damage.

## Claim boundary and next action

This used the already-opened 192-document SELECT role. Natural FINAL, code OOD, and
synthetic outcomes remain unopened. It establishes a strong discovery-level
extractable and selectively removable tensor circuit, not yet terminal/OOD evidence or
a cheaper parameterization of the four native QK/V factors.

Freeze the exact four heads, equality tensor, fixed derangement, metrics, and gates;
then execute FINAL natural and OOD code once. In parallel, factor the selected heads'
QK and OV tensors jointly, requiring the cheaper version to reproduce this circuit's
finite effects rather than only local scores.
