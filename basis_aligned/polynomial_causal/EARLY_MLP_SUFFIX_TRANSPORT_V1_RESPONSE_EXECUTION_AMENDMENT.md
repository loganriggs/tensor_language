# Prospective amendment: paired-response physical execution

Date frozen: 2026-08-28

Status: source/interface amendment only. It does not authorize loading the final role,
does not claim a scientific result, and does not relax any existing terminal gate.

## Purpose

The original transport preregistration fixes the intervention distribution and
scientific statistics, but it did not fully type the physical execution boundary.
This amendment fixes the remaining choices before any final response is observed.

## Frozen response support and objects

- Every response is evaluated on model positions 64 through 255 inclusive.
- MLP1 coordinate means the **raw native MLP1 write** projected through the frozen
  output basis, `native_mlp1_write.float() @ B1`; it is not the residual stream.
- Logit response means the facade's post-softcap logits. Vocabulary centering is
  performed separately for each token response vector.
- Output KL is summed over the same positions 64 through 255.
- Positive-minus-own-baseline and negative-minus-own-baseline are two occurrences.
  They are pooled; they are not replaced by a central difference.

## Authority-derived edit

For global final row `g`, the position and direction index are recomputed from the
frozen final intervention assignment. With selected validation-only multiplier
`alpha`, fitted natural code RMS `r`, normalized float64 direction `v_k`, and sign
`s` in `{-1,0,+1}`, the semantic edit is

```
semantic_delta[g] = s * alpha * r * v_k                    # float64 [64]
```

The canonical executable edit is the float32 cast placed at exactly that row's one
assigned position in a zero tensor of shape `[4,256,64]`. The matching physical MLP0
write is computed once as

```
physical_edit = code_edit @ B0.T                            # float32 [4,256,1152]
```

The semantic float64 delta, executable code edit, and physical edit have distinct
content hashes. The code and physical edits must replay through the frozen B0 basis;
post-mint mutation, a changed row/target token, position, direction, amplitude, sign,
program payload, geometry, calibration, or basis is an integrity failure. Equivalent
GPU operation order may use the inherited replay tolerance `2e-6`; the canonical CPU
construction itself is byte-identical.

The zero-sign baseline has exactly zero code and physical edits, but shares the same
intervention-unit identity with its positive and negative partners.

## Exact teacher path

For each four-row batch, one private exact teacher baseline/positive/negative triplet
is executed and shared across all 22 response actions:

1. run exact native MLP0 and add the matching physical edit exactly once;
2. continue the edited residual trajectory through live attention 1;
3. run exact native MLP1 and capture its raw write projected through B1;
4. use deployed MLP2 N and the frozen ship thereafter;
5. retain only positions 64--255 internally until paired reductions are complete.

The required early call ledger per teacher forward is deployed-N
`((0,0),(1,0),(2,1))`, corrections all zero, and literal exact calls
`((0,1),(1,1),(2,0))`. Every attention and MLP dispatch occurs once. Raw teacher
codes, writes, states, and logits cannot cross the complete batch boundary.

## Student response identity

The old observational final binder hardcodes `trial=0`; reusing it three times would
mint duplicate broker identities. For response execution only, perturbations are
prospectively mapped as:

| perturbation | trace trial |
|---|---:|
| baseline | 0 |
| positive | 1 |
| negative | 2 |

A response-specific outer identity binds that trace to the response-forward plan,
semantic action, physical materialization, complete final action/row identity,
intervention-unit identity, and actual semantic/code/physical edit hashes. The trace
trial alone is not evidence that an edit occurred.

## Receipt and publication requirement

Per-arm reductions must bind the actual observed teacher and student forward-receipt
hashes, not merely planned-forward hashes. LL and LT emit typed code, centered-logit,
and output-KL reductions; the twenty indexed nulls emit centered-logit and output-KL
reductions only.

No batch receipt is returned unless the exact ordered 69 forwards close: three shared
teacher forwards and three student forwards for each of 22 actions. No run receipt is
returned unless all 48 canonical batches close, giving exactly 144 teacher and 3,168
student forwards. Any failure discards partial tensors and receipts. The terminal
final closure must require this run receipt and structured ledger before response
statistics can influence a scientific result.

## Implementation state after the 2026-08-28 19:48 UTC audit

Implemented and tested: authority-derived semantic/code/physical edits, mutation and
replay guards, distinct perturbation-bound student identities, the private exact
teacher forward, private one-use student response consumption, response support,
vector and output-KL reductions, and one fail-closed ordered 69-forward batch router.
The batch router executes three shared teacher forwards plus 66 student forwards,
reduces all 22 arms immediately, binds each reduction to its actual observed teacher
and student receipt hashes, and refuses to return unless the schedule, coordinator,
and 66/66/66 broker ledger close. The affected response/capability/adapter suite passes
78 tests, including a synthetic full-batch transaction and exact fake-model teacher
and student forward tests.

Still NO-GO: the ordered 48-batch accumulator, its mandatory 144-teacher/3,168-student
run ledger, terminal closure wiring, and final-role execution. The synthetic batch
test proves routing and receipt integrity, not scientific behavior on the final role.
