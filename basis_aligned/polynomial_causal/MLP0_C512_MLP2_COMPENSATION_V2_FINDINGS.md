# C512 → MLP2 physical compensation factorial v2

## Verdict

The run is mechanically valid but its registered scientific labels are
**inconclusive** because one common bookkeeping gate was too strict. The raw
source-document ledgers, all integrity diagnostics, and the frozen 20,000-bootstrap
inference are complete and replay exactly.

The stable descriptive result is nevertheless informative:

1. With MLP2 omitted, the C512-induced mismatch is large.
2. With deployed MLP2 present, that mismatch is much smaller in both independent
   waves.
3. The local MLP2 state-by-write interaction point estimate is small
   (0.070 margins; pooled UCB 0.616), but its registered status is inconclusive.
4. The specifically aligned C512-induced MLP2 write does not beat the shuffled
   within-wave-and-cell write control.
5. C512 is still not certified as an equivalent or manipulable MLP0 interface.

Thus deployed MLP2 **and the ensuing suffix response attenuate most of the
observational C512 mismatch**, but this assay provides no support for the stronger
story that MLP2 computes a specially aligned repair write. Because of the failed
common gate, even the attenuation statement is descriptive, not a promoted
preregistered label.

## Immutable artifacts and integrity

- Result:
  `basis_aligned/bilinear_quotient/mlp0_c512_mlp2_compensation_v2_results.json`
- Result SHA-256:
  `06b70d277d4e5cf98d9a044db43a46a685b0c60d50d534fae62e9bccf2875539`
- Authority SHA-256:
  `167ce13f15c584d7e5af29e1c14e994d0d6af2f2741ec1f4d26527f589d943b5`
- Rows: 384 source documents, frozen waves 192/192, 628 chunks, 1,256
  evaluation windows.
- Coverage: 93.545% in wave A, 93.719% in wave B, 93.635% pooled.
- Inference: 20,000 source-document bootstrap draws with one simultaneous family
  over eight contrasts, three consumers, and 16 cells.
- Runtime: 208.24 seconds.

Every substantive runtime integrity check passed:

- candidate calls to the native MLP0 `Down`: 0;
- poison canary calls: 1;
- C512 proxy calls: 1,256;
- forbidden MLP1/MLP2 teacher calls in crossed suffixes: 0;
- all exact parent replays: zero raw-logit, capped-logit, and CE error;
- repeated C512-induced delta write: zero max error;
- carried $x_0$ and $v_1$ identities: zero max error;
- derangement: bijective, no same-document donors, wave/cell preserving;
- inherited capped-logit RMS: exact match;
- source, row, tensor, program, model, control, amendment, and authority hashes:
  exact match.

The V1 norm discrepancy was reproduced as an absolute error of
$9.765625\times10^{-4}$, but its maximum coordinatewise allowance ratio was only
$0.02319$. Hence every position satisfied the pre-existing scale-aware contract.

Independent replay of the result file reproduced the complete frozen inference
object exactly.

## Why the registered labels are inconclusive

The evaluator reported coverage with a float32 mean. The scorer recomputed the same
coverage exactly from integer sufficient-statistic counts, then required equality to
$10^{-12}$ absolute tolerance. The discrepancies were:

| scope | reported | ledger-exact | difference |
|---|---:|---:|---:|
| wave A | 0.9354501963 | 0.9354501856 | $+1.06\times10^{-8}$ |
| wave B | 0.9371935129 | 0.9371935096 | $+3.30\times10^{-9}$ |
| pooled | 0.9363523722 | 0.9363523836 | $-1.14\times10^{-8}$ |

Therefore `reported_coverage_matches_common_ledger` is false while every other
common gate is true. The immutable registered decisions are consequently:

- `mlp2_suppression_replicates = false`;
- `complete_compensation = false`;
- `aligned_mlp2_write_compensates = false`;
- all four component statuses = `inconclusive`.

Here `false` means “not promoted because a common gate failed.” It is not a
scientific falsification of each mechanism.

This is a numerical representation mismatch, not a row-support or model-integrity
failure. But it was discovered after the scientific ledgers were serialized, so it
must not be silently repaired into a promoted claim. The descriptive values below
are reported transparently and do not replace the registered verdict.

## The observed state-by-write pattern

Effects are the maximum over all three consumer families and 16 cells after division
by their practical margins. Smaller is better.

| contrast | wave A | wave B | pooled | pooled simultaneous 95% UCB |
|---|---:|---:|---:|---:|
| observational: `OO` vs `CC` | 0.784 | 0.790 | 0.787 | 1.333 |
| prewrite state: `OO` vs `CO` | 0.330 | 0.338 | 0.334 | 0.880 |
| write on exact state: `OO` vs `OC` | 0.587 | 0.588 | 0.588 | 1.134 |
| write on candidate state: `CO` vs `CC` | 0.610 | 0.612 | 0.611 | 1.157 |
| local state×write interaction | 0.070 | 0.070 | 0.070 | 0.616 |
| MLP2-omitted exposure: `O0` vs `C0` | 3.623 | 3.681 | 3.631 | 4.177 |
| shuffled aligned-write null | 0.568 | 0.580 | 0.574 | 1.120 |
| norm-matched native-write sensitivity | 0.340 | 0.347 | 0.344 | 0.890 |

The raw pooled worst-cell values make the scale clearer:

| contrast | KL | $|\Delta\mathrm{CE}|$ | centered-logit nRMSE |
|---|---:|---:|---:|
| observational | 0.005327 | 0.004593 | 0.03936 |
| prewrite state | 0.000807 | 0.001028 | 0.01671 |
| write on exact state | 0.002946 | 0.002728 | 0.02938 |
| write on candidate state | 0.003267 | 0.003693 | 0.03055 |
| interaction | 0.0000663 | 0.000255 | 0.003514 |
| MLP2-omitted exposure | 0.007332 | 0.027234 | 0.04323 |

### What is robustly visible

The omission contrast is large and stable: its maximum is over 3.6 practical
margins in each wave, driven by CE. The observational maximum is only about 0.79
margins. On the frozen simultaneous bootstrap, the reduction from omission exposure
to observational mismatch has a positive family-wise lower bound in wave A
($1.191$), wave B ($1.395$), and pooled ($1.752$), and observational mismatch is
pointwise no worse in every coordinate.

This is strong descriptive evidence that deployed MLP2 and the ensuing suffix
response attenuate most of the C512-induced error that is exposed when MLP2 is
omitted.

However, observational equivalence itself is not established: each wave UCB exceeds
1 and the pooled UCB of 1.333 exceeds the stricter 0.8 bar. So “MLP2 helps” must not
be upgraded to “C512 is now an equivalent causal interface.”

### What mechanism is *not* supported

The local state-by-write interaction point estimate is small (0.070 margins; pooled
UCB 0.616), but its registered component status remains inconclusive because the
dependency gate requires powered sensitivity. The aligned-write null is actually
better than the observational arm at the family maximum. The registered alignment
comparison therefore points in the wrong direction, and the generic sensitivity
contrast is not powered non-null.

The data do not support a story in which MLP2 recognizes the C512 state and emits a
special vector precisely aligned to repair it. Because the sensitivity control was
unpowered, this does not prove that aligned compensation is absent. A plausible
current hypothesis is that ordinary MLP2/downstream dynamics damp a distributed,
approximately additive state/write mismatch, but the assay was designed to certify
the stronger aligned-repair mechanism rather than identify every possible form of
attenuation.

## Consequence for MLP0 interpretation

C512 remains useful as a compact observational model of the MLP0 `Down` map, and the
physical MLP1 experiment still localizes its main internal error to the changed MLP1
write. The new descriptive result adds that deployed MLP2 and the ensuing suffix
strongly attenuate that error on natural FineWeb inputs.

What it does **not** add is executable or whole-model recovery credit. We still lack
a jointly priced program for the C512-conditioned MLP1/MLP2 interface, an OOD
certificate, and a selective edit/removal demonstration.

The next causal compiler should therefore target the small joint early-state/write
interface and compare it against a direct higher-rank MLP0 program at matched total
price. Repeating the same factorial on the now-spent rows or relaxing its gates after
seeing outcomes would add little trustworthy information.
