# Successor pointer × prefix-coherence interaction V1

## Question and prior boundary

The reconciled successor record identifies a token-specific layer-0 value-cache
pointer read by a distributed successor lookup. Exact self substitution is
lossless, zeroing destroys the behavior, and wrong-pointer substitutions often
produce the imposed element's successor. Historical table extraction also
reports a nuisance: backward pointers are more often overridden by “continue
the coherent run” than forward pointers. That comparison was retrospective and
did not intervene on coherence.

This screen asks whether pointer identity and prefix coherence enter the output
as independent effects or as a causal interaction. It tests the interaction
object directly and does not fit a rank, reconstruct activations, tune gains,
quantize anything, rerun the L8 H3+H7 OOD screen, or claim a complete successor
circuit.

## Frozen rows

Before model execution, build every length-five comma-list prompt whose final
element index is 4–9 for months or 4–7 for digits. For each endpoint freeze
three prompts with the same five elements, final token, position, and correct
answer:

1. `coherent`: the elements are in increasing order;
2. `early_swap_control`: transpose the first two elements while preserving the
   final local transition;
3. `late_swap_incoherent`: transpose the two elements immediately before the
   final element, breaking its local coherent approach.

This yields 30 prompts: six per condition for months and four per condition for
digits. Each prompt has a forward donor (the next element) and a backward donor
(two elements behind), for 60 directed interventions. All prompt, pointer, and
answer strings must be single-token under the frozen GPT-2 tokenizer where the
interface requires one token. No native logits or earlier outcome tables choose
rows.

## Exact intervention and factorial object

At the final sequence-element position, capture the native block-0 value-cache
slice. In every attention layer 1–17 and every head, replace that slice with the
exact block-0 slice computed for either the same final token (`self`) or the
registered donor token. All prompt tokens and all other cache positions remain
fixed. The experiment therefore evaluates prefix condition × pointer identity,
with one native pass, one self-clamp pass, and forward/backward donor passes.

For every directed row, score the donor-successor minus recipient-successor
logit margin. The pointer effect is donor-clamp margin minus self-clamp margin.
For a family and direction, pair rows by endpoint and define the coherence
interaction as the median within-endpoint pointer-effect difference between
`late_swap_incoherent` and `coherent`. The early-swap difference is the matched
nuisance control. Normalize both by the median absolute native
recipient-versus-donor margin on coherent prompts. No coefficients are fitted.

## Registered bars and opposing predictions

Prediction A, instrument, requires finite values, exact row/count hashes, a
maximum native-versus-self logit difference at most `1e-5`, and every donor
pointer delta norm above `1e-3`.

Prediction B, answer-preserving prefix capability, requires native and self
recipient-answer accuracy at least `0.75` in every family × condition cell.

Prediction C, the coherent-run gate, requires all of:

- for backward pointers, the late-swap interaction is at least `+0.20` native
  margin units in both families;
- the backward donor-answer win fraction rises by at least `0.20` from coherent
  to late-swap in both families;
- the backward early-swap control differs from coherent by at most `0.15`
  native margin units in both families;
- forward late-swap and early-swap effects each differ from coherent by at most
  `0.25` and `0.15` native margin units respectively in both families.

Prediction D is the opposing independent-pointer model: all early/coherent and
late/coherent pointer-effect differences are at most `0.15` native margin units
for both directions and both families. If A or B fails, the instrument is
invalid. If neither C nor D passes, record a mixed interaction without repairing
rows, families, directions, thresholds, or intervention sites.

## Price and claim boundary

The maximum price is three batched model forwards and 120 sequence evaluations:
30 native, 30 self, and 60 donor-clamp examples. There are zero fits, backwards,
parameter updates, gain searches, rank searches, or quantization operations.
All model parameters, block-0 value computation, layers 1–17, and the full
vocabulary readout remain charged. A C pass establishes an interaction-level
gate target for localization; a D pass supports an invariant pointer operation
on these controls. Neither result alone identifies the downstream gate or
adopts a compressed executable circuit.
