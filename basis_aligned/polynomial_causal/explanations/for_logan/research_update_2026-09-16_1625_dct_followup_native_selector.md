# September 16 follow-up: DCT briefing and a native subject-number selector

## Bottom line

I read `interaction_decomposition_briefing_v2.md` as a methodological proposal,
not as evidence that DCT by itself solves the circuit problem. Its most useful
constraints are: keep context modes open rather than averaging them away, select
components by causal response rather than weight energy, count every remaining
native-state port, and require prospective prediction, selective removal, and
behavioral composition separately.

The subject-number work now has a real model-native input node: a frozen linear
coordinate of the checkpoint token embedding decodes grammatical number. It
transferred perfectly to a disjoint 16-pair regular/irregular vocabulary panel.
Joining that decoder to the previously frozen direction-specific scalar lookup
and L11H3 rank-one axis produced the executable graph

`token ID -> checkpoint embedding -> number score -> direction -> fixed scalar -> fixed L11H3 write`.

On 16 newly frozen two-clause prompts, this graph selected all 32 directions
correctly without reading the row labels and its two independently installed
writes composed almost exactly. The registered overall outcome is nevertheless
a **near-threshold null**, because one of four single-site/direction cells had
effect RMS `.009893`, just below the preregistered `.010000` floor. That verdict
is preserved.

## What came from the DCT briefing

The literal DCT proposal is a causal-Hessian/tensor-factor search. The embedding
decoder experiment is not such a decomposition and should not be relabeled as
one. It addresses a different bottleneck exposed by the briefing's red-team
checklist: the earlier subject-number circuit still consumed an oracle direction
port.

The briefing did change the experimental order:

1. The earlier native-state generator was selected by downstream causal response,
   rather than by variance or a weight singular value.
2. Context was kept explicit during topology audits. A full-state prototype that
   looked good at one endpoint was rejected when its opposite endpoint missed by
   `61.74`; the corrected open `Y/Z` context prototype was a valid null.
3. Removal and composition were tested as separate causal properties. The
   response-weighted component passed selective removal, and its generated writes
   composed conditionally, but the generator itself shifted out of support on a
   new two-site topology.
4. That topology failure motivated moving the selector earlier, to a checkpoint
   weight interface that is independent of prompt topology: token embeddings.

This is consistent with the briefing's core lesson—an averaged or closed-context
factor can look compact while failing as a reusable node—but it is not yet the
proposed unsupervised one-layer Hessian CP experiment.

## Frozen embedding-number decoder

The decoder input is one RMS-normalized checkpoint embedding row. Across 48
singular/plural noun pairs divided into three pre-existing lexical families,
leave-one-family-out axes achieved `1.0` endpoint accuracy and `1.0` pair-order
accuracy in every held family. The frozen pooled axis then achieved:

- `1.0` endpoint accuracy on all 96 training endpoints;
- minimum signed training margin `2.091`;
- random equal-norm axis median accuracy `.5` over 64 axes.

After freezing, a disjoint 16-pair panel containing six irregular and ten regular
pairs also achieved `1.0` accuracy and pair ordering. Its minimum signed margin
was `1.443`, minimum pair delta `3.270`, and equal-norm random median was again
`.5`. This used zero model forwards, prompt activations, logits, behavioral
effects, gradients, or updates. It establishes lexical OOD decoding from native
weights, not that the model itself causally uses this exact axis.

[Frozen decoder](../../SUBJECT_NUMBER_EMBEDDING_DECODER_FROZEN_V1_ARTIFACT.json) ·
[fresh lexical result](../../../bilinear_quotient/circuits/fast_screens/subject_number_embedding_decoder_fresh_v1_result.json)

## Integrated token-to-write composition

The integration authority was frozen only after the lexical test, so its prompt
topologies are new but its subject words are the already-held lexical vocabulary.
It contains 16 rows, both templates, and all four ordered number pairs four times.
No row was removed after outcomes were opened.

The candidate read only subject token IDs and frozen checkpoint/artifact values.
The row labels were used by a separate oracle audit. Instrumentation passed:

- decoder accuracy: `1.0` over 32 sites;
- minimum signed decoder margin: `1.443`;
- candidate/oracle direction agreement: `1.0`;
- candidate/oracle write maximum absolute error: `0.0`;
- native answer accuracy: `1.0` in every site/number/template cell;
- zero-write replay maximum logit error: `0.0`;
- exactly 5 physical forwards over 80 sequences, with no fit, backward, or update.

Single-site causal effects were:

| Cell | RMS | Positive fraction | Control/target RMS | Registered pass |
|---|---:|---:|---:|---|
| site 1, singular to plural | `.10878` | `1.0` | `.01708` | yes |
| site 1, plural to singular | `.009893` | `1.0` | `.03138` | **no** (`.01` floor) |
| site 2, singular to plural | `.09723` | `1.0` | `.02681` | yes |
| site 2, plural to singular | `.01144` | `1.0` | `.03888` | yes |

The site-1 plural-to-singular values ranged from `.00737` to `.01243`; all eight
had the intended sign. The miss is therefore a small-amplitude boundary miss,
not a sign failure or one anomalous row. It is also expected to be the weak
direction: its frozen coefficient magnitude is `7.606`, versus `31.149` for
singular-to-plural. None of this licenses changing the registered floor.

Conditional on the writes being live, composition passed every registered bar:

| Scope | Cosine | Relative L2 | Sign | Norm ratio |
|---|---:|---:|---:|---:|
| overall | `.999999994` | `.0001071` | `1.0` | `.999987` |
| site 1 | `1.0` | `0.0` | `1.0` | `1.0` |
| site 2 | `.999999988` | `.0001603` | `1.0` | `.999972` |

The later-site-only edit had exactly zero effect on the earlier answer. Thus the
honest statement is: **the registered complete claim is null, while direction
extraction, selectivity, and two-site additive composition all held; one weak
direction missed only the absolute liveness floor.**

[Integrated result](../../SUBJECT_NUMBER_EMBEDDING_WRITER_TWO_SITE_V1_RESULT.json) ·
[preregistration](../../SUBJECT_NUMBER_EMBEDDING_WRITER_TWO_SITE_V1_PREREGISTRATION.md)

## Red-team boundaries and next experiment

The positive red team rules out the most plausible implementation mistakes:
dynamic positions came from the frozen manifest; the decoder exactly agreed
with a label-based oracle; candidate and oracle write tensors were identical;
zero writes exactly replayed baseline; all capability cells were perfect; and
the later edit was exactly anticausal-null. The negative red team retains four
important limitations:

- the decoder is supervised by grammatical-number labels, not an unsupervised
  identifiable DCT factor;
- the integrated prompts reuse the held lexical panel, so only prompt topology,
  not vocabulary, is fresh at this stage;
- cardinality `4` remains an external fixed context port rather than a native
  decoded gate;
- the integrated path has not itself passed selective removal against equal-norm
  causal directions. Earlier removal applies to the response-weighted L11H3
  component, not automatically to this complete selector-to-writer graph.

The focused next test should therefore remove the frozen embedding-number
coordinate at subject tokens, compare with equal-norm random embedding directions
at the same sites, and measure target agreement plus unrelated-behavior
preservation. If that does not selectively damage number agreement, the decoder
is a useful external extractor but not evidence for the model's own causal read
path. In parallel, the literal DCT program should stay local: validate the
one-layer weight/HVP Hessian identity and unsupervised reader recovery before
attempting a deep open-context tensor network.
