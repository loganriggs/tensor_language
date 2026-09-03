# Rung 536: multi-counterfactual circuit identification

**Updated:** 2026-09-03 14:10 UTC

## Decision

DAS is only useful here after the causal variable and valid counterfactuals are specified. Each proposed circuit
must have multiple independently constructed interventions. An activation subspace trained on some constructions
must predict held-out constructions without refitting. The final test is the downstream causal effect, not the rank
or similarity of raw basis vectors.

The next few days should build a reusable circuit evidence base. CPU work—dataset construction, matching, split
freezing, and red-teaming—can proceed in parallel. GPU site tests and projector fits remain serialized through the
managed runner.

## Corrected audit of the 62 BATTERY masks

The first inventory mixed circuit cards from two different census trees. That is invalid because circuit tags are
tree-instance-local. The BATTERY masks come from `diverse-1000row-v1`, but 44 of their same-named registry cards
describe `212row-v1`.

| Current BATTERY status | Circuits |
|---|---:|
| card belongs to another tree; identity must be remapped | 44 |
| tree-matched card has a descriptive membership program but no validated computation | 18 |
| explicit paired counterfactual contract | 0 |

Before applying the tree-identity gate, the registry cards contain one computational story, five surface stories,
31 descriptive programs, and 25 unnamed clusters. The computational and all five surface stories are among the 44
old-tree cards, so they cannot label current BATTERY masks without a member/behavior remap.

The masks also do not represent 62 mutually exclusive circuit states:

- all 1,891 pairs overlap;
- intersection sizes range from 4 to 1,599 positions, with median 38;
- 24 pairs are exact containments;
- among the 52 masks of size 864, all 1,326 pairs overlap (median 26, range 4–334).

This supports organizing masks into candidate behavior families and parent/child relationships before making one
dataset per leaf.

## What counts as several valid counterfactuals

There are three distinct roles:

1. **Interchange:** change the proposed causal variable and the correct answer. This tests whether exchanging the
   state makes the model behave like the donor example.
2. **Necessity:** remove evidence for the variable while leaving the original answer fixed. This tests whether the
   evidence mattered, but there is no unique new answer to recover.
3. **Invariance:** change nuisance details while keeping both the variable and answer fixed. This tests whether the
   learned subspace ignores shortcuts.

The earlier induction `match_break_payload_preserved` design is a necessity test, not an answer-changing
interchange. The corrected contract records these fields separately and rejects inconsistent labels.

## Four initial behavior pilots

| Pilot | Independent constructions | Main failure to rule out |
|---|---|---|
| induction selector × payload | two-valid-source selector swap; payload swap; natural pairs; match necessity; lag/filler invariance | generic token identity or copy service instead of selector/payload computation |
| pending-opener state | opener insertion/deletion; opener-type substitution; closer reset; distance/filler invariance | punctuation identity, position shift, or recency steering |
| successor pointer | last-element swap; coherent whole-sequence shift; internal pointer imposition; prefix invariance | family/token identity or coherence rather than successor lookup |
| increment state | coherent number shift; cross-format transfer; incoherent necessity edit; surface invariance | generic digit/value state rather than increment computation |

IOI remains useful, but is deferred until it has a larger template/name split and a verified intervention-site
ceiling. Existing evidence localizes important attention heads, not yet the proposed MLP product sites.

The validated registry now contains 17 intervention families: 10 answer-changing interchanges, three necessity
tests, and four invariance tests.

## Identification and promotion

For one declared variable:

1. verify base and donor endpoint correctness on a sealed split;
2. measure the native/full-product interchange ceiling at every candidate site;
3. freeze site and rank;
4. perform leave-one-family-out training and causal testing;
5. compare family-specific projectors to a shared-plus-private model;
6. require unrelated behaviors and answer-preserving controls to remain stable;
7. test selector × payload or state × context interactions where applicable;
8. only then compile the projector into quadratic weights and rerun every family and control.

For projectors $P_a$ and $P_b$, a useful secondary distance is

$$
d_{\rm response}(P_a,P_b)
=\frac{\left\lVert W_D(P_a-P_b)\Sigma_\Delta^{1/2}\right\rVert_F}
{\tfrac12\left(\left\lVert W_DP_a\Sigma_\Delta^{1/2}\right\rVert_F+
\left\lVert W_DP_b\Sigma_\Delta^{1/2}\right\rVert_F\right)}.
$$

This is gauge-invariant and discounts differences the MLP output map cannot read. It is diagnostic only; signed
held-out causal transfer through the nonlinear model is authoritative.

## Bottlenecks

1. **Circuit identity:** 44 BATTERY cards require cross-tree remapping; all masks overlap.
2. **Counterfactual validity:** edits can change grammar, difficulty, token frequency, position, or coherence.
3. **Intervention-site ceiling:** a good dataset is useless at a site that cannot carry the relevant state.
4. **Interactions/redundancy:** single-component patching mixes a mediator's own effect with interactions.
5. **Power:** existing successor and bracket instruments are small; high-dimensional fits require expansion.
6. **Storage:** a full 4,888 × 256 × 4,608 BF16 product cache is about 11.5 GB per layer; stream or store selected
   task positions and sufficient statistics.
7. **GPU scheduling:** one RTX 5090 means model jobs are serialized even while CPU dataset work is parallel.
8. **False minimality:** low rank is not success without OOD prediction, circuit extraction, selective removal,
   reusable composition, and stable exact weight compilation.

## Organization rule

Every promoted circuit has one canonical version-2 JSON record under `circuits/`, containing: declared variable; all
counterfactual families and their roles; row/split authority; site ceilings; trained rank and seeds; cross-family
transfer matrix; interaction tests; unrelated-behavior damage; weight-compilation results; immutable failed/null/
invalid events; and hashes/links for every artifact. `registry.json`, `CIRCUITS_INDEX.md`, and `DOSSIER.md` are
generated views. `REPERTOIRE.json` remains a legacy snapshot rather than a second hand-edited source of truth.

Each evidence event has a design key over the causal question and an execution key over the split, seed, checkpoint,
and exact artifacts. New work must query the generated index before execution and explicitly mark a duplicate design
as a supersession or replication.
