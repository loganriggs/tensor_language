# A reusable endpoint field in the composed-lookup model

We have identified an editable endpoint field in the four-attention-layer lookup
model. Exchanging it between two independent joins exchanged the intended answers
on every registered IID and OOD test case. This is a circuit-interface result;
the smaller, structurally explained full program remains unfinished.

## The computation and its consumers

Suppose the input contains u→v and v→w, with the v→w record appearing first.
Layer2, head1 (both zero indexed) sends the earlier record's endpoint w into the
later record. The final attention layer can use this field to answer composed
queries. For a three-edge chain a0→a1→a2→a3, the tested join stores the endpoint
a3 for the suffix a1→a2→a3; the three-hop query from a0 consumes that join.

The isolated contribution has an exact shared form:

    phi(w) = O[L2,H1] V[L2,H1] embedding(w) / 8
    message(t,s) = P[L2,H1](t,s) * gain[L2](s) * phi(w)

Here s is the earlier record's value position and t is either position of the
later record. P is the native, potentially signed attention score. The gain is
the native source RMS normalization factor. The factor1/8 accounts for the two
prefix residual mixtures and the L2 output mixture. phi is shared by every
consumer; its inputs are endpoint identities. The gate and gain still depend on
the contextual native prefix. All remaining producer terms and native paths are
retained as an explicit background.

This splits a native head by semantic record relations and embedding producer
terms, and connects embedding, L2 transport and final readers. It does not claim
that the entire head implements a clean equality rule. That earlier whole-head
replacement failed and remains rejected.

## Evidence against the four requested properties

| Property | Verified evidence | Remaining limit |
|---|---|---|
| Held-out/OOD prediction | Single and joint endpoint swaps answer correctly in all64 hop3 query cases across32 independent IID24-cycle/OODtwo12-cycle worlds; full edited-logit correspondence<=1.21e-13. | Fixed record layouts; contextual routing remains native. Arbitrary-layout semantics is now preregistered. |
| Extraction | Independently exported token-derived executor supplies its own states and weights. Endpoint formula reproduces native hooked paths<=7.77e-16 in the subsequent CPU audit. | Full native background retained; independent execution is not a smaller semantic model. |
| Removal/manipulation | Swaps selectively retarget. Removing each complete selected route reduces its own answer probability by.970–.991. | This removal includes more than the endpoint embedding term. Endpoint-only family removal is the next registered test. |
| Composition/reuse | Both endpoints can be exchanged simultaneously; all target answers remain correct. Final-query joint logit effect equals summed single effects within9.60e-14. | Scope is disjoint writes before one final attention layer. Other architectures and overlapping writes need the full nonlinear executor. |

Single-swap changes to the other query's mean original-answer probability are at
most.00697; joint lower-hop changes are at most.01845. These are group means,
not worst-case per-example guarantees. No model fitting or native-error filtering
was used. A separate contextual origin-field swap failed its80% accuracy bar;
do not generalize this endpoint result to that field.

Primary receipt: [endpoint swap](../JOIN_ENDPOINT_FIELD_SWAP_V1_RESULT.json).
Exact translation: [CPU compiler audit](../FORWARD_ENDPOINT_MESSAGE_COMPILER_V1_AUDIT.json).

## Literal cost and executable interface

The folded independent program retains387968 constants. Its12672-constant saving
relative to the400640-parameter checkpoint is the already known generic final
readout fold. The endpoint dictionary has24×128=3072 derived values,24576bytes in
FP64. It adds no independent coefficients, but removes no native coefficients:
those matrices remain needed by the background. There is currently no verified
structural-description saving. Prefix weights, routing, normalization, readers,
adapters and the arbitrary content of phi all remain charged.

`forward_endpoint_program_reference.py` provides `dictionary`, `joins`, and
`messages`. `joins(tokens)` parses all earlier v→w/later u→v record pairs.
`messages` uses the supplied endpoint map to form edited writes. Subtract native
messages from mapped messages and pass the difference to the exported source
editor's `edit(prepare(tokens), delta)`. Full background is computed from tokens;
no teacher activation cache is an input. The native-hook implementation is a
separate verification oracle.

The parser passes162 exhaustive tiny function/serialization cases, including
multiple consumers sharing one endpoint. Its all-join numerical audit uses
previously opened trained inputs; that algebra audit is not fresh semantic OOD
evidence. [The next protocol](../FORWARD_ENDPOINT_RANDOM_LAYOUT_V1_PREREGISTRATION.md)
tests all parsed joins on randomly serialized worlds, arbitrary endpoint maps,
endpoint-only removal and simultaneous even/odd consumer groups. No layout or
routing changes may be selected from its outcomes.
