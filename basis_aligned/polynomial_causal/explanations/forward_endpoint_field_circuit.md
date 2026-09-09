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

## Random-layout consumer generalization fails for hop2

`FORWARD_ENDPOINT_RANDOM_LAYOUT_V1_RESULT.json` tests3072 variants from32 fresh
worlds with arbitrary binding orders and all parser-selected joins. Numerical
correspondence passes1.42e-13; endpoint-message oracle1.11e-15, identity0,
finalquery composition<=2.14e-13. Native eligible accuracy is.973–1.0.

The registered combined hop2/hop3 hypothesis fails: endpoint-only removal changes
hop2 goldP by only.0086–.0119, and mapping retargets only5.5–6.2%. The receipt's
`instrument_invalid` terminal reflects this failed causal-route requirement
inside its broad A gate. It does not indicate a mechanical or numerical defect.
The separate `FORWARD_ENDPOINT_RANDOM_LAYOUT_GATE_AUDIT_V1.json` shows this
classification without changing the original receipt or promoting the family.

Predeclared hop3 groups separately show.863–.881 joint retarget accuracy and
.883–.888 endpoint-only removal goldP loss. Every hop3 parity target bar and all
raw control/composition bars pass. These useful subgroup results do not turn the
combined hypothesis into a pass; no layouts or native errors are dropped.

A CPU diagnostic on the first4 opened worlds/population reconstructs all saved
native/mapped query outputs within2.77e-13. Final heads0 and3 carry most of the
hop3 response, and neither alone suffices (single retarget.33–.45). Hop2 centered
logit changes remain large (RMS3.99–4.74 versus hop3 5.19–5.33). Thus low hop2
answer damage must not be described as absence of a numerical reader response.
This diagnostic nominates no adopted head subset.

The next [dual-value-field protocol](../DUAL_VALUE_FIELD_INTERCHANGE_V1_PREREGISTRATION.md)
separately edits original binding value E/8 and computed endpoint messages at the
same postL2 positions. Distinct endpoint maps distinguish consumers; complete
source normalization preserves their overlapping interactions. Raw-value-only
causal strength is untested and may fail through redundancy. Mechanical and
semantic gates are reported separately. Seven CPU controls pass; managed trained
integration remains. No structural parameter saving has been established.

## Original and computed fields are not independently editable as specified

`DUAL_VALUE_FIELD_INTERCHANGE_V1_RESULT.json` completes15.05s with mechanical A
and joined-field C true, raw-field B and joint D false. All-position native
correspondence is1.99e-13; captured prefix identity3.55e-15. Removing literal raw
E/8 has strong task effects (.611–.768 goldP loss), but its mapped endpoint is
correct only.620–.711 of the time. Raw edits also damage the joined consumer
by.210–.240 mean goldP. Joined-only interchange passes its independent gate;
joint forwardhop3 retargeting falls to.725–.728. The two named state components
therefore fail the registered independent-field semantics. No scale or producer
expansion is adopted. Their overlapping raw-logit interaction RMS is2.018–2.112.

`DUAL_VALUE_NORMALIZATION_AUDIT_V1.json` is a diagnostic on the first4 opened
worlds/population, not another semantic confirmation. Live outputs replay within
3.27e-13. Holding final source gains fixed reduces interaction RMS from2.64/2.52
to1.84/1.77, but raw retargeting remains.646–.787 and joint outcomes still fail.
Those RMS changes are not additive variance fractions. Frozen gains are neither
a native execution claim nor an adopted repair. Normalization does not by itself
explain the failure.

The next [source-port protocol](../SOURCE_PORT_FIELD_INTERCHANGE_V1_PREREGISTRATION.md)
tests all K1/K2/V source-port combinations on fresh worlds. V-only is the fixed
candidate, allowing us to distinguish an edit's content effect from its changes
to source selection. It preserves query projections and residual skip at every
position and checks the corresponding native projection-hook oracle. Consequently
full-port finalquery agrees with physical source edits, while changed binding
positions generally differ. Nineteen CPU checks cover every port subset, live
edits, restored hooks and this scope distinction. Managed integration remains;
no alternate port subset may be selected posthoc. All opaque weights and caches
remain charged, and structural simplification is still unachieved.

## Value-only interface also fails; selected-field branch closes

`SOURCE_PORT_FIELD_INTERCHANGE_V1_RESULT.json` completes12.58s with A/C true and
B/D false. Full native port correspondence1.71e-13 and all-port finalquery versus
physical-state1.42e-13 validate the instrument. V-only raw retargeting is.615–.750,
with removal loss.569–.790. Raw V-only edits damage the joined consumer by
.168–.203 goldP; joint joined-answer accuracy.742–.755 fails. Key-only cells give
almost no retargeting, and adding keys does not consistently improve it. No
alternate factorial cell is adopted. The joined interface remains supported
within its registered scope; the independent raw/join abstraction does not.

The selected-field interchange branch is now closed as specified. An exact
[payload-lineage audit](../FROZEN_PAYLOAD_LINEAGE_V1_AUDIT.json) changes the object
to exhaustive input-root transport, rather than enlarging a failed field and
calling it identified. Under native attention/RMS gates, key roots, local value
roots, transferred value roots and query roots reconstruct the prefix to2.84e-14.
Causality makes the own-position lineage an exact self/residual recurrence;
individual-root comparison error is0. On eight opened worlds its cosine with
bare E/8 is only.629–.654. This is a missing-lineage diagnosis, not adoption of a
larger raw field. Gates remain opaque; the frozen transport is not the actual
input Jacobian because it excludes changes to routing and normalization.

The next [whole-model gate-reuse test](../GLOBAL_ENTITY_GATE_REUSE_V1_PREREGISTRATION.md)
uses graph-preserving entity renaming and a complete old/new payload×gate
factorial. It tests full distribution and causal-effect prediction, not just task
accuracy. Its primitives pass9 CPU controls; trained integration remains. This
is a different global reuse hypothesis, with all native coefficients and caches
charged. Structural simplification remains unachieved.
