# Fixed-position entity-key, hop-key and answer-payload counterfactuals

Registered2026-09-09 14:29 UTC after the answer-history source test, before these
new text/intervention outcomes. Keep the reconstruction handoff's full goal.

The prior history/control sources differed in age. Hold positions fixed here and
change content fields independently. Same existing checkpoint and full-output
exported decomposition; no retraining. Generate64 cycle documents seed5909 and64
short-cycle permutation documents seed5910. The64 rows are balanced between hop2
and hop3. Binding order is random; query history has48 blocks. Designate source
query16, current query32, and nonmatching same-hop control query15. Fill other
queries with unique keys, excluding original/alternate/control key combinations.
There is exactly one previous matching query in the base document. Alternate entity
is(e+1)%24, control entity(e+2)%24; alternate hop switches2 and3. Choose a foil
different from both the original gold and the alternate-query true function answer.

Arms, all source/current positions fixed:
0 original; P replace source answer with foil; E change source entity key;
K change source hop key; EK change both source key fields; EKP change both source
keys and its payload; RETARGET change both source keys/payload and both current
query keys to match them; CONTROL change only query15's answer to the same foil.
All other tokens stay fixed. Incorrect stored facts are declared token interventions,
not examples claimed to satisfy the function generator. Teacher state recomputes
from every edited token sequence. No row filtering on teacher correctness.

pred_a_instrument: parser counts correct; all original-versus-folded full logits
close at1e-9 on all239 positions/29 outputs/eight arms. Base mean gold probability
>=.8 in every(population,hop) group; if capability fails, score it as a failure and
do not filter rows to rescue interpretation.

pred_b_key_payload_roles: in every(population,hop) group, P gives mean foil
probability>=.8; CONTROL increases foil probability by<=.1; E and K each lower
gold probability by>=.5; EKP increases foil probability by<=.1 with current key
unchanged. These distinguish composite-key retrieval from mere answer recency.

pred_c_joint_retarget: RETARGET gives mean foil probability>=.8 and raises it by
>=.5 over EKP, in every(population,hop) group. This is an unseen simultaneous
key-and-payload intervention, with the original function's alternate answer excluded
from the foil so that copying and fresh computation predict different outputs.

pred_d_factor_specialization (additional screen, not required for key/payload
semantics): capture each of four heads' q1,q2 at current query and k1,k2 at source
answer. Hold base q fixed for this diagnostic and compare source keys from E and K.
For each branch compute E_effect=RMS(qbase dot(k_E-k_base))/RMS(qbase dot k_base)
and K_effect analogously, with denominator floor1e-8. A specialization pairs one
branch with E_effect>=.5,K_effect<=.1 and the other with K_effect>=.5,E_effect<=.1,
allowing either orientation. Nominate qualifying head/orientation pairs on IID;
at least one must satisfy the same clauses on short-cycle OOD to pass. All heads,
both branches and actual query changes are reported. This anchored-query diagnostic
is not an independently executable token program and does not establish all routing
as a symbolic conjunction. If it fails, do not assign entity/hop meanings to the
architecture's separate factors from the product's semantic behavior alone.

Full output vectors are compared for compiler fidelity and counterfactual response;
the key/payload rule predicts which token is copied, not yet every probability.
The full-probability executor still contains priced native contextual weights.
Separate these two evidence levels. A positive next licenses replacing the matching
strength/payload computation with a smaller explicit program on fresh held-out data.
Failure preserves the source-dependence result without this stronger interpretation.

GPU via managed runner; batch4,128 docs x8 arms,1800seconds, tensors<256MiB.
No parameter/threshold search, no checkpoint changes, no new small-model or bilin18
completion claim. Native and compiled constants remain400640 and387968 respectively.
