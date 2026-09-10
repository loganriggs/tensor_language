# QK1/QK2/value dependencies of the saved correlative output partition

Original handoff/pilot and Logan's 10 September OV/QK input-sharing question.
Check modules/INDEX.md, specialist-heads.md, attn-middle-pooling.md, induction.md,
channels.md and circuits/MODULE_DOSSIERS.md. The induction dossier already
describes signed two-score conjunctions. The correlative route/value experiment
tested QK1*QK2 jointly, only through the saved scalar branch. Its partial
sufficiency failures remain closed. The OV pullback found extensive per-head
value-reader overlap, which does not establish full-block redundancy or explain
the complementary task. This screen measures the missing three-factor dependencies.

Freeze saved 26 heads, 14 q blocks and checkpoint. Reuse all existing A1/A2/C
16-row pairs; no fit, filtering, head/rank/gain changes or new OOD claim. For
each selected head, at its CURRENT recipient state after preceding edits:

    head = sum_source a(source)*b(source)*v(source),

where a and b are the two actual normalized, position-transformed native QK
scores divided by128, and v is the native mixed local/first-layer value vector.
Apply the causal mask. Mask bits1,2,4 replace a,b,v by the corresponding donor
factor. Recompute every unselected factor in the changed live trajectory.
The resulting head difference is projected through P=q q^T or R=I-P before
the original output weights. Donor factors come from the natural donor run;
query/source lengths and positions must be aligned before any intervention.

Seven nonempty factor subsets for each branch give14 interventions per panel.
Add native donor/base captures and two independent g.forward_units full P/R
donor swaps:18 forwards per panel,54 body forwards864 sequences total. Extra
factor contractions at26heads per forward are charged by execution time; all
545902902 native weights remain. Source maps/normalizations are not compressed.

Important symmetry: swapping QK1 with QK2 independently within any head leaves
the native function unchanged. These are labels of stored weight ports, not
canonical semantic names. A global score-half preference here would require
headwise/gauge-aware confirmation before promoting a circuit split.

Frozen predicates:

A. Instrument: exactly54/864, finite outputs, q-orthogonality<=1e-5; natural
factor reconstruction per block relative L2<=1e-5 and max error <=
1e-3+1e-5*maxabs(native); full-factor versus independent P/R native logits
maxabs<=1e-3 and relative L2<=1e-5. Recorded full-P/R raw recoveries replay
CORRELATIVE_COMPLEMENT_SPLIT_V1_RESULT within1e-3. The prior P routing-only
(mask3) and value-only(mask4) A1/A2 recoveries replay CORRELATIVE_ROUTE_VALUE_V1
within1e-3. Preserve invalid instrumentation rather than widening bars.

B. Parent dissociation: all48 native pairs correct at both endpoints and
positive cue denominators; P full recovery>=.8 on A1/A2 and abs<=.23 on C;
R full recovery>=.8 on C and abs<=.23 on A1/A2.

C. Different stored score-half dependencies: define the conditional loss of
recovery from keeping factor j live instead of donor,

    I_j = mean((margin_without_j-margin_full)/(base_margin+donor_margin)).

Use P on A1/A2 and R on C. This is in units of the complete native cue effect,
not a ratio dividing by a weak singleton. Let J=I_QK1-I_QK2. A/B must hold;
J_A1 and J_A2 have the same sign, J_C the opposite sign, each abs(J)>=.20,
and abs(J_A1-J_A2)<=.15. This tests differential dependence on the given
stored factor labels, not unique semantic factor identification.

D. Value dependence across both behaviors: A/B and I_value>=.50 separately
for P-A1, P-A2 and R-C. This establishes that each own behavior needs the value
change in this factorial; it does not prove they read the same scalar value
function. Previously known P route-only weakness is an anchor, not a new result.

Save margins for all14 arms, normalized recovery, full-branch vocabulary
errors, and complete endpoint factor-lattice interactions. For each branch,
compute every nonempty Möbius term Delta_S=sum_{T subset S}(-1)^(|S|-|T|)z_T,
with z_empty=native base. Terms add exactly to z_full-z_base; report squared
centered norms per row, individual terms, and total higher-order interaction.
These are whole-trajectory interactions, not the local trilinear coefficients:
later factors and the native suffix recompute after earlier edits. No additive
causal attribution from sums of norms or single marginals.

Post-result CPU: paired bootstrap C/D dependencies and interaction norms using
the existing shared helper, with no bars changed. C failure closes the simple
global two-score-half division of labor; it does not rule out head-specific
input modes. D failure rejects a common whole-value-dependence description for
the tested tasks. Keep the existing P/R swap result and removal failure intact.
