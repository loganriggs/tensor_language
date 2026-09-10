# Draft: does the immediate MLP explain coupling between two shared-head writes?

**Status: NOT EXECUTED OR QUEUED.** Superseded as the immediate next action by Logan’s
10 September request to check module dossiers and fold the attention partition
back through OV/QK weights. This draft is not an execution authorization; its
module-specific prior-art audit remains incomplete.

Follow the original bilinear handoff and its appended structural criterion.
Previous projector/complement donor swaps separated two behaviors, while mean
removal failed selective preservation and donor endpoint effects were nonadditive.
This test asks about the immediate consumer of the two mean-removal writes.
Mean-removal endpoint interaction is measured afresh; it is not the previously
reported donor-swap interaction.

Use all existing recombined A1/A2/C rows, 16 each, no filtering. Freeze the
saved 26 heads, 14 block directions and original FIT means. No layer selection,
new fitting, ranks, gains, token selection or mean adjustment. C is its own
answer-changing either/not task. All native weights remain necessary.

Known prior art: semantic_square_bilinear.py already partitions observed MLP
corners into local products and inherited input interaction. Task14's MLP8
polarized response study already uses invariant bilinear polarization. The new
question is the causal role of the immediate MLP at the correlative live
projector/complement writes. Do not claim the algebra itself as discovery.

At each selected attention block, before its head edit, form the two residual
writes p = O P(mean-live) and c = O (I-P)(mean-live). Here P = q q^T; q has
orthogonality error <=1e-5. Compute both from the CURRENT heads after preceding
edits, including earlier MLP interventions. On the jointly edited trajectory,
the captured raw MLP input is r11. Define r00=r11-p-c, r10=r00+p, r01=r00+c.
These are local hypothetical corners on one common incoming context, not the
four independently propagated whole-model trajectories.

With B(x,y)=Down((Left x)*(Right y)), d(x)=mean(x²)+epsilon, the native consumer
is f(x)=B(x,x)/d(x)+bias. Its mixed response is

    M = f(r11)-f(r10)-f(r01)+f(r00).
    U = (B(p,c)+B(c,p))/d(r11).
    N = B(r10,r10)*(1/d11-1/d10)
        + B(r01,r01)*(1/d11-1/d01)
        + B(r00,r00)*(1/d00-1/d11).
    M = U + N.

The joint denominator anchors this split. N captures the remaining denominator
dependence under this convention; these are not unique causal responsibility
shares. Epsilon is the actual float32 RMSNorm default. FP64 synthetic controls
check direct normalized evaluation, the partition, zero writes and exchange of
the two writes. Local native bridges govern deployed float32 arithmetic.

Nine complete body forwards per panel: native donor; native base; P-only mean
replacement; complement-only mean replacement; full selected-head mean
replacement; full replacement with compiled immediate MLP joint output;
full replacement minus M; full replacement minus U; full replacement minus N.
Each intervention affects all 14 immediate MLP consumers at semantic positions
and recomputes the downstream model. Singletons retain native MLPs because a
local interaction of an absent write vanishes. Each altered trajectory recomputes
its own p,c,r and correction; no cached native offsets are added to changed state.

Price: 27 body forwards, 432 sequences, zero backward passes. Four compiled
arms each perform four raw-corner quadratic evaluations and two polarized
products at 14 MLPs on 16 semantic positions: 16,128 additional B(x,y) row
evaluations. Native MLP outputs are also computed to audit the substitution.
All 545,902,902 native parameters remain; no weight or structural saving claim.

Frozen predicates, scored separately in all three panels:

A. Instrument: counts exactly 27/432; finite outputs; orthogonality <=1e-5;
compiled-vs-native MLP element errors and partition errors <=1e-3+1e-5*abs(reference);
compiled-vs-full-mean endpoint maxabs<=1e-3 AND relative L2<=1e-5; all four
compiled arms visit all 14 blocks and apply finite, nonzero correction norms
for M,U,N (>1e-8). Synthetic controls pass.

B. Capability and parent replay: all 48 pairs correct at both native endpoints,
positive base+donor margin denominators; P/complement/full per-row mean-removal
CE changes replay the frozen parent within maxabs1e-3. No outcome filtering.

C. Immediate-MLP interaction explanation: A/B; native mean-removal interaction
is nontrivial (relative norm >=.10), and omitting M reduces its norm by at least
50% AND leaves <=.10 relative residual in every panel.

D. Numerator-product explanation: the same C bars with only U omitted.

E. Denominator-remainder explanation: the same C bars with only N omitted.

Endpoint interaction for candidate z is center(z-z_P-z_R+z_base). Normalize all
arms by the SAME norm of center(z_full_mean-z_base), floor1e-30. Norm reduction
is against the native mean interaction, floor1e-30. Save per-row squared errors,
base-effect norms, CE changes, margins and local correction magnitudes.

C failure closes sufficiency of removing these immediate local mixed responses
for this composed intervention. A decrease alone is partial mediation. D/E
failure closes the corresponding single-source explanation. Even a pass is a
conditional consumer result, not discovery of a context-free semantic circuit,
OOD generalization, selective removal repair, or independent extraction. Do not
repair failures by choosing layers or fitting cancellation coefficients.

Post-result CPU work: paired bootstrap intervals for fixed endpoint interaction
norms and compare correction-induced CE effects, with no change in bars. If the
immediate consumer is insufficient, use the saved local magnitudes to decide
whether a distinct upstream/live-routing versus later-consumer test is warranted;
do not infer a unique causal attribution from the algebraic split.
