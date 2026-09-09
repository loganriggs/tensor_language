# Upstream producer/final factor atlas — 2026-09-09 14:52 UTC

Question: which actual contextual producers supply the final answer-history router
and value? The fixed hop-state code failed on both query and key sides; do not add
features or rescue one side. Trace native dependencies before guessing their algorithm.

For the existing small attn/MLP/attn checkpoint, write the final residual input as
`x = E + A + M`, where E is the scaled original embedding, A is the first attention
write including its residual coefficient, and M is the additive MLP write. Compute
all three from tokens on each execution. With final RMS gain g(x), each bias-free
projection is exactly `W(gE)+W(gA)+W(gM)`. One producer decomposition and one live
gain serve all five Q1/K1/Q2/K2/V readers. Native absolute cached RoPE stays intact.
This is an exact causal interface, not a new compact semantic operation by itself.

Remove each of3 producer contributions from each of5 final projections (15 arms),
plus remove the same producer from all5 projections jointly (3 arms), plus native.
All4 heads and all positions are included, fixed before outcomes. The final residual
skip stays native. Critically these are **reader-edge cuts conditional on native live
RMS gain**, not upstream state interventions: gain is generated from the complete
token-derived x even when an individual normalized contribution is removed. Report
the distinction; do not claim the upstream producer itself has been removed.

Fresh panels:64 IID cycle documents seed7909;64 short-cycle permutation documents
seed7910. No teacher-correctness filtering or fit. Use the existing source parser to
stratify higher-hop repeated queries versus novel hop1 queries. Report full29-way
logit/probability changes, all15 single-edge and3 joint effects, group sizes, native
capability, and head-agnostic signed gold-probability losses.

Opposing predictions:

1. Instrument: reassembled E/A/M normalized projection equals native at1e-9; direct
   token-to-logit replay agrees at1e-9; explicit independent hooks agree with the
   producer executor for single and joint synthetic cuts. Test causality and live
   contributions. Native repeated higher-hop gold probability>=.8 in both panels.
2. Selective path: at least one fixed producer/reader pair loses>=.25 mean gold
   probability on repeated higher hops while absolute novel-hop1 change<=.10,
   with the **same pair** passing both populations. This nominates a path only;
   count all15 tested pairs and do not call a passing panel fresh confirmation of
   a selected pair. If none qualifies, preserve distributed-dependency result.
3. Joint dependence: for each producer, compare measured all5-reader removal with
   the sum of five single-reader logit differences. Relative RMS mismatch<=.01
   (or absolute<=1e-8 for native joint RMS<1e-6) would support additive accounting;
   otherwise the explicit product interaction must remain in the circuit. No
   probability-additivity assumption.

This diagnostic does not reduce arbitrary constants, export a new circuit, or
satisfy the whole goal. It can identify or falsify a concrete cross-module path
that the next algebraic operation must explain. No rank, head, token-state or
threshold sweep. Reuse loader/source parser/scorer; one finite atlas, batch4 FP64,
1800s guard, <256MiB per analysis tensor, GPU exclusively via managed bqrunner.
