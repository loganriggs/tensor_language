# Prefix memory as the context operand of a source edit

Use the same16 original/fronted pairs and mature three-position MLP8 source
interface as SOURCE_STAGE_INTERCHANGE_V1. This tests a new context operand,
not a source-position/head/rank/response approximation rescue.

For each native layout, capture earlier-position K1,K2 and already mixed V at
attention9..17. These factor the memory M=sum_s k1_s tensor k2_s tensor v_s.
No dense accumulator is allocated. The first jointly informed token and the two
following tokens are excluded from memory. Source changes at those positions
cannot alter earlier states; their K1/K2/V are checked bitwise during execution.

At each recipient attention, keep its live within-suffix computation. Replace only
the prefix contribution for the last3 queries by the donor-memory read. Recompute
Q/Q2 from their current projection outputs, native QK RMS, and donor native rounded
RoPE phases at its last3 positions. Thus time is aligned relative to the mature
boundary. Do not rotate already multiplied states or freeze queries. Source first-V
and receiving residual/embedding/MLP operations remain live. This is a declared
hybrid context port, not a claim that the hybrid is a natural sentence.

Run all source producer p × memory context k × recipient r combinations. Install
B_r+T(C_p) at MLP8, and memory k at layers9..17; source components are unchanged from
the previous protocol. Z[p,k,r] is the resulting full vocabulary output table.
The SAME source-change response is d[k,r]=Z[1,k,r]-Z[0,k,r]. No removal-reference
baseline is needed. Native memory k=r reproduces the previous source-interchange
native/cross outputs, reconstructed from its saved effects and removal outputs.

A instrument: all four native-memory source/recipient grids match parent maxabs
<=1e-3 AND relative Frobenius<=1e-5 on three readers. Own source/own memory full
vocabulary matches native. Source before maturity exactly zero or relative<=1e-8;
source incoming, prefix K1/K2/mixed-V and first-V9 checked bitwise. Prefix attention
outputs unchanged; finite; hooks and overridden methods restored. CPU actual
native-attention-class controls: identity, unequal lengths, live source response,
independent prefix-loop oracle, direct/accumulator identity, invalid-prefix cleanup.

Q_oh-project d and measure correct margin, centered three readers, centered full
vocabulary. Every prediction needs every readout and both directions in all16 pairs.
B memory-following: d10 matches d11 and d01 matches d00, relative errors<=.10.
C remaining-context-following: d10 matches d00 and d01 matches d11, errors<=.10.
D small interaction: ||d11-d10-d01+d00||/max(||d00||,||d11||)<=.10.
Zero scientific denominators fail. All failures retained; no smaller-error winner.

Native capture4 + cube16 forwards per pair =320 forwards/5120 sequence instances,
zero fits, 600-second cap, managed GPU. There are2304 patched attention calls and
4608 prefix contractions, in addition to native operations; factor banks retain
14929920 FP32 scalars per pair (56.953125 MiB), plus6912 BF16 phase scalars. Scratch,
source captures and native weights are additional. All545902902 native parameters
remain, saving0. This is an operational context screen, not independent extraction.
If B passes, evaluate source/context generation and fresh combination reuse before
promotion. If B/C fail, close both simple context descriptions; no automatic layer
walk. An exact memory port by itself is the known recurrence, not semantic discovery.
