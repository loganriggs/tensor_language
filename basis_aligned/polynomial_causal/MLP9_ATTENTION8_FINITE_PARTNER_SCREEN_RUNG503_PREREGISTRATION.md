# Rung503 preregistration: finite raw-source partners of the attention8-driven MLP9 response

Registered 2026-09-02 19:30 UTC, after scoring rung502b and before opening any rung503 local-removal outcome.

## Question and claim boundary

Rung500 established that MLP9 distinguishes the shared `L5H5 score -> L8H4` copy operation from payload and wrong-score
controls. Rung502b found strong attention8-centered source-pair anatomy, but exact pair identity changed under two
allocations of BF16/RMS rounding and one common pair changed sign on held-out data. A third allocation is forbidden.

Rung503 asks a different, finite question:

> Which explicit upstream sources must remain in MLP9's raw input for MLP9 to turn the attention8 score change into its
> copy-selective response?

The experiment physically subtracts one upstream source from the exact deployed pre-MLP9 residual, performs the real
RMS normalization and BF16 MLP9 computation, and measures how the score response changes. This does not allocate a
normalized state among sources and therefore has no normalized-source gauge. A pass identifies a **finite local partner
set at MLP9**, not a complete circuit, portable variable, model compression, or permission to delete upstream modules.
A separately registered suffix intervention remains necessary.

## Frozen states, sources, and data

Reuse the hash-pinned checkpoint,1000 FineWeb rows, copy-task mask, score scale, payload scale, known positive
`L5H5 -> L8H4`, early-present/early-absent backgrounds, and32 discovery circuit tags from rungs500--502b. Rung503
must pin the rung502b preregistration, source, result, and bundle hashes and require its literal A/B=true, C/D/E=false
verdict. Rung502b's source-pair selection is not loaded into the selector.

Selection uses documents`0:248`, with fixed halves`0:124` and`124:248`. Confirmation uses`248:500`, with fixed
halves`248:374` and`374:500`. Documents`500:1000` and the30 validation tags remain closed in this rung.

At MLP9 there are20 registered raw contributions: `E`, attentions`A0...A9`, and earlier MLP writes`M0...M8`, with
the actual residual-mixing coefficients. `A8` is the known changing carrier and cannot be selected as its own partner.
`E` contains the exact raw skip complement needed to absorb sequential BF16 addition roundoff and cannot be assigned a
semantic partner meaning. The frozen eligible partner vocabulary is therefore the18 explicit sources

`A0,A1,A2,A3,A4,A5,A6,A7,A9,M0,M1,M2,M3,M4,M5,M6,M7,M8`.

No source may be subdivided, merged, ranked by norm, added, or dropped after outcomes.

## Exact finite computation

For background `b` and action state `a` in `{recipient_absent, score_donor, payload_donor}`, capture the exact deployed
BF16 residual `x[b,a]` immediately after attention9 and before MLP9 normalization. Let `r_t[b,a]` be the registered
float32 residual contribution of eligible source `t`, including its exact recurrence coefficient. Define the finite
source-removal state by the literal operation

`x_minus_t = (x.float() - r_t).to(BF16)`

and recompute

`W_minus_t = MLP9(RMSNorm(x_minus_t))`

with the deployed BF16 MLP9 weights. The unchanged write `W` is the write captured from the same action trajectory.
Every edited BF16 input reaching RMSNorm must differ from the native BF16 input by nonzero RMS. A changed input may
legitimately produce an identical MLP9 output; that is a scientific zero for the partner effect, not an instrument
failure.

For a fixed background, define the native score response

`Delta = W[recipient_absent] - W[score_donor]`

and its response after removing source `t` in both trajectories

`Delta_minus_t = W_minus_t[recipient_absent] - W_minus_t[score_donor]`.

The finite partner contribution is

`J_t = Delta - Delta_minus_t`.

This is exactly the part of the score response that disappears when `t` is removed with every other source present.
It automatically includes all RMS-normalization and higher-order dependence involving `t`; it is not a derivative or
an additive allocation. Define `J_t_payload` by replacing `score_donor` with `payload_donor` in the same formula.

For the selected partner set `G`, confirmation additionally computes

`J_G = Delta - (W_minus_G[recipient_absent] - W_minus_G[score_donor])`,

where `x_minus_G = (x.float() - sum_{t in G} r_t).to(BF16)`. This is an exact simultaneous finite removal, not the sum
of singleton `J_t` values.

Copy-loss gradients with respect to the recipient-absent MLP9 write may contract with `J_t` or `J_G` to measure local
sensitivity, but gradients cannot establish the finite partner claim by themselves.

## Frozen selection rule

On selection documents only, for every source and each background compute:

- response fraction `<J_t,Delta>/<Delta,Delta>`;
- copy-gradient fraction `<grad,J_t>/<grad,Delta>`;
- the analogous payload projection; and
- the signs of both score projections in each fixed document half.

Source `t` is selected if, independently in both backgrounds:

1. response fraction is at least`.01`;
2. copy-gradient fraction is at least`.01`;
3. the absolute payload projection is at most half the score projection; and
4. both the response and copy-gradient projections are positive in both selection halves.

Retain every passing source. Do not choose top-k. `B` requires the complete set to be nonempty and contain at most10
sources. More than10 is a diffuse local partner description and fails rather than being truncated.

## Frozen predictions

### A — valid finite instrument and parent response

All hashes, rows, masks, source names, intervals, actions, edits, and calls match. Native replays are exact; source
recurrence reconstruction is within the rung502b raw bound; every singleton removal and, if B opens confirmation, the
group removal make a finite nonzero change to the BF16 input reaching RMSNorm. Zero MLP9 output change remains an
allowed scientific null. The
selection phase costs exactly496 full-model forwards (`62 batches * 8`) and6,696 batch-sized local MLP9 evaluations
(`62 * 2 backgrounds * 3 action states * 18 sources`). If B passes, confirmation adds exactly504 full-model forwards
and7,182 local MLP9 evaluations (`63 * 2 * 3 * (18 singleton + 1 group)`), for totals1,000 and13,878. Copy/circuit
backward counts are computed from the frozen nonempty masks before model loading and asserted exactly. The unedited
score/payload parent must pass the rung502b B bars in every half/background and reproduce rung501 MLP9 cosine within
absolute`.03`.

### B — compact partners on selection

The literal selection rule returns a nonempty complete set of at most10 sources. All18 sources and all failure reasons
are stored. B does not use rung502b's pair names or their order.

### C — partner identity and finite group response confirm

Applying the same singleton rule without change on confirmation documents returns exactly the same complete source
set. Every selected singleton preserves positive response and copy-gradient signs in every confirmation half and
background. Without summing singleton effects, the exact simultaneous `J_G` must, in every confirmation
half/background:

- have cosine at least`.75` with `Delta` and positive-scale residual at most`.70`;
- project onto `Delta` at between`.50` and`1.50` of the complete response;
- have copy-gradient fraction between`.50` and`1.50`; and
- have payload-to-score projection ratio at most`.50`.

### D — the same finite group has selective downstream use

For every one of the32 fixed circuit tags, compute member-minus-control gradients at the recipient-absent MLP9 write
and contract them with `Delta`, `J_G`, its payload counterpart, and16 fixed token-position rolls of `J_G`. All32 tags
must have nonzero support in both confirmation halves. In each half/background, the group/full fingerprint cosine is
at least`.75`, positive-scale residual at most`.70`, group norm at least`.25` of complete, cosine exceeds the 95th
percentile position roll by at least`.10`, and payload cosine is at least`.20` lower. Store every coordinate and
unsupported tag.

### E — interpretation

E is true only if A--D hold. The selected `G` is then called a **finite raw-source partner candidate for the MLP9
copy-score response**. The next rung must remove the exact frozen group at MLP9 input or its exact finite MLP9
contribution, recompute layers10--17, and measure held-out copy behavior plus unrelated-circuit preservation. Rung503
alone cannot call `G` a circuit or claim compression.

## Nulls and routing

- A failure repairs only the finite source-removal instrument.
- A true/B false means no compact singleton partner set exists. Next run a separately registered finite pair-removal
  interaction screen or the labeled float32 explanatory control; do not lower bars or rank sources.
- A/B true/C false means partner identity or its aggregate response is document-dependent. Do not take the favorable
  intersection; use the float32 control or change the observation.
- A--C true/D false means finite local MLP9 anatomy exists but the current32 downstream measurements do not identify
  its use. Preserve the anatomy and change the downstream observation before a circuit claim.
- A--D true licenses only the separately registered held-out suffix intervention in E.

This rung saves and adds zero deployed parameters. Rank, quantization, reconstruction-only scoring, post-outcome top-k,
and another normalized-source allocation cannot pass any clause.
