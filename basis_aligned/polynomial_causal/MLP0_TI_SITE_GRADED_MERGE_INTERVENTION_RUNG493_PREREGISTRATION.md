# Rung 493 preregistration: is the T/I distinction progressively merged across attention1 and MLP1?

Date: 2026-09-02 15:24 UTC  
Owner: Codex  
Status: frozen before rung493 model outcomes

## Why this follows rung492

Rung491 established an exact local attribution: attention1's residual write is the only named MLP1-input source
necessary for both the token-only (`T`) and token-by-context (`I`) branch responses. Rung492 then showed that a true
attention1 knockout materially changes both branch effects, but subtracting attention1 only inside MLP1 does not
reproduce that change and loses to position-shifted controls. The single attention1→MLP1 edge is therefore not a
portable circuit.

An independent analysis of rung483's exact finite-response Gram matrices suggested a different object. The fraction
of T/I response energy in their common `(T+I)` direction is `.507/.508` at attention1, `.623/.628` for direct MLP1,
and `.789/.791` for MLP1 after attention1 recomputes. T and I may begin as distinct signals and become progressively
merged by the coupled attention1/MLP1 computation. This rung turns that descriptive gradient into a physical
intervention.

This tests circuit boundaries, downstream grouping, and selective manipulation. It does not select a rank, quantize
weights, compress activations, or claim that smaller storage is interpretation.

## Exact branch and merge computations

For each batch, capture the native attention1 and MLP1 writes `A_N,M_N` and the writes when each exact MLP0 branch
`b in {T,C,I,S}` is absent: `A_b,M_b`. For a pair `(p,q)`, define the response writes

`d_A,p = A_N - A_p`, `d_A,q = A_N - A_q`,

and analogously for MLP1. Their common and distinction coordinates are

`common = (d_p+d_q)/2`, `distinction = (d_p-d_q)/2`.

Making the two absent trajectories share the same write means replacing both site writes by their arithmetic mean:

`A_p,A_q -> (A_p+A_q)/2`, or `M_p,M_q -> (M_p+M_q)/2`.

This removes the distinction coordinate while retaining the common coordinate. It is an exact output-site edit; no
learned fit or approximate direction is involved.

Three physical modes will be run:

1. `A_RECOMPUTE`: make the attention1 writes equal and let MLP1 recompute normally;
2. `A_DIRECT`: make the attention1 writes equal but restore each trajectory's original MLP1 write, isolating the
   direct residual path of the attention1 edit; and
3. `M_ONLY`: leave attention1 unchanged and make the MLP1 writes equal.

All six unordered pairs of `{T,C,I,S}` are run in the same-position condition. For T/I, `A_RECOMPUTE` and `M_ONLY`
also receive the 16 frozen position shifts inherited from rungs483–492. A shift moves only the write adjustment, not
the whole site output.

## Final CE-effect contrast

For a pair `(p,q)`, let

`x = [CE(p absent)-CE(native)] - [CE(q absent)-CE(native)]`.

The native baseline cancels, but it is still recomputed in-process. Let `y_mode` be the same difference after a merge
intervention. The part of the original distinction removed by that edit is

`r_mode = x-y_mode`.

Report:

- `aligned_removed_fraction = <r_mode,x>/||x||^2`;
- `removal_cosine = <r_mode,x>/(||r_mode|| ||x||)`;
- `residual_ratio = ||y_mode||/||x||`; and
- `removal_RMS = ||r_mode||/sqrt(number of scored tokens)`.

The aligned fraction is positive when the intervention removes the original T-versus-I effect rather than merely
adding unrelated damage. Position controls compare the same-position aligned fraction with the 95th percentile of
the 16 shifted adjustments.

For each site and pair, also report the write-space common share

`share_common = ||d_p+d_q||^2 / [2(||d_p||^2+||d_q||^2)]`.

This equals one when the two response writes are identical, one half when they are orthogonal and equally sized, and
zero when they are equal and opposite.

## Frozen data roles

- Discovery: census documents `0:500`, reported independently on `0:250` and `250:500`.
- Conditional validation: documents `500:1000`, reported independently on `500:750` and `750:1000`.
- Validation outcomes remain unopened unless all discovery clauses A–D pass.
- These are new intervention outcomes on a previously used corpus, not new-corpus OOD evidence.
- No final or otherwise reserved corpus is opened.

## Registered predictions

### A. Exact, lawful, live instrument

In each opened phase:

- all frozen parent hashes and the bilin18 checkpoint hash match;
- the native and four absent trajectories reproduce their captured MLP0, attention1, MLP1, normalized-state, and
  exact T/C/I/S identities at the inherited `1e-12`, `1e-8`, and `1e-5` bounds;
- the two same-position writes are bit-identical at the edited site in every pair/mode;
- every merge adjustment and every physical intervention is nonzero;
- native/absent replay and all dispatch/call counts are exact; and
- every unedited branch-effect contrast has nonzero RMS.

### B. Removing the attention1 distinction causally removes T/I effect contrast

In both discovery halves, T/I `A_RECOMPUTE` must have:

- removal cosine at least `.50`;
- aligned removed fraction at least `.20`;
- residual ratio at most `.95`; and
- same-position aligned removed fraction at least `.10` above the shifted-position 95th percentile.

### C. The distinction is progressively merged by the coupled block

In both discovery halves:

- T/I `A_RECOMPUTE` aligned removed fraction exceeds `M_ONLY` by at least `.10`; and
- T/I `A_RECOMPUTE` removal RMS is at least `1.25` times the `M_ONLY` removal RMS.

`A_DIRECT` is reported to distinguish the attention1 direct residual route from the additional change caused when
MLP1 recomputes. Neither subroute is required in advance to dominate.

### D. The gradient is T/I-specific rather than generic early-site attenuation

In both discovery halves:

- the write-space increase `share_common(MLP1)-share_common(attention1)` for T/I is at least `.15` and at least `.05`
  above every other branch pair;
- the physical `A_RECOMPUTE minus M_ONLY` aligned-fraction gap for T/I is at least `.05` above every other pair; and
- T/I is the unique top pair under both comparisons.

All T/C, C/I, and S-containing pair results are retained; an inconvenient control pair may not be dropped.

### E. Prospective intervention-outcome validation

Only if A–D pass, both validation quarters must independently reproduce A–D with the same T/I top-pair identity and
all numerical bars unchanged.

## Strong null and interpretation rules

The discovery strong null fires if A fails, if B fails, if the attention1 edit is not materially more consequential
than the MLP1-only edit under C, or if D shows a generic rather than T/I-specific depth effect. Validation stays
closed after any discovery failure.

A full pass would identify a site-graded circuit boundary: T and I are distinct at attention1 and become more
interchangeable by the complete attention1+MLP1 computation. It would not identify the semantic content of either
branch or license compression. The next step would test the frozen merge on new natural documents and code, together
with unrelated-behavior preservation.

A valid null retains rung491 only as local output attribution and the rung483 common-share gradient only as a
description. It routes to a different object: decompose attention1's exact QK1×QK2×value/output computation by
held-out downstream use across heads, or execute the independently frozen scalar-composition falsifier. Thresholds
will not be relaxed and rank reduction will not substitute for the failed causal grouping.

## Literal price

Per four-document batch, discovery runs five baseline forwards (native plus four absent), 36 same-position forwards
(six pairs × three modes × two sides), and 64 T/I shift-control forwards (16 shifts × two modes × two sides): 105
full-model forwards. One 500-document phase therefore costs 13,125 forwards; conditional validation costs the same.
No deployed parameters are saved or added.
