# Rung 530: source-conditioned downstream bases inside the continuous attention0 computation

**Registered:** 2026-09-03 11:47 UTC

**Status:** prospective CPU analysis of already-open rung-480 sufficient statistics

**Claim level:** circuit-labelled interaction-basis screen, not a causal circuit, compression, or adoption

## Why this is a different question

Rungs 424/425 found a continuous attention0 computation with two six-dimensional score-factor coordinates and one
32-dimensional output coordinate. It predicts held-out attention0 effects very well. Rung 480 then used the 32 known
equality-circuit families to choose one downstream direction in each coordinate space. Those directions were stable
across model refits, but their circuit-effect profiles changed sign or became nearly orthogonal when the downstream
equality matcher changed from the native implementation `N` to the transplanted implementation `H`. A single
universal downstream direction therefore failed.

That failure is compatible with the user's interaction-defined-basis hypothesis: the same attention0 computation may
be read differently by two later implementations. Rung 530 asks whether each downstream implementation chooses its
own reproducible attention0 direction. It does not reduce rank, fit reconstruction, cluster architectural heads, or
reuse rung 480's failed universal gate.

Rung 529 independently showed why the distinction matters. A physical state average can predict the desired action
better than any one donor while still resembling a wrong-sign control on new documents. Therefore downstream
discrimination, not aggregate similarity, must define whether two candidate states mean the same thing.

## Frozen object

Use only `attention0_downstream_canonical_block_rung480_bundle.pt`, whose SHA256 is
`2401831045e5b269806a84d6308a941acc61a31ff4868ab9bc39904b0bea6967`. Its result and source hashes are
`e906cd94eb2d7a97ce6e3df59f9b9a6e270d81e027dc1251585a1d0374fbd9f8` and
`616aa6e103011598fac8ea710b023f7c1cbaf59d96115d17cf04ec14f508b577`.

The bundle contains no validation-circuit responses. It contains, for each of two document halves, two downstream
implementations (`N`: native reference; `H`: transplanted equality-score implementation), member/control masks, 32
discovery circuit families, and three attention0 coordinate spaces:

- first score factor: symmetric `6 x 6` response operators;
- second score factor: symmetric `6 x 6` response operators;
- output/payload factor: symmetric `32 x 32` response operators.

For mode `m`, half `d`, downstream implementation `s`, and circuit `c`, compute the member-minus-control mean operator

`A[m,d,s,c] = operator_sum[m,d,s,member,c]/member_count[d,member,c]
              - operator_sum[m,d,s,control,c]/control_count[d,control,c]`.

These operators transform by conjugation under a legal rotation of the internal coordinates. Projectors, projector
overlaps, and traces `trace(P A)` therefore describe the same downstream computation in every gauge.

## Source-conditioned construction

For every one of the six fixed `(mode, downstream implementation)` pairs, use only document half 0 to form

`G[m,s] = sum_c A[m,0,s,c] A[m,0,s,c]^T`.

Let `P[m,s]` be the rank-one projector onto the leading eigenvector of the symmetric part of `G[m,s]`. Rank one is
fixed only to test whether a distinct downstream direction exists; it is not a proposed compression rank. Independently
construct the same projector from half 1 and from rung 480's aligned second model fit.

The circuit fingerprint of a projector is the centered 32-number vector

`f[m,d,s;P][c] = trace(P A[m,d,s,c]) - mean_c trace(P A[m,d,s,c])`.

No circuit is selected or dropped. Report all six pairs.

## Controls and held-out checks within the open data

1. **Document-half transfer:** fit `P` on half 0 and predict its complete 32-circuit fingerprint on half 1.
2. **Model-refit transfer:** compare `P` with the independently fitted, Procrustes-aligned second-model projector.
3. **Circuit-label control:** for seeds `530300..530315`, independently permute half-1 circuit labels before
   computing fingerprint cosine. This preserves operator sizes and spectra while destroying circuit identity.
4. **Other-reader basis:** on the same mode and half-1 source, replace `P[m,s]` by `P[m,other source]`. If both bases
   make the same circuit predictions, downstream computation has not distinguished them.
5. **Leave-one-root stability:** the 32 circuit tags belong to roots `{0,2,4,6,8,18}`. Refit after omitting every
   root in turn and evaluate the omitted-fit projector on all half-1 circuits. This is a family-level stability test,
   not unopened validation.

## Frozen predictions

**A — exact CPU instrument.** All three authority hashes match; tensor shapes are exactly two
`2x2x2x32x6x6` families and one `2x2x2x32x32x32` family for each of the main/refit fits; counts match rung 480 with
minimum member/control support at least `39/439`; every operator and eigenpair is finite and symmetric within
`1e-10`; all 32 discovery tags are unique; and the bundle still states that validation responses are absent.

**B — at least one reproducible source-conditioned direction.** A `(mode,source)` passes only if:

- half-0 versus half-1 projector overlap is at least `.70`;
- main versus aligned-refit projector overlap is at least `.70`;
- its half-0 versus half-1 circuit-fingerprint cosine is at least `.70` and exceeds the permutation 95th percentile
  by at least `.15`;
- the fingerprint RMS is at least `1e-6` in both halves; and
- at least five of six leave-one-root projectors overlap the full projector by at least `.70` and retain half-1
  fingerprint cosine at least `.60`.

**C — downstream implementations choose different usable bases.** At least one mode has B pass separately for both
`N` and `H`, their projector overlap is at most `.50`, and for each source its own projector's half-1 fingerprint
cosine is at least `.15` higher than the cosine obtained with the other source's projector.

**D — the distinction is not carried by only one circuit root.** For a C-passing mode, both source projectors pass
the five-of-six leave-one-root rule, and omitting each root leaves the same mode as the strongest C candidate at least
five times out of six. Ties are resolved by the minimum of the two sources' fingerprint cosines; this rule is frozen
before computation.

`strong_null = not (A and B and C and D)`.

- A false: repair only the CPU reader.
- A true/B false: the interaction-conditioned rank-one direction is unstable; do not retry ranks or thresholds.
- B true/C false: downstream readers do not distinguish reproducible bases; retain only separate predictive screens.
- B/C true/D false: the apparent distinction is tied to one circuit family.
- A--D true: preregister a physical attention0 intervention on the still-unopened 30 validation circuits. That test
  must remove/replace each source-specific direction under both downstream implementations and show the predicted
  own-reader effect with smaller cross-reader damage than the other-reader basis.

Even a full pass identifies only a candidate interaction-defined basis. It saves zero values and cannot be called a
causal circuit until that physical, held-out intervention succeeds.

## Literal price

CPU only: read one `34.9 MB` bundle, diagonalize twelve matrices of maximum size `32 x 32` plus 36 leave-one-root
variants, and perform no model forwards or backwards. Zero parameters are fit beyond deterministic eigendecompositions;
zero values are added or removed from the model.
