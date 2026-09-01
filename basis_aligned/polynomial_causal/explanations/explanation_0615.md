# Plain-English update — 2026-09-01 06:15 UTC

**The one-sentence headline:** choosing attention directions by what the model actually receives on text, rather
than by raw weight size, has produced our strongest compressed model yet: it is smaller than the former
high-fidelity point, has about one-third of its census error, preserves 61/62 circuits, and reproduces a direct
causal intervention almost exactly.

## Our goal

We are not merely looking for a low validation loss or a small checkpoint. We want a compiled tensor program that
is simultaneously:

1. **predictive** — it matches the model on new text and shifted corpora;
2. **composable** — useful replacements remain useful when installed together;
3. **manipulable** — interventions have the same signed effects in the compiled and native models; and
4. **literally simple** — every stored tensor, index set, and fallback is included in the scalar and byte bill.

The broader scientific aim is to recover the model's executable structure, not just approximate its outputs.

## What we found

Each replaced query/key map is a matrix `W`. Ordinary SVD keeps directions with large Euclidean weight energy.
That was the wrong geometry. If `C = E[x x^T]` is the covariance of the vectors that actually enter the map, the
relevant approximation solves

`min_rank(W_r)<=r E ||(W-W_r)x||^2 = min_rank(W_r)<=r ||(W-W_r) C^(1/2)||_F^2`.

So we take the rank-96 SVD of `W C^(1/2)` and store the corresponding two factors after mapping the input factor
back through `C^(-1/2)`. We do this for all 440 replaced query/key maps in layers 2–17. This removes the old
hand-selected eight-direction “fine band”: that band was compensating for a metric error, not identifying a
special irreducible circuit.

The discovery fit gave `+.001245` census damage and 62/62 certificates. A disjoint covariance fit reproduced it at
`+.001415` and 61/62. On a later WikiText segment, mean damage was `-.001674`, with p95 `+.007033` and worst row
`+.015147`; the slightly negative mean should be read as indistinguishable from zero, not evidence that compression
improves the language model.

## The newly adopted point

The fixed independent-fit artifact is:

- **535,089,462 scalars / 2,024,415,852 raw bytes**;
- census damage **`+.001415`**, with **61/62** circuit certificates;
- fresh-window maximum **`+.0014`**;
- shifted WikiText mean/p95/max **`-.001674 / +.007033 / +.015147`**;
- direct a16 knockout cosine **`.997617`**, normalized error **`.074857`**;
- collateral-circuit Spearman **`.998265`**, a16-own median effect ratio **`1.029761`**.

Every preregistered baseline, identity, price, OOD, and signed-effect gate held, and the null did not fire. This
strictly dominates the former 539,595,062-scalar high-fidelity point (`+.004692`, 54/62): it is 4,505,600 scalars
smaller, substantially more accurate, and preserves seven more certificates.

## What just landed

The immediate test combined this context-QK96 program with the independently adopted MLP0 context-RRR rank-448
program at **529,117,494 scalars / 2,000,527,980 bytes**. It landed at `+.009585` and 49/62 certificates. Simple
additivity predicted `+.009385`: the residual is only `+.000200`, or a 1.021x ratio. New WikiText
mean/p95/max is `+.001609 / +.028941 / +.049312`, and every frozen physical gate held. This is the first evidence
that the recurring ~1.3x tax may be mainly **within** a map family; Q/K and MLP0 residuals appear much closer to
orthogonal. One result is not yet a law. The exact saved artifact then passed its direct signed a16 gate with
cosine `.994186`, normalized error `.113081`, collateral rho `.997653`, and own-effect ratio `1.028231`.
It is formally adopted. The fully gated Pareto frontier now has only two non-dominated points:

- 535,089,462 scalars, `+.001415`, 61/62 certificates — high fidelity;
- 529,117,494 scalars, `+.009585`, 49/62 certificates — substantially smaller.

## Independent next paths

The result changes the ordering of our next experiments:

1. **Context-QK rank ladder.** Test ranks 88, 80, and 64 under the same independent covariance. This avoids
   cross-module tax and asks how much of rank 96 is genuine functional dimension. Every eight ranks removed saves
   exactly 4,505,600 more scalars.
2. **Dual-context MLP composition ladder.** If rank448 composes, test MLP0 ranks 512 and 640 with context-QK96.
   These cost about 529.78M and 531.11M scalars and may become higher-fidelity Pareto points.
3. **Joint or downstream-aware metric.** Fit projections against downstream loss or joint residual action, rather
   than independent local squared error. This is the most direct mathematical attack on the repeatable 1.3x
   composition tax; sequential covariance refitting has already failed, so the objective—not stale inputs—must
   change.
4. **Tail-robust covariance.** Lower MLP0 ranks failed only on rare WikiText rows. Reweighting contexts by leverage,
   loss gradient, or a minimax tail objective could preserve those rare directions without returning all ranks.
5. **The embedding-folded MLP0 structural route.** Exact position-zero token inputs still make MLP0 unusually
   tractable, but toy experiments showed that block/tree/DAG support is gauge-nonidentifiable from held-out R2:
   a wrong structural prior can fit equally well. Reopen this route only with an external discriminator such as
   intervention transfer, OOD behavior, or a literal price advantage. Finite MoE router states remain legitimate;
   generic per-token top-k remains a compute policy, not a small tensor network.
6. **Vocabulary sharing with sparse exceptions.** The shared input/output vocabulary code showed real signal but
   concentrated error on unseen rare targets. A fit-selected sparse residual is still an independent large-upside
   route, though the context-metric QK ladder is now better evidenced and cheaper to test.

The strategic step-back is simple: the strongest repeated law is now **functional rank is metric-dependent**.
The main risk is no longer finding low-rank maps in isolation; it is preserving rare contexts and controlling
interactions when several locally good maps are composed.
