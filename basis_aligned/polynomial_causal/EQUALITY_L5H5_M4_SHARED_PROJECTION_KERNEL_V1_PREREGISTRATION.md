# Equality L5H5 M4 shared-projection interaction kernel V1

The validated explicit interaction graph evaluates the residual-to-L5H5 score
node four times, using 16 Q/K projections.  This cheap-first experiment tests a
direct shared-projection kernel before any behavioral rerun.

For exact baseline, child, remainder, and joint residual ports, form baseline
`b`, child delta `c=child-b`, and remainder delta `r=remainder-b`.  Project only
`b,c,r` through each of the four L5H5 Q/K matrices (12 projections).  Construct
the four raw projected states by addition, carry each exact residual's RMS
denominator as an open context scalar, then apply the native Q/K RMS, rotary,
causal score, and Möbius cross-difference.  Joint residual rounding not captured
by `b+c+r` is intentionally omitted; measured error prices that approximation.

Frozen gates:

1. The inherited four-score graph is lawful and exact.
2. Shared baseline score error is at most `.01` and cosine at least `.999` on
   natural and code; failure here marks the shared instrument invalid.
3. Shared joint score error versus the four-score authority is at most `.01`,
   cosine at least `.999`, and both code-half errors at most `.015`.
4. Shared interaction error versus the authority interaction is at most `.10`,
   cosine at least `.95`, and both code-half errors at most `.15`.
5. The shared four-node graph closes within `2e-6`, uses 12 rather than 16 Q/K
   projections, and has zero learned parameters.

Natural rows are diagnostic only; nothing is fitted or selected.  Passing this
receipt authorizes a prospective behavioral replay/removal experiment.  Failure
after a valid baseline is a shared-kernel compression null, not grounds to alter
the interaction graph.

Price: one checkpoint load, 192 natural and 192 code prefix documents, no full
behavior forwards, gradients, fits, parameter updates, or new text.
