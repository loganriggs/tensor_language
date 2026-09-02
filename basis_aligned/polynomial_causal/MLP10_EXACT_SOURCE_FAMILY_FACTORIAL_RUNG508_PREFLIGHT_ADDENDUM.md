# Rung 508 preflight addendum: account for direct replay and intact factor captures

Status: frozen during CPU implementation, before any rung508 model outcome or CUDA smoke was opened.

The original registration's scientific design, families, data, bars, and routes are unchanged. Its forward-price
formula omitted two required calls that the same text demands:

1. each singleton batch needs one direct native forward to verify exact analytical replay; and
2. a later joint-pair batch must recapture each of the four intact score trajectories to construct the exact family
   term changes before patching them.

Therefore a248-document singleton phase costs

`62 * (one direct + one absent capture + four sources * 23 arms) = 62*94 = 5,828` forwards.

Run discovery singletons first, then confirmation singletons. Only if2--8 terms confirm, run their joint arms on
both phases. A joint phase with `C=choose(q,2)` pairs costs

`62 * (one absent capture + four sources * (one intact factor capture + C joint patches)) = 310 + 248*C`.

The exact total after confirmation is consequently

`2*5,828 + 2*(310 + 248*C) = 12,276 + 496*C`,

at most`26,164` forwards for`q=8`, not`25,544`. If discovery returns outside2--8, stop after5,828. If discovery is
identifying but fewer than2 terms confirm, stop after11,656. Backwards, fitted vectors, and deployed-parameter
changes remain zero.

This correction is binding and must be hash-pinned by the implementation. It adds no arm and changes no scientific
outcome: both calls were already required by the registered numerical algebra.
