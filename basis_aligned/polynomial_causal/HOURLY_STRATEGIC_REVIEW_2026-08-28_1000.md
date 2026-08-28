# Hourly strategic review: compose the owned tensor core

Date: 2026-08-28 10:00--10:25 UTC

Status: role-free composition work plus evidence consolidation. No new corpus, final,
selection, or promotion authority.

## Fraction explained

- Structural component inventory: 36/36.
- Named behavioral explanation: $32.1\%\pm6.4\%$.
- Named causal recovery: 10.923%.
- Strict simplified whole-model recovery: still 0%; no compressed MLP or fully owned
  top-level program has passed.
- Exact executable component ownership: all 36 attention/MLP components now compose
  bitwise with zero native component calls. Their 430.004M values are 78.77% of exact
  stored model values.
- Attention simplification: shared-QK-384 retains 99.44% of the attention stake while
  saving 55.738M values, 10.21% of complete stored model values if crossed with dense
  exact MLPs.

These quantities remain distinct. Exact ownership is not semantic explanation;
attention-subsystem recovery is not whole-model recovery; and the named circuit ledger
uses a separate causal denominator.

A late diagnostic resolves the earlier covered-position anomaly and narrows the strict
denominator further: every S1747--S1758 local "program" is a pure current-token
function once installed at all 36 sites. Those values measure a per-token null, not
contextual transformer reconstruction, and contribute 0% to strict whole-model recovery.

## Largest gaps

1. The top-level shell still reads native wte, residual lambdas, and lm_head. RMSNorm
   and softcap are explicit but not yet packaged into an owned model object.
2. MLPs are 52.51% of exact stored values and have only a dense identity point.
3. The table/correction grammar has no sequence primitive. When installed everywhere it
   deletes attention and has identically zero cross-position influence. It also
   catastrophically fails unseen-token all-position CE:
   fidelity recovery +0.502 becomes -0.392 nat; efficiency +0.0788 becomes -0.6999.
4. No compressed attention/MLP factorial program exists; the hybrid oracle's -2.17559
   nat interaction forbids additive extrapolation.
5. Shared routing coordinates remain gauge-dependent and lack OOD, extraction, editing,
   and stability consequences.

The arm-difference diagnostic localizes the failed table arms' output differences to
all 335 uncovered positions in the batch and none of 1,201 covered positions. The
apparent conflict is resolved: a direct poke propagates by 0.118 nat in the live model
but exactly 0 in the installed program, even for a covered-position control. The
installed grammar removes the only cross-position path.

## Pruned directions

- The entire position-wise table/correction grammar is pruned as a contextual compiler,
  regardless of rank. By induction it can only compute a function of the current token.
- Post-forward hooks are pruned as executable evidence even when they overwrite every
  main write, because native calls and the attention bus survive.
- Ordinary-SVD QK sweeps are pruned after the weighted matched control.
- Independent QK interfaces are pruned as the default: they buy only 0.4 recovery
  points at 21.43% more complete attention storage than the shared interface.
- Coordinate-level semantic stories are pruned until the routing gauge is canonical and
  stable across data/OOD roles.

## Ranked next five

1. **Enforce contextual-program admissibility.** Every compiler candidate must own a
   sequence primitive and retain measured prefix intervention effects with no native
   fallback, alongside all-position CE. This prevents a per-token null from being
   mistaken for a compressed transformer again.
2. **Own the exact top-level shell.** Clone wte, residual lambdas, and lm_head into one
   program that directly sequences RMSNorm, both banks, and softcap. This is the
   cheapest path to a genuinely standalone reference program.
3. **Universal polynomial MLP compiler inside real attention.** Replace lookup tables
   with shared Left/Right input codes, Down-output quotients, embedding-conditioned
   lexical structure, and suffix-selected exact products while the owned attention bank
   preserves prefix transport. Validate unseen token identities and all-position CE.
4. **Compressed whole-program factorial.** Cross shared-QK attention with each admitted
   MLP candidate inside the owned shell and measure interaction, CE, and context
   recovery directly.
5. **Gauge and consequence certification.** Canonicalize shared routing coordinates and,
   at matched CE, test extraction, selective removal, collateral damage, OOD transport,
   and data-doubling stability.

## Highest-priority action executed

The simultaneous 36-component composition gate was preregistered, implemented, tested
34/34, committed before invocation, and run after the localisation GPU owner exited.
It passed in 9.5 seconds with bitwise writes/buses/logits/CE, dual transaction closure,
mutually disjoint storage, and zero literal native component calls. The artifact lists
the five remaining exact facade interfaces rather than claiming complete model ownership.

After the late diagnostic landed, I also wrote the contextual compiler no-go theorem
and its operational admission gate. It proves that the old position-wise grammar has
zero cross-position Jacobian and requires paired-prefix and internal-poke context
recovery, all-position CE, total support, and zero fallback for every future candidate.
