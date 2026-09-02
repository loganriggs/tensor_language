# Pre-outcome derivation: finite interchange after an attention1 downstream-use match

**Written:** 2026-09-02 16:57 UTC

**Status:** CPU mathematical continuation while rung495 runs. This is not a preregistration, does not inspect a
rung495 result, and cannot license a claim by itself. It fixes the meaning of the registered pass-route phrase
"physical interchange" before any candidate identity is known.

## Why a second experiment is necessary

Rung495 assigns each exact raw attention1 piece `theta_p` a local downstream-use signature

`f_p(c) = <d L_c / d w_attn1, theta_p>`,

where the derivative passes through the real RMS normalization, MLP1, residual path, and later layers. If two pieces
have proportional signatures, later circuits respond similarly to infinitesimal changes along those two directions
at the observed state.

That does not imply they can replace one another at their natural finite size. RMS normalization and every later
bilinear layer make the suffix nonlinear in the raw attention write. The pass successor must therefore edit the raw
write, recompute normalization and the whole suffix, and compare actual finite causal effects.

## Exact finite arms

Let the frozen cross-head pair selected by rung495 be `p` and `q`. On each natural example, let

- `w` be the complete native raw attention1 write;
- `theta_p` and `theta_q` be the two exact Möbius pieces; and
- `a_(p->q)` be the scalar fitted on rung495's selection half such that `f_q` is approximately
  `a_(p->q) f_p`.

The scale used to substitute `q` for `p` is therefore `1/a_(p->q)`, fixed before new outcomes. The physical arms are:

- native: `w`;
- remove p: `w - theta_p`;
- remove q: `w - theta_q`;
- replace p by q: `w - theta_p + theta_q/a_(p->q)`;
- replace q by p: `w - theta_q + a_(p->q) theta_p`;
- matched controls using the frozen position-shifted donor and the closest noncandidate cross-head piece.

Every arm enters the model at the raw attention1-write site. The model then performs its own RMS normalization and
complete suffix. The pieces are not injected after normalization and no MLP1 polarization approximation is used.

The exact factor decomposition must be recomputed in the same process. Reconstructing `w` from the absent endpoint
plus all 63 pieces must match the native raw write at the registered numerical tolerance before any scientific
clause is scored.

## Measurements

For every frozen downstream circuit `c`, measure the finite signed removal effects

`Delta_p(c) = L_c(w - theta_p) - L_c(w)` and
`Delta_q(c) = L_c(w - theta_q) - L_c(w)`.

Then measure the direct replacement errors

`R_(p<-q)(c) = L_c(w - theta_p + theta_q/a) - L_c(w)` and
`R_(q<-p)(c) = L_c(w - theta_q + a theta_p) - L_c(w)`.

A true operational equivalence should satisfy both:

1. the scaled finite removal vectors remain aligned on held-out documents and circuits; and
2. direct replacement perturbs the target circuit family substantially less than removal while not increasing
   unrelated-circuit effects.

These are distinct conditions. Matching removal effects without successful replacement can arise from common
sensitivity in a nonlinear background. Successful replacement without selective preservation can arise from a
generic low-impact direction.

## Required controls and selection discipline

A later preregistration should freeze the pair and scale from rung495. It must not reselect them on finite outcomes.
Use only documents/circuit tags that rung495 did not use for pair selection; distinguish confirmation data from a
genuinely new corpus rather than calling both OOD.

At minimum compare against:

- the same donor piece rolled to frozen other token positions;
- the closest cross-head noncandidate under raw write cosine, to test generic geometric similarity;
- the closest noncandidate under gradient signature that failed rung495's confirmation/control clauses;
- a within-head piece matched on finite removal size; and
- a norm-matched random residual direction passed through the same raw-write site.

The candidate must beat controls in both substitution directions. Report all target and unrelated circuit tags; do
not discard negative effects, low-effect branches, or a direction that fails.

## What would count as progress

- **Finite operational equivalence:** both scaled removal vectors confirm and both replacements preserve the target
  effects better than every control while unrelated circuit changes stay below a frozen preservation bar. This is an
  identified shared downstream variable and advances cross-head grouping, extraction, selective manipulation, and
  reuse. It is still not a compressed program until the shared generator and literal cost are specified.
- **One-way substitutability:** one direction works and the reverse fails. Record a refinement or containment
  relation rather than equivalence; the stronger piece may carry the weaker use plus additional computation.
- **Gradient-only match:** finite removals or substitutions fail although rung495 confirmed. Retain the local tangent
  statement and close finite equivalence at this grain. Do not tune scale, rank, or thresholds.
- **Nonselective preservation:** replacement looks good only because both pieces have little effect or all circuits
  move together. The liveness and unrelated-circuit clauses reject this as a circuit.

## Literal price to freeze before execution

Once the pair and available untouched data are known, derive the exact number of native, removal, bidirectional
replacement, and control suffix forwards. Bill recomputation of all 63 pieces and every prefix cache. If a fresh
corpus must be loaded or generated, register its identity and hash before inspection. No GPU job should run until the
instrument tripwires, effect-size floor, preservation bars, call counts, and result-dependent routing are frozen.
