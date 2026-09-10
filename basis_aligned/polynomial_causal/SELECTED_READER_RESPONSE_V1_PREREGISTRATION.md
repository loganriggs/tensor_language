# Exact selected-reader response state and a full-output closure witness

10 September 2026, three-hour mathematical review. CPU only, zero native model
forwards. Use trained MLP17, existing e and existing G runs/run reader g. These
are two output readers, not an independently extracted grammatical model.

For m(u)=D[(Lu)*(Ru)]+b and unit input-edit direction e, define for reader v
K_v=L^T[(D^T v)*(Re)]+R^T[(D^T v)*(Le)] and
a_v=(D^T v)^T[(Le)*(Re)]. Then its exact response is

    response(q,delta)=delta*q+delta^2*a; q=K*u.
    next_state(q,delta)=q+2*delta*a.

Sequential shifts along the same e compose exactly. This is a local conditional
response program with two initial state scalars; initializing them still reads
the full normalized MLP input. It does not compute absolute MLP outputs, context
producers, intervening normalization, attention, or arbitrary-direction edits.

Compile K and a from weights only. Construct sixteen FP64 synthetic context
reflection pairs u+=p+z, u-=p-z, with p in span(e,K_e,K_g) and z perpendicular
to that span. Fix ||p||=||z||=sqrt(1152/4), hence ||u||^2=1152/2 inside the
native RMS map's open ball. These are admissible post-RMS values, but not
claimed to be reachable from natural text. Gaussian seed9111440, no filtering.
Use delta=.25 and composition commands .25 then -.125. Save all outcomes.

Predictions, frozen before execution:

* A FP64 instrumentation: direct-versus-folded relative response error<=1e-10;
  equal e/K features and squared norms have scale-normalized errors<=1e-10.
  Tiny generic algebra control also passes. Check checkpoint and input hashes.
* B selected-reader extraction: A; two-state program predicts both reader
  responses on all32 contexts with relative L2 error<=1e-10. Sequential command
  composition and state updates agree with direct total shift at same tolerance.
* C closure witness: A/B; at least15/16 equal-state pairs differ in their full
  MLP output response by normalized difference>=.10. Ratio is
  ||r+ - r-|| / sqrt(2*(||r+||^2+||r-||^2)). It equals the minimum relative L2
  error of a common pair prediction because equal states force one prediction.
  B alone must not be described as full-output or full-model closure.

No rank, sample, gain or selected-layer sweep. MLP17 is the existing final-layer
dossier subject; no choice from outcome. Exact linear-state theorem is proved
in the mathematical review, not inferred from numerical rank. Store K(2x1152)
and a(2):2306 runtime coefficients. Initial full u, old e and reader authorities
remain explicitly charged when needed; only per-command local response can run
without the native MLP after state initialization. All model-level dependencies
remain, with no whole-model weight saving. No behavioral removal/OOD promotion.
