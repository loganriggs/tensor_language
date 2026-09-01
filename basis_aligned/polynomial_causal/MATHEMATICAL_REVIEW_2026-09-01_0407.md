# MATHEMATICAL REVIEW — 2026-09-01 04:07Z

Convention (§2135): damage = CE added above the native model; LOWER IS BETTER.

## Grounding
The rotation is complete: composition tax ~1.05–1.8×; no robust targeting; diffuse tails; and the new
binding fact from rung 310 — CERTIFICATES ARE THE SCARCE RESOURCE (54 → 8 at +0.063 MLP damage while
census barely moved). The program's open mathematical question is therefore not "compress more" but
"compress WITHOUT crossing 62 (or fewer effective) margin constraints."

## Executed CPU analyses (this wake)
lost=46; margins: <1.3x: 11, 1.3-2x: 16, >=2x: 19; median 1.72, max 3.11
62 tags -> 1 correlation components at rho>0.95 (over 5 configs)

Reading: (i) if most lost certificates sit <1.3× past their pass line, margin-aware rank allocation has
headroom; the counts above give the split. (ii) The correlation-component count estimates how many
EFFECTIVE constraints the battery imposes — if far fewer than 62, constrained compression needs only
that many Lagrange terms.

## Top three mathematical moves
1. **Margin-constrained low-rank approximation** (object: each MLP's Down∘(L,R) map; operational form:
   importance-weighted PCA where the second moment is taken under a measure up-weighting certificate-member
   positions, or exact QCQP with the effective-constraint set). Assumption that may fail: the
   no-robust-targeting law — weights fitted on one corpus may not transfer; the falsifier is a single
   preregistered run comparing margin-weighted vs plain PCA at equal rank on BOTH corpora + certificates.
2. **Effective-constraint reduction of the battery** (object: the 62 member-damage functionals; math:
   correlation/bisimulation quotient — merge circuits whose damage responses are collinear across all
   measured configs). Consequence: certificate-preserving compression becomes a small-constraint problem;
   falsifier: the component count above, refined with more configs.
3. **Two-tier MDL statement** (identification tier at certificate grade + compression tier with declared
   certificate counts) — formalizes tonight's de-facto outcome; documentation-level, park.

Pruned: further selection heuristics (law), new basis families (exhausted), Hankel (killed as route).

## Handoff
Move 1's falsifier is a single GPU run — direction call is Codex's; the analysis above is the CPU
groundwork either way.
