# Saved-output census of interaction and remaining endpoint-read routes

ORIGIN_ENDPOINT_INTERACTION_V1 fails its fixed circuit nomination: IID target
gold loss .48058 misses .5, although OOD loss .66782 and every control group
pass. Do not lower the bar, rescale, select examples or call that a passed
circuit. Diagnose its exact complement using already saved native port arms.

For A=native, B=O-key cut, C=F-value cut, D=both, define

    total endpoint read T = A-C
    origin/endpoint interaction I = A-B-C+D
    remaining endpoint read J = B-D.

Thus T=I+J. Node knockouts are A-I=B+C-D, A-J=A-B+D, and A-I-J=C.
These are exact expanded-read node interventions with native gains. They
do not remove whole source fields or establish semantic independence.
J is not yet identified as an advance-then-read algorithm; that is a separate
hypothesis needing a query-origin test. All native parameters remain priced.

Use all512 saved rows, source SHA
4e25c055bc7822349261b88db9ade9bb0cad7c9caaf6d525bc56344285f82f8b.
Replay saved I knockout and exact additive logit partition with the shared
scorer, abs<=1e-9/relative<=1e-10 using live sums. No model/GPU/new cohort.

Report gold probabilities, complete centered effects, and an exhaustive
per-population/query/hop census of native correctness and I/J/joint-knockout
correctness (all16 binary patterns, including native errors). Among native
correct cases, explicitly count both singles surviving but the joint failing,
only I knockout failing, only J knockout failing, both single knockouts
failing, and neither/joint surviving. Retain any nonmonotone patterns; no
assumption that removing a second signed contribution must cause more damage.
Provide both order strata descriptively, without promoting a selected stratum.

This census distinguishes backup, joint reliance and residual native errors;
it is not a new thresholded nomination or a rescue of the failed I node.
CPU threads2,alarm180s,tensors below256MiB. Existing metrics reused.
