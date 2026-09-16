# Equality L5H5 M4 explicit interaction graph V1

This package represents the nonadditivity of the rank-128 child and rank-128
remainder as a first-class Möbius node:

`I = S(child+remainder) - S(child) - S(remainder) + S(baseline)`.

The four-node graph closes with zero measured score and behavioral replay error
on 192 natural and 192 code-OOD documents.  Removing only `I` changes copy-token
NLL by `.301` of the joint-vs-baseline effect overall and `.270–.321` across
every subtype and half, while changing noncopy mean NLL by only `-.000345` nat.
A one-query rolled interaction preserves full score norm within `1.17e-7` but
has only `.522` behavioral-effect cosine with true interaction removal.

The graph has zero learned parameters and depends on the exported rank-256 M4
mode boundary.  Its current implementation requires baseline, child,
remainder, and joint score ports—four score evaluations.  It is an extracted,
causally validated reusable graph node, but not yet the cheapest algebraic
implementation of the interaction.
