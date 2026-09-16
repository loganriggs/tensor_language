# Equality L5H5 M4 port Möbius order V1

All four native child Q/K ports are jointly necessary when selection is
restricted to raw-port subsets.  This experiment changes representation: treat
native replacement of `Q1,K1,Q2,K2` as a four-variable Boolean function and
compute the exact Möbius expansion of the child score over its 16 corners.

For each panel, certify that the sum of all 15 nonconstant Möbius terms
reconstructs the full native child score.  Report every term's norm/cosine to
the total correction and the cumulative score error after retaining terms of
order at most 0, 1, 2, 3, or 4.

On 192 natural documents, compare interaction-removal vectors for cumulative
orders 0 through 3.  Select the lowest order whose removal vector has relative
L2 at most `.20` and cosine at least `.90` in every copy subtype and half;
break ties by worst-cell error.  If no proper order passes, retain the best only
as a diagnostic.  Freeze the order for 192 code documents.  Order 4 is an exact
implementation-positive control on both panels.

Gates:

1. Inherited authority is lawful; exact Möbius score reconstruction error is at
   most `2e-6`; and order-4 behavioral removal has relative error at most
   `2e-5` and cosine at least `.99999` in every natural/code cell.
2. A proper cumulative order passes the natural removal gates.
3. The frozen order passes the same gates on code.
4. Code removal magnitude is at least `.20` of the joint-vs-baseline effect in
   every cell/half and incremental noncopy change is at most `.01` nat.
5. The graph has zero learned parameters and fewer than 15 nonconstant terms.

Failure of gate 1 is invalid.  Failure of gate 2 after a valid control is a
valid low-order-graph null; failure only on gate 3 is a valid OOD null.  Exact
composed replay is not a gate because algebraically closing to the native joint
corner makes it automatic and therefore non-diagnostic.

Price: one checkpoint load; 192 frozen natural and 192 frozen code documents;
four proper cumulative orders and one exact control; no fits, gradients,
parameter updates, or new text.
