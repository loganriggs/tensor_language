# Equality L8H4 score/raw-payload node V2

V1 projected each value through the L8H4 output matrix before summing across
keys.  That is algebraically equivalent in real arithmetic but not in deployed
BF16 arithmetic.  V2 follows the native order exactly:

1. mask the score by input-token equality;
2. contract raw head values across keys;
3. apply the native L8H4 output-projection slice.

The executor has no new learned parameters.  It consumes named native ports and
can be inserted or removed as one sparse graph edge.
