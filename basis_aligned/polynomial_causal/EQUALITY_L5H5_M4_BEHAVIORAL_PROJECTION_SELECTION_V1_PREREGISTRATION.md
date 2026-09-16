# Equality L5H5 M4 behavioral projection selection V1

The 13-projection K2 correction was selected by score L2 and failed behavioral
replay.  This test changes both the numerical construction and the selection
metric prospectively.

For the baseline, remainder, and joint corners, compute the twelve Q/K ports in
the model's native BF16 order.  Derive the child ports from those three corners
with the exact RMS-scale identity.  A correction map replaces one derived child
port with its native direct projection.  Evaluate every proper subset of the
four maps (`Q1,K1,Q2,K2`) on the 192 natural documents.  Select the
lowest-cardinality subset passing the behavioral gates below, breaking ties by
worst-cell removal error and then fixed lexical order.  If none passes, retain
the best proper subset only as a frozen diagnostic.  Freeze the selection before
the 192 code documents.

The full four-map replacement is an implementation-positive control on both
panels.  It must reproduce the already validated exact interaction graph; if it
does not, the experiment is invalid rather than a compression null.

Gates:

1. The inherited exact graph remains lawful, and the full-replacement control
   has composed replay relative L2 at most `2e-5`, removal relative L2 at most
   `2e-5`, and removal cosine at least `.99999` in every copy subtype and half
   on both panels.
2. On natural data, the selected proper subset has composed replay relative L2
   at most `.05`, removal relative L2 at most `.20`, and removal cosine at least
   `.90` in every copy subtype and half.
3. Frozen on code, the same replay and removal gates hold in every subtype and
   half.
4. Code interaction removal remains at least `.20` of the joint-vs-baseline
   effect in every cell/half, incremental noncopy change is at most `.01` nat,
   and composed recovery is in `[.80,1.05]` and within `.02` of the exact graph.
5. The selected executor uses at most 15 Q/K projections and zero learned
   parameters.

Failure of gate 1 is invalid.  Failure of gate 2 after gate 1 is a valid natural
behavioral compression null.  Passing gate 2 but failing code gate 3 is a valid
OOD behavioral compression null.  Aggregate recovery or score fidelity alone
cannot promote a candidate.

Price: one checkpoint load; 192 frozen natural and 192 frozen code documents;
15 proper correction subsets plus one full positive control; no fits,
gradients, parameter updates, or new text.
