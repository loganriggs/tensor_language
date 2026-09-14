# Correction-only replay of the native framing-role split

V1 mislabeled its fifth arm: the runner called `graph.write(states)`, which
removes the complete S+R+O head rather than O alone. Native and framing anchors
replayed bitwise, the exact role partition passed, and the description versus
instruction arms are unaffected, but frozen `pred_a` requires the all-O anchor
and therefore correctly failed. Preserve V1 and its result.

V2 changes only arm 4 to write `states[2]`, the complete O source sum, through
the frozen output map. Rows, masks, six readouts, five-arm order, 240-forward
price, thresholds, and predictions A-D are byte-for-byte the V1 scientific
design. No V1 score is used to change a prediction or row.

