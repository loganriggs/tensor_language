# Rung 570 preregistration: digit/word numeric-sequence native gate

For digit answers, the numeric margin compares the correct leading-space digit against all other eligible
leading-space digits. For number-word answers it compares against all other leading-space number words. A standard
endpoint cell passes at 75% accuracy with a positive 2,000-group-bootstrap lower mean margin.

1. Both endpoints of digit, number-word, and cross-format `+1` state shifts must pass separately.
2. Both endpoints of the digit/word surface and copy families must pass separately.
3. The coherent base of the middle-value family must pass. Changing only the middle value while holding the first,
   final, and registered answer fixed must lower the answer margin in at least 65% of groups, with positive bootstrap
   lower mean reduction. Failure distinguishes a last-value successor from the claimed relation-reading circuit.
4. The `+2` conflict reports arithmetic-`+2` versus last-value-`+1` margins but has no capability threshold; it is
   reserved as a later intervention-selectivity control and cannot rescue another cell.

FIT opens first; this circuit's SELECT opens only if all of its FIT bars pass. The numbered-list decision is
independent. FINAL_TEST/OOD remain closed. Maximum price for this circuit is 457 unique sequences in 15 forwards, zero
backwards and no fitted parameters.
