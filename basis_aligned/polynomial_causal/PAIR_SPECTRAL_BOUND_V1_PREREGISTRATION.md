# Exact scalar-contrast bilinear bounds

September14 10:39UTC. Matrix-output and all-contrast aggregate improvements
did not improve prompt-selected task contrasts. Freeze original/minimax.
For each of the six existing UK/US pairs, form the scalar error matrix
E_pair=E_uk-E_us, shape1152x128. Its largest singular value exactly equals
max over unit residual z and unit head a of |z^T E_pair a|. No alternating
search or population assumptions are needed for this scalar bound.

A: singular-vector witness replays singular value<=1e-10relative.
B: every minimax pair bound<=original. Report all pairs without selection,
plus descriptive fresh-cache own-effect error energy grouped by the prompt's
relevant pair. This is a diagnostic comparison, not native-data fitting.
An absolute bound is not relative native fidelity or sufficiency.
CPU120seconds/two threads; no fitting/adoption or further native forwards.
