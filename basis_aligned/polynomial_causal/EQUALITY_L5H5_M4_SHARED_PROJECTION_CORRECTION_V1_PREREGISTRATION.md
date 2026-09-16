# Equality L5H5 M4 shared-projection one-map correction V1

All four 12-projection arithmetic gauges are lawful but miss the frozen code
interaction gate by about `7e-4`; natural selection chose the child-derived
gauge.  This prospectively tests the smallest arithmetic correction rather than
returning to the 16-projection authority.

For the selected child-derived gauge, compute the residual mismatch between the
exact child residual and `joint - remainder + baseline`.  Project that mismatch
through exactly one of the four L5H5 Q/K maps and add it only to the derived
child raw port.  Candidate corrections are Q1, K1, Q2, and K2.  Select the
lowest natural authority-relative interaction error, tie-breaking in that
order, then freeze on code.  Every candidate costs 13 Q/K projections.

Gates retain the previous `.10/.95` natural/code interaction fidelity and `.15`
half-error bars; baseline/joint error `.01` and cosine `.999`; closure `2e-6`;
13 projections versus 16; zero learned parameters.  The correction residual
norm and projected correction-to-port norm are reported.  Passing authorizes
behavioral replay/removal; failure is a valid one-map correction null.

Price: one checkpoint, 192 natural and 192 code prefix documents, four natural
candidate corrections and one frozen code correction, no behavior forwards,
fits, gradients, updates, or new text.
