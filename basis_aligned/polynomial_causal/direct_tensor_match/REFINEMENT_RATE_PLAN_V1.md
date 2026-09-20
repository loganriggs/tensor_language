# Refinement learning-rate and checkpoint control — 2026-09-20 17:05 UTC

The obstruction diagnostic changes the interpretation: overlap0.5 residual initialization already has0.65% error after joint writer refit, but300 Adam steps end at44.65%. Overlap0.95 starts11.78% and ends24.08%. Both recover with1500 steps. Thus final endpoint selection and refinement rate must be tested before calling this a bad initialization or structural obstruction.

Reproduce three ratio0.5 teachers and residual stages. Compare300-step Adam refinement at0.05/0.005/0.0005, same cosine schedule, unit raw input factors, exact variable-projection coefficient loss. Record initial, final and best-training-objective checkpoints, including the initial checkpoint. Select strictly by the training objective; no heldout or teacher component information. Preserve historical receipts unchanged. This is a hyperparameter control, not evidence of universally successful residual pursuit.
