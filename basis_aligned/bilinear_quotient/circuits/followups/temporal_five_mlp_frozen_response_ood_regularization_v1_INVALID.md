# Frozen response OOD v1 — invalid task-label audit

The v1 receipt is preserved but is not decision-bearing. Its `control_report`
classified a row as temporal only when its `task_id` exactly equalled the older
matched-control task ID. The prospective temporal-v10 authority uses a versioned
temporal task ID, so its 16 P rows were incorrectly grouped with is/was and
normalized by the smaller is/was target scale. This inflated the reported
is/was control-margin fraction to 0.271–0.280.

V2 was preregistered before rerunning, changes only the family classifier to
the sealed `temporal_auxiliary.` prefix, and leaves the projectors, rows, noise
seeds, causal procedure, thresholds, and model-forward price unchanged.

