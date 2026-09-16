# MLP9 equality projected-write execution note

Two attempts terminated before producing a result receipt.  The first used
`r500._task_masks`, which intentionally exposes only three aggregate cells,
while the preregistered runner requested the detailed near/far and predecessor
cells.  The second switched to the authoritative detailed mask builder but
still assumed it supplied `all_noncopy`; that aggregate is defined by rung 500,
not by the detailed helper.

The final runner uses `rung498_copy_task_portability_diagnosis.build_masks` for
the five named copy cells and explicitly constructs `all_noncopy` as valid
positions 64--255 minus `copy_positive`.  A CPU preflight verified supports of
890 copy-positive, 222 near, 668 far, 394 single-predecessor, 496
multiple-predecessor, and 11,398 all-noncopy tokens before the corrected GPU
run.  Neither failed attempt wrote the result namespace or exposed a scientific
outcome.  Their traces remain in the managed run log.

Corrected runner SHA-256:
`e47c8aa60e2f7e44eb8fc7c7b4b3f8b8f49adffe6142265d8d485f2e5b4c9279`.
