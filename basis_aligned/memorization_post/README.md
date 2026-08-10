# Memorization in Bilinear Layers — post experiments

Spec: ../memorization_post_handoff.md (Logan's handoff, self-contained; follow it exactly).
Logan has prior results on a local machine that these will be checked against — everything
seeded, conventions exactly as the handoff states, deviations documented in results.md.

Layout: part1.py part2.py part3.py part4.py (one re-runnable script each), figures/ (PNG+SVG+npy),
results.md (claim checks with numbers), surprises.md (anything contradicting the handoff's
pre-registered predictions — the post quotes these directly), predictions/ (timestamped
prediction files committed BEFORE the corresponding measurement, per the handoff's C1 rule).

Coordination: multiple agents may work here concurrently (part1 vs part2). Before any commit:
git pull --rebase --autostash. Only touch your own part's files plus your own sections of
results.md / surprises.md (append, marked with the part number).
