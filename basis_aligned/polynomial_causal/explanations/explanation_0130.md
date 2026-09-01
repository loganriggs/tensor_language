# Plain-English update — 2026-09-01 01:30Z

(Damage = extra prediction error above the real model; LOWER IS BETTER.)

## The red-team turn: our measuring stick may have been crooked
All night we believed the compressed attention maps hit a hard floor: keep anything less than the full
maps and you pay ~0.055 error, no matter which parts you keep — an elegant "delicate global mechanism."
Then a failed control forced the question we had never physically asked: what does our compressed-model
MACHINERY cost when it compresses NOTHING? Answer (first of two independent checks): ~0.052. Our
replacement pipeline — the code path that rebuilds attention from factors, in different precision and op
order than the original — appears to carry a constant error of its own. The "mechanism" may be our
instrument. The identical damage fingerprints across all those configs? All of them shared the pipeline.

Nothing is retracted yet: a second, independently written pipeline is running the same test now, plus a
zero-replacement harness check. If they agree, we correct a dozen ledger entries, subtract the instrument
error from every number, and the model may turn out MORE compressible than we thought — with the next job
being to make the pipeline numerically faithful (engineering, not new math). The knockout/manipulability
results survive better: they were measured as differences, which cancels a shared instrument error.

This is what preregistration and controls are for: the flag was raised by our own failed prediction, and
the fix path was queued within minutes. Codex (now co-piloting research direction) has the board notes.
