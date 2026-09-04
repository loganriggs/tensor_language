# Task 17 FIT publication-repair provenance correction

**Correction written:** 2026-09-04 05:16 UTC. **Status:** append-only provenance correction; model execution remains
blocked. This document changes no scientific or instrument content.

## Statement being corrected

The immutable publication-repair amendment with SHA-256
`0c4a20b751cc05c5373b3a1d0eab95164ffc70e5dbe685cc12a9dbb341ff8301` begins with the statement
“Frozen prospectively: 2026-09-04 05:22 UTC.” That time is false: it is later than both the Git freeze event and the
wall clock at which the discrepancy was found. It was a future-time transcription error, not evidence that the file
was frozen at 05:22 UTC. The committed amendment is retained byte-for-byte so that its history is auditable.

## Authoritative freeze event

The authoritative immutable freeze event for that amendment and its publication repair is Git commit
`538cef96451b3e8f07758f20cca2be1b7bfdf561` (`Harden task17 publication against namespace races`). The commit object's
author and committer timestamps are both exactly `2026-09-04T05:13:56+00:00`. The commit contains the amendment bytes
whose SHA-256 is `0c4a20b751cc05c5373b3a1d0eab95164ffc70e5dbe685cc12a9dbb341ff8301`.

For provenance, the Git commit object and its timestamp supersede only the amendment's incorrect embedded 05:22 UTC
timestamp. The amendment's publication protocol, producer binding, tests, restrictions, and all other content remain
unchanged.

## No retrospective science or authorization

At the authoritative freeze event, the managed adapter had `EXECUTION_AUTHORIZED=False`. The repair used only source
editing, unit tests, and a model-free dry run. It did not access a model checkpoint or GPU, run model forwards, read or
publish task-17 outcomes, enqueue work, or create final result, evidence, receipt, or localization namespaces. Thus the
repair remained prospective relative to any task-17 science despite the incorrect written time.

This correction grants no model, checkpoint, GPU, queue, enqueue, outcome, localization, SELECT, TEST, OOD, or
execution authority. The real adapter branch remains blocked pending the already required fresh independent review
and later authorization amendment.
