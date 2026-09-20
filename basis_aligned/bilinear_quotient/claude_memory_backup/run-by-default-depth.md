---
name: run-by-default-depth
description: Logan wants the circuit lane running by default (never idle between hourly reviews) and prefers depth on one circuit over breadth
metadata:
  type: feedback
---

Logan (2026-09-18): "Depth is preferred as well as running by default." Also asked for an input-to-logit explanation of one circuit with diagram, examples, math and code (`for_logan/in_depth_circuit.md`, written).

**Why:** hours of idling between cron reviews (04:35–13:32 on 18 Sep, one bounded item per review) were wasted GPU time; the collection is broad (7 families / 16 lines) but stops at MLP ports on the contextual families.

**How to apply:** every hourly review leaves a bounded item queued (`bash ops/enqueue.sh`), and "stop the thread" means redirect to the next depth item. Prefer pushing one component through its declared port (MLP 8 on the pronoun / number lines, unit-grain) over opening new behaviours. See [[claude-circuit-lane-2026-09-17]] and [[tensor-language-ops-conventions]].

**Update 2026-09-19 ~02:00 UTC (Logan):** no STOP allowed at all — the lane must always have a queued item; if a thread completes, open the next bounded one in the same review. Logan also asked for exponential scaling of token-class sizes in any token-table measurement (4/16/64/256) so that scaling behaviour is visible, and prefers weight-folded paths from the embedding (single-token inputs) for early MLPs over unit counting.
