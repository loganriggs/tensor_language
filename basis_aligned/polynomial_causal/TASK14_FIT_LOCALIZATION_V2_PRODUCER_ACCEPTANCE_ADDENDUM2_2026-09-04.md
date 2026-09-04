# Second addendum: task-14-v2 instrument-invalid publication semantics

**Frozen:** 2026-09-04 11:20 UTC, before producer construction and while the compiler remains vetoed.

**Controls over:**

- `TASK14_FIT_LOCALIZATION_V2_PRODUCER_ACCEPTANCE_2026-09-04.md`, commit `ecb37c0ab`, SHA-256 `1724fa6de7ece875cd633976841159302e04033ca008af6e6437ee159a935b46`; and
- `TASK14_FIT_LOCALIZATION_V2_PRODUCER_ACCEPTANCE_ADDENDUM_2026-09-04.md`, commit `ea50dcfdf`, SHA-256 `c28e6dc2a453a08027673a2420bbf2053e94a0cb02b18a6f0579f747c81a4d96`.

An independent exact-byte check found one contradiction in the first addendum: it required a task-specific schema for all nine frozen terminals, including `instrument_invalid`, while also saying every optimizer failure leaves no completed package. The frozen v2 authority assigns some fully observed finite optimizer-health or seed-health failures to `instrument_invalid`. Those must not be erased.

The executable distinction is:

1. A **fully completed, finite-evidence scientific-path `instrument_invalid`** is publishable. Examples include a finite optimizer-health or five-seed health predicate that can be evaluated only after every scheduled call for that stage completes. The task-specific validator must reconstruct the failed predicate from complete retained primitives, require exact active-chunk completion and ledger/hash closure, require every in-scope numeric value finite, and enforce the terminal's exact later-field null schema.
2. A **runtime or incomplete-instrument fault** is not a publishable task terminal. This includes failed preflight or source/checkpoint/runtime/canary identity; malformed or nonfinite input, gradient, array, scalar, or JSON; incomplete/reordered call slice; deadline/watchdog stop; out-of-memory or memory-cap breach; uncaught model/runtime exception; missing evidence; and staging/publication failure. Such a run leaves no completed result/receipt package and grants no retry authority.
3. The task-specific result schema still defines all nine frozen scientific-path terminals. It must separately define the absence of a completed package for operational faults; it may not encode those faults by trusting a producer-supplied `instrument_invalid` string.

This correction changes no scientific threshold, terminal precedence, call, authority, evidence value, retry rule, or execution permission. It is not compiler approval, implementation/model/GPU/canary/queue authority, or enqueue authorization.
