# High-accuracy repeated-input direction audit

Retain the exact coefficient objective, fixed independent grade normalizers,
formal-fit arm0 starting frame, and analytic pairing correction. Use fresh
seeds73240/41 for two65536-probe-per-degree gradients. Each consists of16
independent4096-probe blocks; store block gradients and running means at4096,
16384 and65536 probes. Degree1 is computed analytically with no sampled remainder.

Use only replica1's direction for QR steps0.05,0.15,0.5. Score against the start
on16384 new probes per degree, seed73242, with paired improvement standard errors.
The old two-replica sample-size forecasts are uncertain; this is a deliberate
accuracy audit, not a guarantee that65536 probes suffice. Nothing is fit to text.

- A: exact helper exhaustive checks<=1e-9, native analytic tangent FD<=1e-4,
  orthogonality<=1e-8, finite values.
- B: final independent-gradient cosine>=0.5. Earlier checkpoint cosines are
  diagnostics and cannot replace a failed final criterion.
- C: some step improves balanced squared coefficient loss by>0.001 and>3
  paired standard errors, runtime<=840seconds. Three-step exploratory screen.

Null: even this budget does not expose reliable useful descent. A pass supports
testing optimization under the exact metric; it does not establish convergence,
global recovery, native prediction, extraction, removal or reuse. Existing
native circuit failures remain failed. A miss must distinguish unresolved
gradient noise from an absent useful direction before changing the claim.

Price: zero body forwards;393216 sampled gradient probe cases across degrees2–4
and two replicas, each requiring reference and candidate contractions.65536
validation probe cases, each with one reference plus four start/candidate
evaluations. Batch64 FP64, one managed GPU,900-second hard timeout. Store about
40MB of block gradients plus frames, checkpoints and validation arrays. All
native weights, background and normalization ports remain charged.
