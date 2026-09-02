# Rung 477b preregistration — split-aware response-tensor repair

Registered after opening rung477 and before recomputing any product response. Rung477 remains A=false and its
2/4/4-stable-term, zero-graph outcome remains a diagnostic strong null. This repair exists because the saved tensor is
not safe input for the mixed-direction successor, not to rescue the native-coordinate graph.

## Exact defect

The production facade requires batches of four. The frozen document halves end at row250, which is not divisible by
four. Rung477 assigned the complete batch starting at row248 to half0, so member/control responses from rows250–251
were accumulated in the wrong half. This produced a minimum half-member count38 even though the CPU support audit
correctly predicted39. The total across halves is unaffected; only its half allocation is wrong.

## Only permitted repair

Keep every row, source, MLP, circuit-family split, member/control mask, gradient definition, response-graph algorithm,
threshold, permutation seed, and output restriction from rung477. For a batch that crosses row250, intersect each
loss mask separately with rows below and at/above250, execute the corresponding two gradients, and accumulate each in
its correct half. All other batches are unchanged. Recompute the exact nonempty-backward formula from these split-aware
masks. Do not open odd-root validation families or SEALED attention0.

## Frozen repair checks

### A — corrected instrument

- all rung477 source/result/bundle and preregistration hashes match;
- replay relative squared error is at most`1e-12`, factor error at most`1e-10`, and term-versus-write contraction
  relative squared error at most`1e-8`;
- the corrected member/control support minima are at least39/439;
- observed forwards and split-aware backwards match their CPU formulas;
- all entries are finite and validation-family/SEALED outcomes remain closed.

### B — native-term instability is robust to the accounting repair

No MLP has more than20 source/half-stable native product terms under the unchanged rung477 rule (the opened counts
were2/4/4, versus the original230 bar).

### C — no native-coordinate cross-MLP graph appears

Every unchanged pair graph still has zero qualifying joins. This is a robustness check, not a fresh independent test.

### D — only half allocation changed

Summing corrected response sums and counts across the two halves reproduces rung477's corresponding totals with exact
count equality and response relative squared error at most`1e-10`.

### E — reserved outcomes remain sealed

The output contains no validation-family product responses, raw tokens, logits, or hidden states, and SEALED
attention0 remains unopened.

## Routing and price

If A+D hold, use the corrected discovery tensor—not rung477's half-contaminated bundle—as the fixed input to the
sparse mixed-product response-direction experiment. The native-coordinate strong null remains in force regardless of
B/C because its original receipt is not rescored. Zero deployed parameters saved or added; execute only through the
managed runner.
