# Successor pointer × prefix-coherence V2 instrument correction

V1 is invalid because its self substitution differs from native logits by
`2.2411346435546875e-5`, above the frozen `1e-5` bar. Pointer deltas are live
and every native/self capability cell passes, but no interaction verdict is
adopted. The V1 interaction tables were not used to choose this correction.

The cause is an avoidable numerical mismatch. Native layer-0 values are formed
by `c_v` over a `30 × 10` token batch. V1 recomputed the same final-token value
on a `30 × 1` batch; the different GEMM shape changes floating-point rounding
before the self substitution. V2 computes the self value with the exact
layer-0 operation order on the native `30 × 10` token matrix. For the 60 donor
arms it copies the directed prompt matrix, changes only the final token for this
static value calculation, and computes `c_v` on the corresponding `60 × 10`
batch. The intervention prompt itself stays unchanged. Because layer-0 values
at a position are token-local, this supplies the same exact pointer object with
matched batch geometry.

Rows, prompts, conditions, directions, pointer site, scientific metrics, bars,
and terminal grammar are unchanged from V1. V2 adds two explicitly priced
static-value projection passes over 900 token positions; it still executes
three full-model forwards and 120 sequence evaluations with no fits, backwards,
updates, gain/rank search, or quantization. The V1 result and runner are
hash-bound. If self replay still exceeds `1e-5`, preserve another invalid result
and do not change the bar.
