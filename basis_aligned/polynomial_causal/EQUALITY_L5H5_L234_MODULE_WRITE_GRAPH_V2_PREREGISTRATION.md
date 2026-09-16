# Equality L5H5 L2–L4 module-write graph V2 arithmetic correction

V1 is an invalid instrument receipt. It flattened the six FP32 write tensors as
`A2+M2+A3+M3+A4+M4`; the parent graph constructs
`(A2+M2)+(A3+M3)+(A4+M4)`. The changed addition order produced maximum
all-six parent-score error `4.195e-4`, violating the frozen `2e-6` exactness
gate. Its apparent five-write support and behavioral results are not adopted.

V2 changes only merge arithmetic: combine attention and MLP inside each layer,
then add layers in `L2,L3,L4` order. For a partial subset, a layer contributes
its selected singleton or its parenthesized selected pair; empty layers are
skipped. Rows, natural-only selection rule, thresholds, code arms, metrics,
price, and all scientific predictions remain exactly those of V1:

- all-six parent replay at most `2e-6`;
- natural qualification at `.10` relative error / `.995` cosine, selected by
  fewest writes then error then mask;
- a sparse pass requires at most four writes;
- frozen code parent-score error at most `.15`, cosine at least `.98`, and half
  errors at most `.20`;
- recovery in `[.80,1.05]`, within `.08` of the correction-free parent, every
  copy cell/half above `.55`, noncopy damage at most `.01` nat;
- complete Möbius closure at most `2e-6`, zero learned parameters, and inherited
  negative wrong-donor specificity.

If exactness now passes but more than four writes are required, that is a valid
module-level sparsity null rather than grounds for another arithmetic change.

Price remains one checkpoint load, 96 prefix executions, 192 complete code
forwards, no fits, gradients, parameter updates, or new text.
