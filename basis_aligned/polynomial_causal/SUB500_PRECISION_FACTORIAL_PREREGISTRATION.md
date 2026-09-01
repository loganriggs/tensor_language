# Rung 438 preregistration — one-cell precision factorial for the sub-500M tail

Date: 2026-09-01T21:49Z

Status: **withdrawn before execution at 2026-09-01T21:52Z**. The user correctly pointed out that numerical
quantization is not the structural/interpretable compression sought by this program. No GPU job was enqueued and
`E01` was never produced. This frozen proposal is retained only as a record of the unexecuted diagnostic; it carries
no evidence or research priority.

Claim level: causal numerical attribution only. This rung cannot rescue, adopt, or re-gate the failed rungs414/436.
No precision sweep follows.

## Question

Rung436 falsified the prediction that the `.11764` worst-position deviation in rung414 was caused mainly by storing
the 14,984-value quadratic MLP16 program in BF16. Three cells of a two-factor storage experiment already exist. Run
only the missing fourth cell to decompose the deviation exactly.

The factors are:

- `U`: upstream storage treatment—source FP32 tensors rounded to BF16, QK64 factors stored FP16, and generated
  MLP0/4 programs stored BF16;
- `F`: the factored MLP16 program stored BF16 instead of FP32.

The four per-position cross-entropy vectors are:

- `E00`: neither treatment, rung392,
  SHA256 `2032dfcce74dc9f56b2ede0490f608c222e151e905cda9509d94facbb816bd79`;
- `E10`: upstream treatment only, rung436,
  SHA256 `0c2dfba33c8c732c241b5e939a5303d17dea48020017e13d930dc0d6f5782ba7`;
- `E11`: both treatments, rung414,
  SHA256 `ebf9df89c5e4b34dedb3d059dc82912ed024ddcc122f92b5c047186785f55969`;
- `E01`: MLP16 BF16 only, the sole new arm.

For every census position define

`U_main = E10-E00`,

`F_main = E01-E00`,

`interaction = E11-E10-E01+E00`,

`total = E11-E00 = U_main+F_main+interaction`.

Report mean/p95/max absolute value, Euclidean norm, cosine with `total`, and squared-norm ratio to `total` for every
term. Non-orthogonal squared-norm ratios are diagnostic and need not sum to one.

## Missing-cell physical artifact

Execute the exact rung392 program and roles, changing only its four MLP16 factor tensors from FP32 to BF16 before
the hook; runtime explicitly dequantizes them to FP32. Source tensors retain the rung392 source treatment, QK64
factors remain FP32, and generated MLP0/4 programs remain FP32. Exact artifact price is 495,847,230 scalar values
and 1,867,419,260 bytes, 29,968 bytes below rung392. This is a diagnostic package, not a candidate frontier point.

## Frozen predictions

### A — instrument and isolation

All three existing hashes match; `E01` has the same shape; the source/QK/MLP precision and fit identities match
rung392; the MLP16 artifact has exactly four tensors,14,984 BF16 values, no dense form, live hook, and exact bill.

### B — local coefficient rounding is small

`F_main` maximum absolute magnitude is at most `.050` and its squared-norm ratio to `total` is at most `.10`.

### C — coefficient/upstream interaction is small

The interaction maximum absolute magnitude is at most `.050` and its squared-norm ratio to `total` is at most `.10`.

### D — upstream treatment explains the failed tail

`U_main` has cosine at least `.95` with `total`, and the residual
`F_main+interaction = E11-E10` has norm at most `.25` times the `total` norm.

## Strong null and routing

The strong null fires on instrument failure; if either `F_main` or the interaction has maximum at least `.100`; or
if `||E11-E10||/||total|| >= .50`. A/B/C/D with null false identifies upstream precision perturbations propagated
through the nonlinear composite as the dominant source. B failure identifies local quadratic coefficient rounding;
C failure identifies a genuine precision interaction. In every case rungs414/436 remain failed and rung437 remains
unexecuted. The result changes future composition tests, not the current frontier.
