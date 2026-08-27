# Canonical pricing of the mlp2 affine rank family

The recovered family is one centered affine map,

```text
y = ym + (x - xm) U_r diag(S_r) Vh_r,
x = concat(attn2.c_proj, mlp1),
```

with every candidate obtained from a prefix of one frozen SVD. The artifact is
identified by SHA256
`3750288b4656bc5fd147a8815ca7698ef91fcfd5b191dcdfccb873247f277f6d`.

`affine_codec.py` folds the centering into a single bias, fixes the SVD sign gauge,
quantizes at step `2^-16`, and serializes an exact, length-delimited compressed byte
stream. It rejects near-degenerate singular strata because sign fixing alone does
not quotient rotations within a repeated-singular-value subspace.

| rank | canonical Mbit | legacy parameter Mbit | held-out fidelity | marginal fidelity / Mbit |
|---:|---:|---:|---:|---:|
| 8 | 0.527 | 0.44 | 0.2723 | 0.5167 |
| 16 | 1.003 | 0.88 | 0.3618 | 0.1881 |
| 32 | 1.953 | 1.77 | 0.5139 | 0.1601 |
| 64 | 3.854 | 3.54 | 0.6865 | 0.0908 |
| 128 | 7.653 | 7.08 | 0.8160 | 0.0341 |
| 256 | 15.245 | 14.16 | 0.8893 | 0.0097 |
| 512 | 30.459 | 28.31 | 0.9134 | 0.0016 |
| 1152 | 68.571 | 63.70 | 0.9200 | 0.0002 |

All eight points remain nondominated under held-out fidelity. The operational knee
is broad around ranks 64–128: beyond rank 128 the marginal fidelity per encoded
Mbit falls by more than 3× at each of the next two doublings. Exact serialization
prices are 7.6–19.8% higher than the old scalar-count estimate, with the largest
correction at low rank where graph/bias overhead matters most.

The exact price and held-out lanes are now verified by executing the decoded byte
programs. Quantization changes CE by at most `5.95e-5` across all eight ranks, while
the decoded weights differ from their unquantized rank maps by `2.59e-4` relative
Frobenius norm. The fixed attention-background composite, three-member OOD, and
lexical22 causal-handle lanes are also measured. Extraction retention rises from
`0.2725` to `0.9201`; aligned output-span removal damage rises from `0.0093` to
`0.7283` CE. Equal-rank random removal is much smaller below full rank, but signed
per-target ratios are unstable around near-zero controls and must not be read as a
smooth simplicity score. Disjoint-row no-refit replication preserves monotone
extraction and raw removal effects, but rank 8 fails the preregistered signed-ratio
gate by flipping `-0.201` to `+2.284`. Whole-frontier certification therefore stays
open, with final checkpoint/commit provenance also unfilled.

At fixed rank 128, a separate 2^-6 through 2^-20 precision ladder shows that the
2^-16 production choice is conservative. Fidelity reaches `0.815708` by 2^-8 at
3.024 Mbit and stays within `3.7e-4` through 2^-20; from 2^-14 onward it varies by
only `1.7e-5`. Thus rank-128 can be priced at roughly 3.0 Mbit for nearly all of its
behavior, or 6.33 Mbit at the start of the measured noise-scale plateau, rather than
assuming that the 7.65 Mbit 2^-16 stream is operationally necessary.
