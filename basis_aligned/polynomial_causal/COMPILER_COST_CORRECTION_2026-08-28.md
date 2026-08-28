# Cost correction: current compiler points are probes, not compressed executables

Date: 2026-08-28 07:44 UTC

Status: CPU re-accounting of committed S1751/S1752 receipts. No row, model, or
evaluation role was loaded. Fidelity values are unchanged.

## Correction

The recent table-plus-low-rank scripts reported only trainable factor reals:

$$
P_{\rm factor}=36r(b\cdot1152+1152),
$$

where $b$ is the number of input feature blocks. This is a valid *marginal rank
price* because the table is fixed across ranks. It is not the description length of
the program. The hook also contains one 1152-vector for every covered token type at
every replaced site. With 5,419 covered token IDs,

$$
P_{\rm table}=36\cdot5419\cdot1152=224{,}736{,}768\text{ reals}.
$$

At rank 8, $P_{\rm factor}=663{,}552$, so the conditional covered-support program has
$225{,}400{,}320$ values before precision, support indices, code, or metadata. The
factor count is only 0.294% of this conditional value count.

## Corrected S1751 efficiency

| rank | held-out nats | factors | conditional table+factors | factor-only nat/M | conditional nat/M |
|---:|---:|---:|---:|---:|---:|
| 8 | 0.60060 | 0.664M | 225.400M | 0.9051 | 0.002665 |
| 32 | 0.63897 | 2.654M | 227.391M | 0.2407 | 0.002810 |
| 128 | 0.63865 | 10.617M | 235.354M | 0.0602 | 0.002714 |

Thus the factor-only efficiency overstates conditional whole-program efficiency by
339.7x at rank 8, 85.7x at rank 32, and 22.2x at rank 128. Because one common table
dominates all three denominators, rank 32—not rank 8—has the numerically best
conditional value efficiency in this spent discovery sample, although the difference
has no untouched-role uncertainty estimate and earns no frontier claim.

The same correction makes S1752's richer local features strictly worse under both
fidelity and complete conditional value count: A/B/C recover 0.38578/0.35004/-0.80171
nat while conditional storage increases from 225.400M to 225.732M to 226.064M values.

## Stronger executable caveat

The conditional count above is still not a standalone model price.

1. The current implementation materializes dense `[50257,1152]` tables at every site:
   2,084,258,304 table reals, plus factors.
2. A forward hook runs *after* the native module. Native attention/MLP computation and
   parameters therefore remain present even on covered positions.
3. On uncovered token IDs, `torch.where` explicitly returns the native output.
4. Only covered positions are scored. No total-support fallback has been admitted.

Accordingly, the executed artifact adds at least 2.085B table/factor reals while
retaining the entire native model and its FLOPs. It is an informative causal probe of
a candidate function class, not executable compression. A zero-native-call frontier
point requires a replacement called instead of the module, complete input support,
all constant tensors priced, and a native-call ledger proving zero.

## Consequence for direction

Lookup structure is still potentially compressible; this correction does not assume
each table entry is irreducible. It changes the burden of proof. A real program must
factor, cluster, share, generate, or otherwise encode those tables and demonstrate its
cost and transport. Until then:

- factor-only cost is labeled incremental optimization capacity;
- conditional table+factor bits are the minimum description proxy;
- actual hook cost is native model plus allocated replacement tensors;
- standalone executable cost is recorded as unavailable.

Exact machine-readable receipts and source hashes are in
`program_cost_audit_2026-08-28.json`. `compiler_program_cost.py` implements the four
denominators and refuses to produce a standalone efficiency for native-fallback hooks.
