# MLP2 CMR v1: trajectory-complete selector findings

## What was measured

For each of 191 fresh FineWeb documents and eight independent categorical-Fisher
probes, the experiment differentiated the sampled downstream log-probability through
all layers with respect to one document-specific scale on each centered MLP2 product.
The scale was shared across every token position, so the response includes later
attention-mediated effects rather than only MLP2's immediate write.

The response tensor had shape `191 x 8 x 4608`. Each document was normalized to unit
Frobenius response energy, then a CPU float64 rank-256 ridge-leverage selector chose
512 products. The complete run used 48 forwards and 384 backwards and took 58.99
seconds. The native baseline was bit-exact. No validation, finite deletion, loss,
accuracy, or replication was opened.

## Stability and invariance

- Independent four-probe halves: score-rank Spearman `0.94909`.
- Independent four-probe top-512 supports: Jaccard `0.78397`.
- Signed dyadic reciprocal channel gauges: exact canonical and support replay.
- Arbitrary channel permutation: exact physical-support equivariance.
- General non-dyadic reciprocal gauges: maximum canonical-function relative error
  `4.73e-16`.

This is a stable fit selector on these documents, not yet a finite simplification.

## Corrected relationship to the controls

The original result JSON incorrectly reported all overlaps as zero because it put
scalar tensors, rather than their integer values, into Python sets. The support
tensors and hashes were unaffected. A source-closed CPU correction now supersedes
only that summary.

| Pair | Shared products | Jaccard |
|---|---:|---:|
| SUFFIX--LOCAL | 369 | `0.56336` |
| SUFFIX--RMS | 364 | `0.55152` |
| SUFFIX--MASS | 308 | `0.43017` |
| SUFFIX--DERANGED | 87 | `0.09285` |
| SUFFIX--HASH_RANDOM | 56 | `0.05785` |

The SUFFIX score correlates strongly but imperfectly with LOCAL: Spearman `0.85966`
and Pearson `0.82939`. It is less aligned with weight-only MASS: Spearman `0.62563`.
Thus downstream differentiation changes 143 of the 512 LOCAL choices, but does not
reveal an unrelated coordinate system.

The fraction of total SUFFIX ridge-leverage score captured by each 512-support is:

| Support | Captured SUFFIX score |
|---|---:|
| SUFFIX | `25.534%` |
| LOCAL | `23.963%` |
| RMS | `23.867%` |
| MASS | `22.399%` |
| DERANGED | `12.589%` |
| HASH_RANDOM | `11.581%` |

SUFFIX therefore improves its own fit objective over LOCAL by only about 6.6%
relative. That is real and split-stable, but modest. It does not imply the registered
requirement of at least 5% lower **finite teacher KL** than every control.

## Interpretation

The useful conclusion is a narrowed fork:

1. If SUFFIX wins finite validation, roughly one quarter of its chosen products were
   discovered specifically through downstream response and are doing consequential
   compensation that local write energy misses.
2. If SUFFIX and LOCAL tie, downstream response geometry mostly confirms the simpler
   immediate-write score, and the 143 exchanged channels were not important enough
   under joint deletion.
3. If all 512-product candidates fail the absolute CE/KL/logit gates, the problem is
   not selector noise: native product coordinates at this width do not compose
   through the suffix, so the project should move to block or response-conditioned
   factors rather than sweep nearby widths on the same role.

No strict causal or storage ledger changes until that finite test physically executes
only 512 Left rows, 512 Right rows, 512 Down columns, and the folded bias.

## Provenance

- Selector bundle SHA-256: `cb3f8d3caecab86881eba825785cabd58c1b7ac8e2aa1eb93b459168cff17ce1`
- Selector result SHA-256: `ab08dc0f0a71b5daf21228991b9e78a272aa74d226d97189ac414a546dc16f62`
- Selector receipt SHA-256: `b61c7308409ec64dc05601206bda21e1f4e24097871ba8dff0c92bc84e761e1f`
- Correction result SHA-256: `ffd5a826962f09ffec0af6c842eaf0bf64530423b827f6239556bc43db9d7ff4`
- Correction receipt SHA-256: `dd557dc6366503bea2f3f7649d6312abc8a89857bb277ea4e844d8822e4e968a`

