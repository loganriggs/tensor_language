# Scope-complete hybrid learned-constant accounting

The previous 713,578,445-bit subtotal was deliberately partial: it charged the
current MLP0--3 candidates, the 139-head decoded Q/K program, and the exact scalar
schedule, while assigning `null` rather than zero to everything else. It could not
be compared with the full checkpoint payload because their scopes differed.

`hybrid_payload_accounting.json` closes that scope mismatch under one narrow
convention: keep the existing candidate streams, and retain every other learned
checkpoint constant literally. The partition covers all 545,902,902 checkpoint
elements exactly once:

| Charge | Covered checkpoint elements | Bits |
|---|---:|---:|
| candidate MLP0--3 codecs | 63,705,600 | 422,718,360 |
| retained MLP4--17 tensors | 222,969,600 | 7,134,769,152 |
| decoded Q/K program, 139 heads | 81,985,536 | 290,859,424 |
| retained Q/K slices, 23 heads | 13,565,952 | 434,110,464 |
| retained V/O tensors | 47,775,744 | 1,528,823,808 |
| exact structural scalar codec | 54 | 661 |
| retained token embedding | 57,950,208 | 927,203,328 |
| retained unembedding | 57,950,208 | 1,854,406,656 |

The hybrid learned-constant payload is **12,592,891,853 bits**, versus
**16,541,356,896 bits** for the literal checkpoint tensor payload. That is a ratio
of `.761297` and a reduction of 3,948,465,043 bits under the shared convention.

## What the number means

This is the first scope-complete accounting of learned constants for the current
partial program. It resolves a bookkeeping ambiguity: missing modules are expensive
literal retained tensors, not free zeros. It also pays shared tensors once. In
particular, block-0 `c_v` is retained inside the single V/O charge even though its
value is used as the cross-depth `v0` bus.

It is **not** a whole-program MDL result. Both sides condition on the architecture,
loader, tensor schema, and execution machinery. The hybrid side additionally omits
decoder and assembly-graph code. Retained tensors use literal storage rather than a
quotient codec, and the candidate programs are not proved minimal. The assembled
hybrid has not been jointly verified across all five operational lanes. Therefore:

- do not call the 23.9% payload reduction a model compression ratio;
- do not infer joint fidelity from separately linked candidate evidence;
- do not compare this total with an unconditional Kolmogorov or quotient minimum;
- do use it as a complete conditional learned-constant budget and a prioritized
  inventory of where the remaining 12.59 Gbits reside.

The next accounting improvement is not another local codec. It is to charge the
decoder/assembly graph, then replace literal remainder charges with independently
verified canonical programs without changing the element-ownership partition.
