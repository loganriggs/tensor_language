# Head 5.7 — the constant/bias head (THE most-studied single part; do not re-derive)

**One line:** the single costliest attention head (zero-ablate +0.92 nats, 8× the next head) is a
learned CONSTANT: it reads position 0 (99.8% of queries), where mlp4 manufactures a fixed
high-norm vector, and adds ~that same vector at every position. It is ~98.5% understood as a bias.

## Established facts (with §refs and numbers)
- **Cost:** zero-ablation +0.916 (§429 map; reproduced 0.912 §1083, 0.879 §1087 at different N).
  8× the next head; more than most whole-layer ablations.
- **Mechanism (§432 + sink_bias_test/sink_source/sink_origin, ~ledger line 10550):** reads
  position 0 for **99.8%** of queries (neighbor 5.6: 5.3%); position-0 value norm 730 vs 197
  elsewhere. The parked vector is written by **mlp4 at position 0** (output norm 155k there vs
  14.8k ordinary; direction cos 0.998 across documents, even with different first tokens).
  Head output projection reproduces its average write at cos 0.999. Chain: mlp chain → fixed
  vector at pos 0 → head 5.7 fetches → adds everywhere.
- **It is a constant, functionally (§1089):** output 91% fixed by norm (mean 740, dev 72).
  Replace output with its global mean vector: cost **0.013** (batch-mean 0.013, donor-document
  mean ~free, prior arc found −0.005). No positional ramp (norm flat, final/early 1.03).
- **The constant = the stream's baseline (prior arc):** cos 0.99 with the mean residual direction
  through L6-11; accounts for 62-72% of residual magnitude at L6-8. Top |coords| sit on the
  massive/gain dims (7/8 overlap with top-8 massive dims, §1089) BUT the payload is BROAD:
  truncating the replacement constant to top-4/8/16 coords is WORSE THAN ZEROING (+1.36/1.41/1.65
  vs 0.91; top-64 works, 0.139; §1091). Rotating it costs 6.3 nats; halving/doubling ~free
  (prior arc). Removing only its U_c-projection (1.64) or the complement (2.38) both exceed full
  zeroing — partial removals are off-manifold (§1087).
- **Signature:** zeroing is content-tilted rare/freq 2.16 (§1087) — a property of gain
  disruption, NOT evidence it carries topic. Donor interchange ~free (0.021) ⇒ carries ~no
  document-specific information (§1087). "Content gatherer" framings (§1006-1007, §1083) are
  RETIRED (§1008 correction + §1089/§1092).
- **Uniqueness (§1091 + prior sweep):** only ~one other head is fully constant-absorbed; median
  constant-share among the 12 heads costing >0.02 is 39%. L5H7's bias = **86% of the whole
  stack's bias value**.

## Benchmark status
~**98.5% understood**: stand-in = one fixed 128-dim (pre-c_proj slice) vector. Files:
`head_const_map_results.json`, `l5h7_generic_results.json`, `l5h7_probe_results.json`.

## Gotchas
- Never interpret partial removals/truncations of this head as "shares" — super-additive
  off-manifold effects dominate (§1087/§1091).
- Its rare/freq tilt does not mean topic content; it means gain/baseline.
- Window-restriction experiments on L5 must allow position 0 through or they blow up (+1.11
  → +0.08 when pos 0 allowed; prior arc).

## Open
- Nothing pressing. The head is solved. If touching mlp4, see `mlp-transition-L3-5.md` (its
  position-0 constant-manufacturing job is separate from its ordinary-position context role).
