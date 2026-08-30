# Price cliff, sublayer-resolved — result (self-reviewed)

**Run:** `price_cliff_sublayer_v1.py`, 74 s, 2026-08-30 17:49 UTC via `bqrunner`; 64 fresh rows
(384–447, untouched by earlier observability artifacts), 8 draws. Registered in the script header.
**pred_a HELD | pred_b HELD | pred_c FAILED.** Artifact: `price_cliff_sublayer_v1_results.json`.
The replicated block forward reproduces the model's CE to 1e-6 (self-check before scoring).

| site | r = 0.25 | r = 0.5 | r = 1.0 |
|---|---:|---:|---:|
| block 5 input | 0.016 | **0.075** | 1.04 |
| after attn5's residual add (before mlp5) | 0.250 | **1.722** | 3.48 |
| block 6 input | 0.193 | 1.693 | 3.94 |
| after attn6's residual add | 0.393 | 2.294 | 4.24 |
| block 7 input | 0.309 | 2.022 | 4.46 |

Single-position arm at block 6, r = 0.5: perturbing one position raises CE at **that position by
0.824 nat** and at **all later positions combined by 0.132** (ratio 0.16).

## Reading

1. **The cliff is attn5's write** (pred_a: 1.72 ≥ 5 × 0.075, a 23× jump inside one sublayer).
   mlp5 adds nothing to it (pred_b: block-6 input 1.69 ≈ 1.72); attn6 adds a further ~35 %. After
   the last gatherer-band attention has written, the stream is committed: a random error of half
   the stream's norm costs 1.7 nat instead of 0.075.
2. **The cost is local, not conducted** (pred_c failed at 0.16 against 0.5): an error at position t
   after block 5 damages the prediction *at t* and barely touches later positions. So the expense
   is not that later attention reads the error and spreads it; it is that from block 6 on the
   per-position stream *is* the answer for that position, in a form the tail cannot repair.
   Combined with the 18-site profile (price decays after block 7 as the tail becomes more linear
   in the stream) this says: the band's job is finished at attn5, and what follows is a
   per-position readout pipeline whose input must be exact.
3. **For the program:** lane 1's certified arm keeps attention real yet carries rel-MSE 1.74 at
   block 6 — the front MLP tables' error (0.51 already at block 1) is *amplified by attn5* into the
   expensive regime. Since error before attn5 is cheap (0.075 nat per half-norm at block 5) and
   error after it is not, the compressed front must be exact in whatever attn5 reads, and nothing
   else. The own-error quotient run (in flight) measures which directions of that block-6 error
   are the expensive ones.
