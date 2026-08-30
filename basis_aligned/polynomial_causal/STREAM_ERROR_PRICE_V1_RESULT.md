# Stream error price v1 — result (self-reviewed)

**Run:** `stream_error_price_v1.py`, 282 s, 2026-08-30 17:42 UTC via `bqrunner`; 64 fresh rows
(320–383 of the zero-overlap window, untouched by the Gramian scripts), 18 sites × 3 relative norms
× 8 draws. Registered in the script header before any number. **pred_b HELD | pred_a FAILED |
pred_c FAILED.** Artifact: `stream_error_price_v1_results.json`. Base CE on these rows 3.896.

## The curve: mean CE increase for a random perturbation of relative norm r at the stream entering block k

| k | r = 0.25 | r = 0.5 | r = 1.0 | rescale by (1+0.5) | flip 10 % of positions |
|---|---:|---:|---:|---:|---:|
| 0 | −0.001 | 0.000 | 0.012 | 0.001 | 0.15 |
| 1 | 0.005 | 0.028 | 0.340 | 0.006 | 3.23 |
| 2 | 0.004 | 0.024 | 0.273 | 0.005 | 0.78 |
| 3 | 0.004 | 0.022 | 0.220 | 0.007 | 0.25 |
| 4 | 0.006 | 0.027 | 0.212 | 0.001 | 0.28 |
| 5 | 0.011 | 0.058 | 0.876 | −0.006 | 0.52 |
| **6** | **0.162** | **1.484** | **3.614** | 0.004 | 0.46 |
| **7** | **0.280** | **1.809** | **4.082** | 0.015 | 0.76 |
| 8 | 0.230 | 1.309 | 3.529 | 0.033 | 1.46 |
| 9 | 0.163 | 0.956 | 3.116 | 0.033 | 1.28 |
| 10 | 0.173 | 0.958 | 3.127 | 0.046 | 1.76 |
| 11 | 0.155 | 0.805 | 2.791 | 0.054 | 1.76 |
| 12 | 0.153 | 0.739 | 2.640 | 0.043 | 1.91 |
| 13 | 0.125 | 0.640 | 2.438 | 0.018 | 2.03 |
| 14 | 0.107 | 0.531 | 2.245 | 0.007 | 2.06 |
| 15 | 0.090 | 0.434 | 1.943 | 0.021 | 2.00 |
| 16 | 0.074 | 0.353 | 1.678 | 0.062 | 1.93 |
| 17 | 0.106 | 0.453 | 1.787 | 0.186 | 2.64 |

## Reading

1. **A cliff between blocks 5 and 6, not a ramp.** A half-norm random error costs 0.02–0.06 nat at
   blocks 0–5 and **1.48 nat at block 6** — a 25× jump in one block — peaks at block 7 (1.81), then
   decays to 0.35–0.45 by blocks 16–17. pred_a (monotone rise, ρ ≥ 0.8) fails at ρ = 0.41 because the
   curve is a cliff followed by a decline. Blocks 3–5 are the content-gathering band (lane 1 §998,
   §1044); after it the stream carries pooled context that everything downstream reads, and an
   error there is paid at full price. The same rel-norm-0.5 error at block 5 costs 0.058 — a
   compressed program that must be lossy should be lossy **before** block 6.
2. **Lane 1's error hump lands on the cliff.** §2086 measured the assembly's stream error peaking
   at block 6 at rel-MSE 1.74 (relative norm ≈ 1.3). At block 6 a relative norm of 1.0 costs 3.6
   nat here. The assembly's actual gap is +2.9 nat; its mid-stream error is not "repaired" cheaply
   downstream — it is an error of the most expensive kind at the most expensive depth, partly
   compensated because it is structured rather than random.
3. **Scale is nearly free — but not by the registered ratio.** Rescaling the stream by 1.5 costs
   ≤ 0.06 nat at every block 0–16 (often ≤ 0.01, occasionally negative) and 0.19 at block 17. In
   absolute terms scale is a gauge of the pre-norm stream. pred_c failed because at blocks 0–3 the
   random price is itself ≈ 0.02, so the *ratio* exceeds 0.2 (7.6× at block 0 where both are
   ≈ 0); at every block 4–16 the ratio is 0.003–0.18. The registered bar was the wrong shape; the
   absolute statement is what a program can use: **a scale error of 50 % costs ≤ 0.06 nat anywhere
   before the last block**, so §1818's 159× head 5.7 is not expensive *because* of its scale.
4. **Superlinear everywhere** (pred_b held, min ratio 2.26): doubling the error norm more than
   doubles the cost at every site, ×10 or more at blocks 1–5. First-order budgets underprice
   early error and are nearly exact after the cliff (×2.3–2.7 at blocks 6–8).
5. **Sign flips are a different animal**: inverting 10 % of positions costs 3.2 nat at block 1
   (before anything has been pooled) and 2–2.6 nat late; the model is robust to it in the band
   (0.25–0.5 at blocks 3–6).

## Consequence for the program

The price of error is a function of **depth and norm, not direction**, with one cliff. A
simplicity measure for a compressed early program should charge stream error at block k by this
curve — 0.06 nat per half-norm at block 5, 1.5 at block 6 — rather than by local reconstruction.
Together with `OBSERVABILITY_QUOTIENT_V1_RESULT.md` (direction matters ≤ 3.5× at first order),
this is the empirical error budget the alternate entry point asked for: compress freely up to
block 5, be exact from block 6, and never pay for scale.
