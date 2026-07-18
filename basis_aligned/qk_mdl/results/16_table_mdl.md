# Compressing the tables: three methods and a champion config

Logan's three candidates for making the stream tables MDL-efficient, audited in the
windowed-D W=4 harness (bilin18; uniform per-stream vq1024 = the prior baseline).
All structural bits use the frozen 32-bit convention; estimation data for every arm
is the same 524k early-slice tokens.

| arm | ΔCE (W=4) | atom/basis floats | index bits |
|---|---|---|---|
| full tables (reference) | +0.099 | 2,086M | — |
| uniform vq1024 | +0.081…+0.094* | 42.5M | 18M |
| **low-rank r=32 per stream** | **+0.074** | 59.3M | 0 |
| low-rank r=128 | +0.088 | 237M | 0 |
| **shared codebook k=4096** | +0.098 | **4.7M** | 22M |
| shared k=8192 | +0.119 | 9.4M | 24M |
| edge-guided k tiers | +0.087 | 55.3M | 16M |
| **COMBO: r=32 basis + vq1024 coefficients** | +0.089 | **2.5M** | 18M |

\* re-clustered per run; the spread is the k-means chaos band (results/14).

## Findings

1. **Low-rank r=32 beats the full tables** (+0.074 vs +0.099): each (V×1152) table is
   effectively a rank-≤32 object plus noise, and truncating the noise *helps* — the
   third instance of compression-as-denoising (after vq1024-is-free and H5-content
   filtering). This is the best-quality description.
2. **Sharing atoms across streams works** (4.7M floats for +0.098): most of the atom
   budget in per-stream vq was redundant across streams. Pushed too far it breaks
   (k=8192 worse than k=4096 — union k-means degradation).
3. **Edge-guided budget allocation is a wash**: giving causally-important streams more
   atoms (tiered 4096/1024/64 at matched budget) changed nothing (+0.087). Bits saved
   on unimportant streams were already nearly free; bits added to important ones hit
   the same denoising ceiling.
4. **The champion description**: r=32 basis per stream + vq1024 over coefficient rows —
   the entire long-range information flow of the 546M model in **2.5M floats + 18M
   index bits (~12 MB)** at +0.089. The coefficient quantization returns the denoising
   bonus, so quality-vs-bits now has a clean two-point frontier: r=32 plain (quality)
   vs r=32+vq (description).

Housekeeping: mlp17's table had 621 fp16 overflow entries (present harmlessly in all
prior vq runs — overflow rows isolate into their own clusters); sanitized at load from
now on.

Files: `../e4_table_mdl.py`, `../e4b_combo.py`, `../e4_table_mdl.json`.
