# 22 September — Does "the read side is simple" survive softmax? Three models, one protocol

**Question (Logan, 21 Sep):** is the bilin18 attention result — every head's pattern is a fixed positional kernel plus a low-rank content term, with the 162 kernels spanning a ~4-dimensional space — a property of transformers, or of this model's unnormalised squared attention?

**Answer in one line.** The structure survives softmax on our own architecture almost unchanged; on GPT-2 small it survives in kind (kernel + content, low-dimensional kernels) but not in price (the content is much higher rank, and absolute positions plus the first-token sink need their own term).

**Models.** (a) bilin18: 18 layers × 9 heads × 1152, unnormalised squared attention, rotary. (b) Its twin `Elriggs/gpt2-bilinear-18l-9h-1152embd`: identical in every respect (bilinear MLPs, rotary, QK-norm, tokenizer, 9,726 training steps) except softmax attention. (c) GPT-2 small: 12 × 12 × 768, softmax, learned absolute positions. Same tokenizer for all three, so the same held-out rows (192 × 512 tokens, skip7000) and the same fit rows are used throughout. Every number below is CE added on held-out text; every instrument replays the model's own forward to 5 decimals before an edit is made.

**Protocol (identical across models).** Value of a head = CE cost of replacing its output by its mean (mean ablation); joint value = all heads mean-ablated at once. Program per head, applied to the pre-softmax logits (for bilin18, to the pattern itself): off-diagonal logit(i, j) := κ_h(i−j) + [P_r(i, j) − κ_r(i−j)], with κ_h a per-offset kernel and P_r the head's own logit computed from rank-r factored query/key maps (exact rank by construction). Kernels and factors fitted jointly by Adam with early stopping on a validation split and parameters snapshotted at the validation minimum; then the 24 most valuable heads are mean-ablated inside the program to check that the program preserves what each head is worth.

## Results

| | bilin18 (squared) | softmax twin | GPT-2 small |
|---|---|---|---|
| native CE | 3.132 | 3.114 | 3.398 |
| joint value (all heads mean-ablated) | 3.996 | 3.388 | 3.345 |
| most valuable head | 0.3: 0.062 | 4.7: 0.092 | 0.1: 0.549 (layer 0 holds 0.89 of the 1.68 total) |
| kernels only, closed form (per-offset mean) | +1.45 | +1.79 | +5.80; +2.92 with the first-token column kept native |
| energy of the kernel matrix in its top 4 SVD shapes | 96.4% | 97.4% | 99.96% |
| cost of restricting kernels to 4 shapes (inside kernels-only) | +0.004 | +0.057 (8 shapes: 0) | +0.77 (the energy is a smooth logit trend, not what the loss uses) |
| **kernel + low-rank content, fitted** | **+0.072** (rec 0.982; rank 16 for 111 heads, 64 for 51; QK numbers ÷3.7) | **+0.088** (rec 0.974; same ranks; ÷3.4) | sink column native + rank 32: **+0.160** (rec 0.952; ÷1.8); rank 16: +0.322 (÷3.6); rank 64 (= full content for the 71 valuable heads): +0.075 (÷1.5) |
| head values preserved inside the program (Spearman / median ratio, top 24) | 0.84 / 1.08 | 0.71 / 1.04 | 0.54 / 0.85 at rank 32 |

**What the kernels look like.** In bilin18 (pattern units) they are short windows and taps. In the softmax twin (logit units) the dominant shape is a recency slope that goes *negative* at long range — the positional prior is written as a smooth function of distance on the logit scale and the exponential sharpens it, which is why a basis of two exponential windows and two power-law tails spans 92% of bilin18's kernels but only 54% of the twin's (√d, d, a constant and exp(−d/48) span 90% there). In GPT-2 the per-offset mean is dominated by the first-token sink and by absolute position; separating the sink column is necessary before any offset kernel makes sense.

## What I conclude

1. **Not the softmax.** On the twin, the whole read side of attention is a fitted kernel-plus-low-rank program at +0.088 CE with head values preserved — the same statement as for bilin18 within 0.016 CE. The unnormalised attention made the arithmetic cleaner (the kernel is literally the mean pattern), not the phenomenon.
2. **Low-dimensional kernels are general across the two rotary models**, but the shapes are model-specific and the specific curves are a basis convention either way (v741).
3. **GPT-2 is different in kind.** Two known facts — absolute positions and the first-token sink — break the per-offset template until handled; and its valuable heads (layer-0 token-matching heads) carry high-rank content: at the same compression ratio as bilin18 (÷3.6) it costs 0.32 vs 0.07. Whether that is the position encoding, the data, or the size is not separated here; a rotary softmax model of GPT-2's size would do it.

Files: `ops/run_softmax_values_kernels_v750.py`, `run_softmax_program_fit_v752.py`, `run_gpt2_values_kernels_v751.py`, `run_gpt2_program_fit_v753.py`, `run_gpt2_program_fit_uniform_v754.py`; backends `ops/softmax_backend.py`, `ops/gpt2_backend.py`; results `circuits/followups/softmax_*`, `gpt2_*`. Preregistered predictions and their scores are in each result JSON (v750 3/5, v751 3/5, v752 5/5, v753 3/5, v754 2/5 — every failure is a bar I set from bilin18's numbers that the other model did not meet).
