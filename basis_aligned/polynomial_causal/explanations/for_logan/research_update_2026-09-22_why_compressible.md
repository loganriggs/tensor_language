# Why are the bilinear models' attention heads so compressible? A multi-model ladder (22 September 2026)

**What you asked (22 Sep).** "Do try a few sets of pythia models, and then other models to continue to test the specific hypothesis of why we're seeing what we're seeing." The thing to explain: on bilin18 and its softmax twin, replacing every head's read-side (its query/key match) by a per-offset kernel plus a rank-16/64 content term keeps ~98% of what the heads are worth, while on GPT-2 small the same program at rank 32 keeps 95% and at rank 16 only 90%. Candidates: (H1) GPT-2's learned absolute positions vs bilin18's rotary; (H2) model size; (H3) training length; (H4) something about the bilinear/QK-normed architecture itself.

**Short answer.** Within one architecture (Pythia-160m, five checkpoints of the same run), compressibility falls steadily with training tokens: recovery at rank 32 goes 1.009 → 0.982 → 0.963 → 0.947 → 0.932 from 2B to 300B tokens. bilin18 was trained for roughly 5B tokens, and it lands almost exactly on that curve (0.982 at ~5B; the Pythia checkpoint at 8B is 0.982). So the most economical explanation of "why we see what we see" is **H3: the bilinear models are short-trained, and short-trained heads have simple read-sides.** Absolute positions (H1) are not the obstacle — OPT-125m, with learned absolute positions, is *more* compressible (0.964) than Pythia-160m with rotary (0.932) — and size (H2) is flat between 160m and 410m at the same token count (0.932 vs 0.935). There remain family-level shifts of a few hundredths that training length does not explain (OPT and SmolLM sit above the Pythia curve at their token counts); H4 cannot be separated from H3 with the models available, and I say below what would separate them.

![ladder](attention_compressibility_ladder_2026-09-22.png)

## The protocol (one protocol for every model)

Identical to the softmax-replication note, with one fix that the Pythia models forced.

- **Value of a head** = CE added when its output is replaced by its mean over 480 FineWeb rows; **joint value** = all heads mean-ablated at once; **recovery** of a program = 1 − (program's CE cost) / joint value.
- **Program per head**, applied to the pre-softmax logits: off-diagonal logit(i, j) := κ_h(i−j) + [P_r(i, j) − κ_r(i−j)], κ_h a per-offset kernel, P_r the head's own logit from rank-r factored query/key maps. Kernels and factors fitted jointly by Adam (300 steps), early-stopped on a validation split, snapshotted at the validation minimum; costs measured on 192 held-out rows; then the 24 most valuable heads are mean-ablated inside the program to check the program preserves what each head is worth (Spearman / median ratio).
- **The fix: row-centring.** Softmax does not see a constant added to a whole query row, but Pythia's raw logits carry row constants of 10⁴–10⁵ (the "massive activation" tokens). Estimating per-offset kernels on raw logits mixes those constants in, and the program collapsed (Pythia-160m recovery 0.64). Every kernel is now estimated on row-centred logits (mean over keys 1 ≤ j < i subtracted) and the program adds each row's native mean back; the first-token ("sink") column and the diagonal stay native. With centring Pythia-160m goes to 0.932. bilin18 has no softmax, so this was never an issue there; on GPT-2 it changes little (0.952 uncentred; the centred run v765 is queued).
- **Same text everywhere**: the same FineWeb rows, decoded from the GPT-2 tokenisation and re-tokenised per model (512-token rows). Evaluation CE is in each model's own tokens, so absolute CEs are not comparable across tokenisers; recoveries are.

## The numbers

| model | positions | tokens trained | native CE | joint value | rank 32: cost / recovery | rank 16: cost / recovery | values preserved (Spearman / median ratio, top 24) | kernel-matrix energy in top-4 shapes |
|---|---|---|---|---|---|---|---|---|
| Pythia-160m step 1000 | rotary (¼ dims) | 2.1B | 4.646 | 2.041 | −0.019 / **1.009** | +0.021 / 0.990 | 0.97 / 0.96 | 0.975 |
| Pythia-160m step 4000 | rotary | 8.4B | 3.794 | 2.898 | +0.053 / **0.982** | +0.152 / 0.947 | 0.86 / 0.98 | 0.960 |
| Pythia-160m step 16000 | rotary | 34B | 3.549 | 3.164 | +0.118 / **0.963** | +0.281 / 0.911 | 0.85 / 1.02 | 0.961 |
| Pythia-160m step 64000 | rotary | 134B | 3.447 | 3.245 | +0.172 / **0.947** | +0.336 / 0.896 | 0.77 / 1.09 | 0.947 |
| Pythia-160m final | rotary | 300B | 3.473 | 3.118 | +0.213 / **0.932** | +0.388 / 0.876 | 0.69 / 0.96 | 0.974 |
| Pythia-70m final | rotary | 300B | 3.869 | 2.673 | +0.319 / 0.881 | +0.518 / 0.806 | 0.94 / 1.10 | 0.982 |
| Pythia-410m final | rotary | 300B | 3.098 | 3.526 | +0.227 / 0.935 | +0.308 / 0.913 | 0.83 / 1.05 | 0.951 |
| OPT-125m | learned absolute | 180B | 3.435 | 3.215 | +0.116 / **0.964** | +0.233 / 0.928 | 0.66 / 1.07 | 0.992 |
| SmolLM-135M (from the log; result file pending re-run) | rotary (full dims), GQA | 600B | — | 3.17 | +0.142 / **0.955** | +0.376 / 0.881 | 0.84 / 1.10 | 0.86 |
| GPT-2 small (uncentred, v753) | learned absolute | ~40B? (WebText, epochs unknown) | 3.398 | 3.345 | +0.160 / 0.952 | +0.322 / 0.905 | 0.54 / 0.85 | 0.9996 |
| **bilin18** | rotary, QK-norm, squared attn | ~5B (9726 modded-nanogpt steps) | 3.132 | 3.996 | mixed ranks 16/64: +0.072 / **0.982** | — | 0.84 / 1.08 | 0.964 |
| **softmax twin** | rotary, QK-norm, softmax | ~5B | 3.114 | 3.388 | mixed ranks 16/64: +0.088 / **0.974** | — | 0.71 / 1.04 | 0.974 |

Data: 64 fit rows (kernels), 96 validation rows, 192 evaluation rows of 512 tokens; head values from 480 rows. Pythia's per-step token count is 2,097,152 (batch 1024 × 2048).

## What each comparison says

**Training length (H3) — supported, and it is the only factor with a clean monotone effect.** Same model, same data order, five checkpoints: recovery drops at every step, at both ranks, and the drop is roughly linear in log(tokens) (≈ −0.035 per decade at rank 32, −0.055 at rank 16). The kernel-only cost tells the same story from the other side: at step 1000 the kernels alone (no content) already recover 40% of the joint value; by the end they recover 21%. Early heads are mostly positional; content reads grow, and grow in rank, with training. The values-preserved column also degrades (Spearman 0.97 → 0.69): late in training the program not only costs more, it re-orders which heads matter.

Why the bilinear models look like the step-4000 checkpoint: 9726 modded-nanogpt steps at the default 0.5M-token batch is ≈5B tokens — between Pythia's step 1000 (2B) and step 4000 (8B). Their recoveries (0.982 / 0.974, at mixed ranks that average below 32) sit where the Pythia curve predicts. I do not think this is a coincidence.

**Absolute positions (H1) — falsified as the explanation of GPT-2.** OPT-125m has learned absolute positions and no rotary, and is the most compressible of the long-trained models (0.964). Pythia-160m has rotary and is the least (0.932). GPT-2 small's 0.952 is in between. Whatever makes GPT-2 harder than bilin18, it is not that positions are absolute.

**Size (H2) — flat above 160m; 70m is an outlier the other way.** At 300B tokens, 160m → 410m moves recovery 0.932 → 0.935 (rank 32) and 0.876 → 0.913 (rank 16; the bigger model's heads have more headroom for rank 16 to be enough). Pythia-70m is *less* compressible (0.881): with only 6 layers × 8 heads, each head carries more, and its per-head content is denser. So size does not rescue the bilinear models' number — a 160m Pythia trained as long as they were would be just as simple.

**Family-level shifts — real, unexplained, small.** At matched tokens OPT-125m (180B) sits ~0.02 above the Pythia curve and SmolLM (600B) ~0.03 above it. Two things I can see in the data: OPT is strongly sink-anchored (kernels-only costs +5.05 with the sink column also replaced, +1.62 with it kept native — the first-token column carries most of the read-side work, and the program keeps it native), and SmolLM has one head (9.3) worth 1.03 nats on its own, a third of the joint value, which the program preserves at rank 32. Both features make a model *look* more compressible under this program, because the parts that carry the load are the parts the program does not touch. I would not read those shifts as architecture-level simplicity.

**Architecture (H4) — cannot be separated from H3 with these models.** The bilinear models differ from Pythia in four ways at once (QK-norm, bilinear MLPs, squared/softmax attention, ~5B tokens on FineWeb). The token count alone predicts their recovery to within the noise between checkpoints, so nothing here requires an architectural explanation. The test that would separate the hypotheses is a QK-normed rotary softmax model trained for ≥100B tokens, or the bilinear models' own training checkpoints if any survive — on the H3 story they should show the same descent.

## Two things to keep in mind

- **Evaluation is in-distribution for the bilinear models** (trained on FineWeb, evaluated on FineWeb) and out-of-distribution for the others (Pile, WebText, SmolLM-corpus). That could inflate the bilinear recoveries a little, but it cannot produce the Pythia checkpoint curve, which is the evidence for H3.
- **Every number is recovery relative to a mean-ablation value on 192 rows.** Joint values differ (2.0 to 4.0 nats), so a 0.03 difference in recovery is 0.06–0.12 nats. The Pythia checkpoint steps are 0.02–0.03 apart in recovery and monotone in both ranks, which is well above the run-to-run variation I have seen (±0.005 between re-fits of the same model).

## What the ladder does not change

The results Logan asked about earlier still stand: the four-shape kernel dictionary is a rotation-invariant fact about the kernel *matrix* (its rank-4 energy), not a claim that particular shapes are canonical; the compression comes from the low-rank content term, with the kernels 83k of the program's 25.9M numbers; and the softmax twin replicates bilin18, so squared attention is not what makes the heads simple. This note adds the reason both bilinear models are simple: they are early in training.

## Runs

v755/v756 (uncentred Pythia-160m/70m, kept as controls), v760–v762 (Pythia 160m/70m/410m, centred), v766–v769 (Pythia-160m steps 4000/1000/16000/64000), v763 (OPT-125m), v764 (SmolLM-135M; tripped its price bar after every stage completed, re-queued with a corrected bar), v765 (GPT-2 centred; crashed on a rotary branch taken for a non-rotary model, fixed and re-queued). Results in `basis_aligned/bilinear_quotient/circuits/followups/{pythia*,opt125m,smollm135m,gpt2_*}_v7xx_result.json`; scripts `ops/run_pythia_all_v76x.py`, `ops/run_hf_all_v763/4.py`, `ops/run_gpt2_all_v765.py`, backends `ops/{pythia,hf,gpt2}_backend.py`.
