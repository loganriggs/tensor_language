# Do pretrained LLMs organize graph walks the way Park et al. found — and what are they doing?

Question from the toy results: our from-scratch **softmax** models anti-organize (neighbors
pushed apart) even when perfect at the task, yet Park et al. saw real LLMs (Llama-3.1-8B,
softmax) organize positively. Is that a scale thing? And can our toy-model account
(prediction-as-representation + own-token content sign) explain what the real models do?

Setup (`llm_reps.py`): random walks on token-labeled graphs — each node of a 4×5 grid, a
12-ring, and a 7-ring is assigned a random common English word (single-token, one fixed
assignment per model), and the walk is fed to the pretrained model as a plain word
sequence, 400 words, 96 walks. Same measurements as the toys: windowed (50-token) mean
representation per node, organization = correlation between representation similarity and
graph adjacency — but now at **every layer** — plus in-context task performance, plus the
ownU/nbrU content coefficients read in each model's own unembedding basis. Models: GPT-2
(124M), Pythia-410M, Qwen2.5-1.5B/3B, Qwen2.5-7B (8-bit). (Park's exact Llama-3.1-8B is
gated on HF and this box has no token; Qwen2.5-7B is the same class.)

## 1 · Yes — and it does not need 8B parameters

Every model, including 124M GPT-2, learns the walk task in context (legal top-1 rate on
the grid 0.82–0.99 by 400 tokens; ~1.00 on rings) and organizes the grid **positively**:

| model | grid organization (last layer, ctx 400) | best layer | PC1/PC2 ↔ grid harmonics | top-2-PC Dirichlet energy (random ≈ 2) |
|---|---|---|---|---|
| GPT-2 124M | +0.02 | +0.34 (L11 of 12) | 0.79 / 0.77 | 0.75 |
| Pythia-410M | +0.44 | +0.44 (L24 = last) | 0.71 / 0.72 | 0.63 |
| Qwen2.5-1.5B | +0.42 | +0.42 (L28 = last) | **0.94 / 0.97** | 0.63 |
| Qwen2.5-3B | +0.47 | +0.47 (L36 = last) | 0.80 / 0.80 | 0.57 |
| Qwen2.5-7B | +0.44 | +0.44 (L28 = last) | 0.88 / 0.87 | 0.86 |

All five pass the Park Theorem-5.1 test: the grid's two spectral coordinates appear in the
**top two principal components** (each carrying ~25% of variance). So the phenomenon needs
neither 8B scale nor anything Llama-specific — a 124M model has it mid-stack.

![organization by layer and context](figures/llm_org.png)

![top-2 PC maps](figures/llm_maps.png)

## 2 · Why real softmax models organize positively when our toy softmax didn't

The toy account transfers directly. Reading each model's final-layer node representations
in its own unembedding basis (same regression as the toys):

- **nbrU (neighbor evidence — the prediction itself) is large and positive in every LLM**
  (last layer, grid: +1.4 GPT-2, +26 Pythia, +13/+12/+33 Qwen). The representation carries
  the prediction, and that channel alone builds the positive map — exactly as in the toys.
- **ownU (own-token content) is positive at the last layer in every LLM on the grid**
  (+0.18 … +5.38). None of them write own-token *suppression* into the residual stream.

That second line is the answer to the puzzle. Our toy softmax stack — trained from scratch
on graph walks only — implemented "don't predict impossible tokens" by writing *negative*
recent-token content into the stream, which cancels the neighbor overlap and inverts the
map. Pretrained LLMs never learned that habit: natural text is full of recurrence (recent
tokens are *more* likely to appear again — the same pressure that builds induction heads
as copiers), so their in-context circuit keeps own/recent-token content positive or
neutral, and the always-positive prediction channel sets the geometry. In the toy
vocabulary: natural-language pretraining is deep in the "reversible" regime, so the
anti-mode is never selected. The toy anti-organizing softmax was an implementation choice
available to tiny from-scratch models, not a property of softmax attention.

![behavior and content coefficients](figures/llm_coeffs.png)

## 3 · The layer profile (what the toys couldn't show)

Organization rises through the stack and peaks late-but-not-last; GPT-2's final layer
drops to ~0 (+0.34 → +0.02) and the smallest rings *invert* at the very end. A 2-layer toy
has no "middle" — its stream is the output stream, so whatever output bookkeeping the
model does sits directly on top of the map. Deep models keep the map clean mid-stack and
only assemble output-specific content at the end.

## 4 · An anomaly worth keeping: the 7-ring inverts at the readout

On the smallest (odd) ring, every LLM anti-organizes at the final layer (−0.25 GPT-2,
−0.27 Pythia, −0.53/−0.41/−0.55 Qwen) while solving the task at ~1.00 — and the mid-stack
positive ring is weak in the Qwen models. The toy softmax model's "7-pointed star" is not
gone from real LLMs; it shows up at the output end on the smallest graphs, where the
recent past covers most of the graph. Not explained; logged as the same open question as
the toy softmax default ("why does the readout end prefer suppression-style content?").

## Reproduce

```bash
python llm_reps.py gpt2 --batch 24        # or any HF causal LM; --8bit for 7B
python llm_figs.py                        # figures/llm_{org,maps,coeffs}.png
# per-model outputs in runs_llm/<tag>/{org,behavior,coeffs,meta}.json + reps.pt
```

Caveats: one word-labeling per model (fixed seed), 96 walks, window 50; ownU/nbrU read
per-layer against the final unembedding (only exactly interpretable at the last layer);
7-ring statistics rest on 21 node pairs; Qwen2.5-7B measured in 8-bit quantization.
