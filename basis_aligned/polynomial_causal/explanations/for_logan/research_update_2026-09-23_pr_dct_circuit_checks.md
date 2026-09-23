# Do PR-regularized DCT factors find real sparse circuits? — No: the penalty finds where one MLP unit sits on an attention-carried interaction, and the factors it returns mostly do not survive a change of seed (23 September 2026, 01:00 UTC)

Claude (Fable). Project directory `pr_dct_checks/` (moved there from `for_logan/dct/`); full tables in `pr_dct_checks/RESULTS.md`, raw JSON in `pr_dct_checks/results/`.

## 1. Context: what was claimed and why it needed checking

**The method.** A Deep Causal Transcoder (DCT) looks for triples (l, r, u): two input directions at a source layer and an output direction at a target layer such that perturbing the residual by αl + βr moves the target along u *jointly* — the score is the mixed second derivative u·∂²Δ/∂α∂β at 0, the "u·H[l, r]" term. AJ (ajskateboarder) added a **participation-ratio (PR) penalty**: for every MLP hidden unit strictly between the source and target blocks, compute the same mixed derivative of that unit's activation, call the vector c, and penalise PR(c) = (Σc²)²/Σc⁴, the effective number of units carrying the interaction. His objective per factor is ½(u·H[l,r])² − λ·S·PR/D, fitted by a fixed-point iteration (the factors are replaced by their normalised gradients each round).

**The claim** (his note, "shallow weight circuits"): on the bilinear-MLP transformers, the penalty drops the median PR from ~300–400 to ~1 (a 400× sparsification) while factors keep 70–100% of their causal strength, which would mean the interactions are routed through single neurons — sparse, shallow circuits.

**Why that inference is not safe** (from the handoff README, which I followed): PR is scale-invariant, so it measures how *concentrated* the MLP-mediated part of an interaction is, never how *much* of the interaction is MLP-mediated; AJ's PR range excludes the source block's own MLP and every attention layer, exactly where a bilinear interaction can live; a bilinear hidden unit (a·n)(b·n) is itself a rank-1 interaction, so PR ≈ 1 may mean nothing more than "the factor points at one neuron"; and the evidence was 4 training / 8 held-out prompts with no stability check. The handoff shipped four checks to settle this, each computed per factor on held-out prompts:

- **E1 pathway completeness** — freeze a group of units at their clean values and measure what fraction of u·H[l,r] disappears (top-k units in the PR range, k random units, all MLPs in the range, the source-block MLP, all attention; 1 = the whole interaction goes through the group, values are signed and can exceed 1 when pathways cancel).
- **E2 neuron read-off** — how well (l, r) aligns with the effective read directions of the best single unit, against a random-pair null, and how much of the factor's energy a raw neuron gets when scored *as if it were a factor*, with no optimisation.
- **E3 finite ablations** — the finite-size interaction f(αl+βr) − f(αl) − f(βr) + f(0) projected on u, with the top-k units patched to clean versus random units, at 5% and 20% of the mean residual norm.
- **E4 stability** — Hungarian-matched similarity of factors across seeds and across disjoint halves of the training prompts, against a random-dictionary null.

## 2. What I ran

- **Models**: all four Elriggs 18-layer checkpoints — bilinear MLPs with softmax attention (`gpt2-bilinear-18l-9h-1152embd`), SwiGLU with softmax attention, and the two squared-attention variants. Source block 8, target block 12, so the PR range is blocks 9–11 (13,824 units); the last 3 positions are scored.
- **Data**: AJ's prompts (the AdvBench `target` strings, "Sure, here is …", right-padded with EOS to 32 tokens — so the three scored positions are usually padding attending to the text), and an off-distribution set of real FineWeb text (first 32 tokens of documents, no padding).
- **Scale**: E0 at AJ's scale (4/8 prompts, 4 factors, 5 iterations, weights 0/1/10/100), then the handoff defaults (32 train / 64 held-out, 8 factors, 10 iterations, seeds 0 1 2, weights 0 / 0.1 / 1), then a 30-iteration convergence run.
- **Code**: running AJ's cloned repository against the real checkpoints is blocked in this environment, so I ran the checks on this repository's own model loader (the same modded-nanogpt code as his `tensor_model.py`; forward parity against his `TensorMiddleSpan` is exactly 0 on a random model, and the adapter matches the repository's block code exactly on the real one) and on a port of his fitting loop written after reading it in full (same init, QR, mixed Hessians, gradient objective, fixed-point update, penalty scale). The port passes its own toy tests (a planted single-unit circuit is recovered; weight 0 reproduces the plain DCT bit for bit). E1–E4 are the handoff's code, plus two measures I added (next section). Two bugs fixed on the way: SiLU's backward has no forward-mode derivative (E1–E3 need second-order forward AD), so the adapter computes it from primitives; and the per-unit pullbacks are chunked.

## 3. One structural fact first

u·H[l, r] is a mixed second derivative, so it is **symmetric in l ↔ r**. The score is a symmetric bilinear form in (l, r), whose maximum over unit vectors is at l = r (the top eigenvector), with a continuum of equally good pairs when eigenvalues tie. For a single bilinear unit, (l, r) = (a, b) and l = r = (a+b)/√2 score identically, and the fitting lands on the latter. On the real models **every penalised factor has cos(l, r) = 1.00**: the "asymmetric" DCT is in practice a symmetric one, (u, v) with l = r = v. This matters for reading the checks: the handoff's E2 alignment and E4 similarity compare l with a and r with b separately and are capped near 0.5 for a factor that *is* a neuron, so I report **span-based versions** alongside them (canonical correlations between span{l, r} and span{a, b}, and between two factors' spans; a random pair scores 0.01).

## 4. Results

### 4.1 The headline replicates on AJ's data and shrinks off it

| setting | median held-out PR, w = 0 → penalised | held-out energy kept |
|---|---|---|
| bilinear, AdvBench, AJ's scale (4/8 prompts) | 305 → 1.5–1.8 | 1.2–1.4× (more than the unpenalised fit) |
| bilinear, AdvBench, 32/64 prompts, 3 seeds | 386 → 3.1 / 2.7 (w = 0.1 / 1) | 0.70 / 0.66 |
| bilinear, FineWeb (off-distribution) | 190 → 12.5 / 10.2 | 0.98 / 0.97 |
| SwiGLU, AdvBench | 72 → 1.1 / 1.1 | 0.97 / 0.98 |
| bilinear + squared attention, AdvBench | 1576 → 834 / 173 | 0.96 / 0.73 |
| SwiGLU + squared attention, AdvBench | 192 → 141 / 123 | 1.04 / 0.76 |

The "~400 to ~1 with 70–100% retained" numbers are real for the softmax-attention models on AdvBench prompts. Off-distribution the drop is 15×, not 400×. On the squared-attention models the penalty does not concentrate anything (no factor reaches PR ≤ 5 at any weight) and at w = 1 costs a quarter of the energy.

### 4.2 What the concentrated factors are (E1, E2, E3)

Per-factor counts over the 24 penalised factors (8 × 3 seeds) at w = 1, bilinear / AdvBench: 11 have PR ≤ 5 and 10 have top-1 completeness in [0.5, 1.5]; the other 13 stay diffuse (PR 15–130) with 70–95% of their interaction in the **source-block MLP**, which the PR range does not see. So the penalty concentrates about half the dictionary and leaves the scale-invariance loophole open in the other half; the medians AJ would report (PR 2.7) hide this.

For the concentrated half the picture is consistent across bilinear and SwiGLU:

- **Freezing all attention removes 52–74% of every factor's interaction**, penalised or not, on every softmax model (0.98–1.00 on the squared-attention models). The group fractions sum to about 2 while freezing everything gives exactly 1.00, so the pathways overlap: the interaction passes through attention *and* through one MLP unit. "PR ≈ 1" describes only the MLP-unit part of that route.
- **They are partly neuron-shaped.** Span alignment to the best unit is 0.3–0.5 (null 0.01). On AdvBench a raw neuron scored as a factor recovers 34–43% of the factor's energy; on FineWeb it recovers **1.8–2.1×** the factor's energy for 21 of 24 factors — the fit lands near a neuron and a plain neuron is the better factor.
- **On SwiGLU two factors recur exactly in all three seeds** with PR 1.1 and top-1 completeness 1.00 / 0.99 — genuine single-unit interactions at the derivative level (source MLP 0.02–0.03; attention 0.53 / 0.82 by overlap). But patching that unit to clean at 5% of the residual norm changes the finite interaction by **+211%** and **−950%** — the finite response is outside the quadratic regime, and its sign is not the derivative's sign. On the bilinear model E3 and E1 agree at 5% (top-1 patching removes 0.4–0.65 for concentrated factors, random units 0.00) and disagree at 20% for everyone.

### 4.3 Stability (E4): the dictionary is mostly a function of the seed

| setting (32/64 prompts, 8 factors) | cross-seed matches > 0.8, of 24 pairs | best cross-seed match | split matches > 0.8, of 8 |
|---|---|---|---|
| bilinear, AdvBench, 10 iterations, w = 1 | **0** | 0.58 | 3 |
| bilinear, AdvBench, **30 iterations** (converged), w = 1 | 3 (one factor, in all three seeds, at 1.00) | 1.00 | — |
| bilinear, FineWeb, w = 1 | 10 (a few factors at 0.89–1.00) | 1.00 | 2 |
| SwiGLU, AdvBench, w = 1 | 6 | 1.00 | 3 |
| bilinear + sqrd attention, w = 1 | 3 | — | 0 |

Random-dictionary null: 0.0001. At AJ's iteration count the fits have not converged (energies still rising) and no factor recurs across seeds on his data. Converged, exactly one factor recurs in every seed and the other seven are init-dependent; the penalty's effect is itself seed-dependent (same weight and scale: median PR 2.8 / 97.7 / 1.6 across seeds 0 / 1 / 2). The handoff's split test fits both halves from seed 0 and is confounded with shared initialisation (split matches exceed cross-seed matches). By the README's own reading table, "E4 matches near the null" means the per-factor retention numbers are not meaningful, and that is the state of most of each dictionary.

## 5. Verdict, in the README's terms

- **Not** "PR ≈ 1 but completeness low, source-MLP high" for the concentrated half (that describes the diffuse half and the unpenalised fits).
- **Not** "sparse mediation of inputs that aren't neuron-aligned" (the interesting case): alignment is well above null and a raw neuron matches or beats the factor.
- **Closest row: "the factor is a neuron"**, with two qualifications the table did not anticipate. (i) The neuron is one stage of an interaction that attention carries 50–75% of, so the circuit is not "shallow" in the sense of living in the MLP units; PR cannot see that. (ii) Except for one to two factors per dictionary, the neurons found depend on the random start, so the method does not reliably find *which* neuron either. On the squared-attention models the penalty fails outright because the interactions are attention interactions.

Negative results, stated plainly: the 400× figure is specific to AJ's prompts; half of each penalised dictionary is not sparse; no factor on AJ's data recurred across seeds at his iteration count; finite ablations contradict the derivative picture for the cleanest SwiGLU factors at the tested scales; the squared-attention variants do not concentrate at all.

## 6. Caveats and what was not done

- AJ's own code was not executed on the real checkpoints (blocked); the port and the in-repo loader were validated as described, and the reproduction of his headline numbers is itself evidence the port is faithful. AJ's `swiglu` results, if any, could not have used `F.silu` under forward-over-forward AD as his code is written.
- E3 was run only at 5% and 20% of the residual norm; the SwiGLU sign reversals call for a ≤1% scale, which was not run.
- E5 (the weight-space, moment-only fit) applies only to norm-free models and was not run on the real ones.
- 10 iterations is AJ's default and is not converged; the 30-iteration run was fits-only (no E1–E3) and on AdvBench only.
- The squared-attention SwiGLU run's E4 (split fits) was still finishing when this note was written; its E0–E3 numbers are in the table.

**Files.** `pr_dct_checks/RESULTS.md` (all tables), `results/e0_ajscale_bilinear.json`, `results/full_{bilinear,swiglu,bilinear-attn,swiglu-attn}_advbench.json`, `results/full_bilinear_fineweb.json`, `results/conv30_bilinear_advbench.json`; code in `circuit_checks/` (handoff + `inrepo_model.py`, `dct_fit.py`, span measures in `checks.py`), runner `scripts/run_checks.py`, tables `scripts/summarize.py`, tests `tests/test_toy.py` (7) and `tests/test_fit.py` (3).
