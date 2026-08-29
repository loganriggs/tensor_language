# High-Rigor Circuit Detection/Extraction in Real (Non-Toy) Models: A Prioritized Literature Map

## TL;DR
- The papers that most cleanly clear your three-criteria bar in *real* models are: **Prakash et al. "Fine-Tuning Enhances Existing Mechanisms" (entity tracking, Llama-7B/Vicuna/Goat/Float)** and **Feng & Steinhardt "How do LMs Bind Entities" (binding-ID subspace, LLaMA-13B)** for FUNCTIONAL EQUIVALENCE + subspace-level isolation; **Hanna et al. "greater-than" (GPT-2)** and **Anthropic "Biology of a Large Language Model" (Claude 3.5 Haiku, addition)** for PREDICTION of OOD/edge-case behavior; and **Wu et al. Boundless DAS (Alpaca-7B)** for subspace-level, OOD-robust functional equivalence. Clean surgical ablation is the *weakest-supported* criterion — the honest state of the art is Marks et al. "Sparse Feature Circuits"/SHIFT plus a large negative literature (Hydra effect, faithfulness-metric fragility, subspace illusions).
- Your specific object of interest — the *shared, conditional, subspace-level* mechanism across polysemantic components — is directly addressed by **binding-ID subspaces (Feng & Steinhardt; Prakash binding), DAS/Boundless DAS/MDAS (Geiger, Wu, Huang), successor-head "mod-10" subspaces (Gould et al.), copy-suppression rank-1 QK/OV (McDougall et al.), and cross-layer-transcoder attribution graphs (Anthropic)** — but each is shadowed by a matching negative result (Makelov subspace illusion; Hydra self-repair; faithfulness-metric non-robustness) that you should treat as part of the benchmark's ground truth.
- For your benchmark on "verifiable component replacement," the most transferable evidence standards are **interchange intervention accuracy (IIA)** from the causal-abstraction line and **faithfulness/completeness/minimality** from IOI — but you must adopt the corrections from Miller et al. (metrics not robust), Makelov et al. (dormant pathways), and the Hydra/self-repair work (ablation ≠ importance), or your scores will be artifacts of ablation choice.

## Key Findings

**Three criteria, at a glance (best-in-class):**
1. **PREDICTION of OOD / margin behavior:** greater-than circuit (GPT-2) predicts *over-application* to "less-than"-shaped contexts; Anthropic addition circuit (Claude 3.5 Haiku) shows the same `_6+_9→_5` lookup feature firing in astronomy tables, spreadsheets, and citation years, confirmed by intervention; Boundless DAS shows Alpaca's two-boolean price-tagging algorithm is robust across input/instruction changes.
2. **FUNCTIONAL EQUIVALENCE / EXTRACTION:** IOI (GPT-2, faithfulness/completeness/minimality); entity-tracking circuit transfers with 0.88–0.97 faithfulness across four Llama variants; Edge Pruning recovers >99.9%-sparse circuits matching full-model performance up to CodeLlama-13B; sparse feature circuits (Pythia/Gemma); OpenAI weight-sparse transformers (fully reverse-engineered, but trained-to-be-sparse, not a real dense model).
3. **CLEAN / SURGICAL ABLATION:** SHIFT (Marks et al.) ablates human-judged spurious features to fix a classifier; RAVEL/MDAS disentangles co-located attributes; but this is heavily undercut by Hydra-effect self-repair, backup heads, and faithfulness-metric fragility.

**The redundancy problem is real and is the central obstacle.** The IOI circuit itself surfaced "backup name mover heads" that activate only when primary heads are ablated. The Hydra effect (McGrath et al., arXiv 2307.15771, Chinchilla-7B) shows that after ablating an attention layer, downstream attention layers "collectively act to restore approximately 70% of the reduction in token logits" at middle layers — and crucially the model "was trained entirely without dropout, stochastic depth, or layer dropout," ruling out regularization as the cause. This means naive ablation-based importance systematically misstates the mechanism, which is exactly your criterion-3 concern.

## Details

### Criterion 1 — PREDICTION (OOD / edge-case behavior from circuit understanding)

**Hanna, Liu, Variengien — "How does GPT-2 compute greater-than?" (NeurIPS 2023, arXiv 2305.00586).** Model: GPT-2 small. Task: year-span "The war lasted from the year 1732 to the year 17__", predict two-digit end-years > YY (full-model baseline probability difference = 0.817; correct-guess rate 0.992). Method: path patching + logit lens + neuron-level attribution. Circuit: **MLPs 8–11 compute the greater-than operation "in tandem, and in steps"** (MLPs 9/10 show an upper-triangular logit-lens pattern upweighting years > YY; MLP 11 enforces a max duration ~50 yrs), with attention heads (a9.h1; a8.h11, a8.h8, a7.h10, a6.h9, a5.h5, a5.h1) transmitting the start year YY in embedding space. No single neuron computes greater-than (sparse, distributed over MLP-10 neurons). Evidence strength: restricting to the circuit yields "The probability difference is 72.7% (89.5% of the original) and the cutoff sharpness is 8%—sharper than pre-patching"; circuit-node patching drops performance to −36.6% (necessity). **PREDICTION payoff:** the same circuit activates across new templates (price-range template **88.9%** loss recovery; "started/ended" template **98.8%**) AND the authors *predicted over-generalization* — prompts like "ended in the year 17YY and started in the year 17__" or "7YY BC to the year 7__" wrongly trigger greater-than with the identical circuit, while GPT-2 "even failed at some tasks that were solvable using the greater-than circuit, like '17YY is smaller than 17'; it always predicted YY." On monotonic-increase number sequences only 52.7% recovered until adding MLP 7 + two extra heads (a7.h11, a6.h1), then ~82.8%: "Similar tasks seem to use similar, but not identical, circuits." Caveat (authors' own): mechanism is "something in between" generalization and memorization; "could function internally as a lookup table"; no formal held-out predictive-accuracy test.

**Anthropic — "On the Biology of a Large Language Model" + "Circuit Tracing" (Transformer Circuits Thread, 2025).** Model: Claude 3.5 Haiku (18-layer companion model for methods). Method: **cross-layer transcoder (CLT)** replacement model with 30M features → **attribution graphs**, validated by feature inhibition/swap interventions; feature labels/supernodes chosen *before* measuring perturbation results. **Addition case study (strongest OOD-prediction evidence in the corpus):** Claude computes a+b via parallel "bag of heuristics" — low-precision "add ~57" features + "lookup table" features (e.g., "add numbers ending in 6 and 9 → ends in 5"). **Generalization confirmed by intervention:** the `_6+_9→_5` lookup feature fires far outside arithmetic — astronomy measurement tables, financial spreadsheets (arithmetic sequences), and academic citation years (volume 36 / founding year 1959 → publication year ends in 5). Suppressing the lookup feature changes the output; swapping `_6+_9`→`_9+_9` changes 1995→1998. **Other predict-then-confirm cases:** (a) *Hallucination/entity recognition* — a default "can't answer" circuit inhibited by "known entity/answer" features; they predicted and confirmed that promoting "known answer" features induces hallucinations (e.g., inventing a sport for "Michael Batkin"), and that inhibiting them induces refusal; a natural Karpathy-misattribution hallucination is explained as a known-answer "misfire." (b) *Jailbreak* ("BOMB" acrostic) — predicted that removing punctuation delays refusal onset (because "new sentence" features gate refusal); confirmed empirically. Caveats (authors', verbatim): "we've found that our attribution graphs provide us with satisfying insight for about a quarter of the prompts we've tried... even in our successful case studies, the discoveries we highlight here only capture a small fraction of the mechanisms of the model," and hallucination interventions require unnaturally high strengths. This is an existence-proof methodology, not a completeness guarantee.

**Wu, Geiger, Icard, Potts, Goodman — "Interpretability at Scale: Boundless DAS" (NeurIPS 2023, arXiv 2305.08809).** Model: Alpaca-7B (instruction-tuned LLaMA). Task: "Price Tagging" numeric reasoning. Finding: Alpaca implements a causal model with **two interpretable boolean variables**; the alignment of neural subspaces to these variables (found by a learned orthogonal rotation, replacing DAS's brute-force search) is **robust to changes in inputs and instructions** — an OOD-generalization claim backed by high IIA. This is simultaneously a criterion-1 (robust/OOD) and criterion-2 (functional-equivalence via IIA) result and, crucially, operates at the *distributed subspace* level you care about.

### Criterion 2 — FUNCTIONAL EQUIVALENCE / EXTRACTION (verified surrogate/replacement)

**Prakash, Rott Shaham, Haklay, Belinkov, Bau — "Fine-Tuning Enhances Existing Mechanisms: A Case Study on Entity Tracking" (ICLR 2024, arXiv 2402.14811).** Models: **Llama-7B, Vicuna-7B, Goat-7B, Float-7B** (real 7B models). Task: box/entity tracking ("Box R contains the rabbit… which item does Box R contain?"). Method: path patching + **DCM (Desiderata-based Component Masking)** + **CMAP (cross-model activation patching)**. Circuit: 90 heads in four groups (50/10/25/5 in Groups A/B/C/D — last-token, query-label, and previous-query-label positions; prominent head e.g. L21H3). **Functional-equivalence payoff:** the *same* circuit discovered in Llama-7B transfers to the fine-tuned models with no re-optimization — "Vicuna-7B has almost a perfect faithfulness score of 0.97, while Goat-7B and FLoat-7B exhibit slightly lower scores of 0.89 and 0.88, respectively" (random circuits ≈ zero accuracy) — evidence that fine-tuning "enhances rather than alters" a shared mechanism. Directly relevant to your "shared across components/models" object.

**Feng & Steinhardt — "How do Language Models Bind Entities in Context?" (ICLR 2024).** Model: LLaMA family (up to 13B). Finding: LMs bind entities↔attributes via **abstract binding-ID vectors that occupy a continuous "binding subspace" with a metric structure** (nearby IDs are confusable, far-apart IDs reliably distinguishable); erasing binding info (subtracting ΔE, ΔA) drops accuracy to chance; binding IDs generalize across the binding task. A near-perfect example of your target: a **shared subspace that multiple components read/write when doing binding but not otherwise**, isolated causally. Follow-ups: **"Representational Analysis of Binding" (arXiv 2409.05448)** finds a low-rank **OI (object-identity) subspace** in Llama2-7B (and code-fine-tuned Float-7B) that causally affects binding; **"A retrieval-conditioned rebinding circuit" (arXiv 2606.08644, 2026)** finds the binding signature lives in Q/K subspaces in Gemma but K vectors in Llama — a redundancy/family-dependence result you'd want in a benchmark.

**Wang, Variengien, Conmy, Shlegeris, Steinhardt — IOI (ICLR 2023, arXiv 2211.00593).** Model: GPT-2 small. The canonical "in the wild" circuit: **26–28 heads in 7 classes** (name movers, S-inhibition, duplicate-token, induction, previous-token, backup/negative name movers). Evaluated by **faithfulness, completeness, minimality**. This is the reference standard for extraction metrics — but the paper itself documents redundancy (backup heads) and admits it does **not** fully understand adversarial/edge-case behavior (e.g., S-inhibition heads attending to duplicated IO in natural sentences). Treat IOI as the "clean-ish two-head story that breaks down under scrutiny" that motivates your whole inquiry.

**Bhaskar, Wettig, Friedman, Chen — "Finding Transformer Circuits with Edge Pruning" (NeurIPS 2024, arXiv 2406.16778).** Models: GPT-2 up to **CodeLlama-13B**. Method: gradient-based pruning of *edges* (not nodes) via learned binary masks. Results: GPT-2 circuits with <half the edges of prior methods at equal faithfulness; scales to 100K IOI examples; **perfectly recovers Tracr ground-truth circuits** (note: Tracr = toy); on CodeLlama-13B finds two circuits at **>99.96% sparsity matching full-model performance** for instruction-prompting vs in-context learning. Best-in-class for scalable, faithful *edge-level* extraction in a real large model.

**Marks, Rager, Michaud, Belinkov, Bau, Mueller — "Sparse Feature Circuits" (ICLR 2025, arXiv 2403.19647).** Model: Pythia-70M/others via SAEs. Replaces polysemantic heads/neurons with **monosemantic SAE features** as circuit nodes, using indirect-effect (do-calculus) attribution. Enables **SHIFT** (criterion 3 — ablate human-judged spurious features to improve classifier generalization) and unsupervised discovery of thousands of feature circuits. The most credible bridge from head-level to your desired feature/subspace-level accounts.

**Geiger, Wu, Potts et al. — Causal Abstraction / DAS / "Finding Alignments Between Interpretable Causal Variables" (JMLR/PMLR).** The theoretical backbone: **interchange intervention accuracy (IIA)** as a graded causal-abstraction metric; **Distributed Alignment Search (DAS)** finds high-level variables aligned to *disjoint subspaces* (not subsets) via learned orthogonal rotations — precisely the "subspace they all write to" formalism. **MDAS/RAVEL (Huang, Wu, Potts, Geva, Geiger, ACL 2024)** extends this to disentangle co-located attributes in **Llama2-7B** (state-of-the-art on RAVEL). MIB (below) later finds **supervised DAS is the best causal-variable-localization method, and SAE features are no better than raw neurons** — an important, benchmark-relevant verdict.

**OpenAI — Gao, Rajaram, Coxon, Govande, Baker, Mossing, "Weight-sparse transformers have interpretable circuits" (arXiv 2511.13653, Nov 2025).** Trains GPT-2-style models with L0 weight sparsity (~1/1000 weights nonzero) so circuits are **fully reverse-engineered at the lowest level**; sparse circuits ~16× smaller than dense at equal loss. Caveat: these are *trained-to-be-interpretable* models, not real dense models; preliminary results extend to explaining dense models. A skeptical follow-up ("Weight-sparse circuits may be interpretable yet unfaithful," LessWrong 2026) is cited in the paper. Relevant to your benchmark as the "gold-standard-but-only-because-we-built-it-that-way" endpoint.

### Criterion 3 — CLEAN / SURGICAL ABLATION (target-only removal) — and why it mostly fails

**Best positive evidence:** SHIFT (Marks et al., above) — ablating human-judged task-irrelevant features removes a spurious signal without destroying the target task, verified by generalization improvement. RAVEL/MDAS — disentangles and independently edits co-located attributes (e.g., a city's country vs. continent).

**The negative literature you should treat as ground truth:**
- **McGrath et al. — "The Hydra Effect" (DeepMind, arXiv 2307.15771).** Chinchilla-7B (no dropout). Ablating one attention layer causes another to compensate, restoring ~70% of the logit drop at middle layers; late MLPs downregulate the max-likelihood token. Direct effect ≠ total effect. The definitive demonstration that clean single-component ablation is confounded by self-repair.
- **Miller, Chughtai, Saunders — "Transformer Circuit Faithfulness Metrics are not Robust" (COLM 2024, arXiv 2407.08734).** Faithfulness scores are highly sensitive to ablation methodology (zero vs mean vs resample; dataset size — the IOI 87% figure reproduces with ~4 prompts). "The ablation determines the task." Essential caveat for any replacement-scoring benchmark.
- **Makelov, Lange, Geiger, Nanda — "Is This the Subspace You Are Looking For?" (ICLR 2024, arXiv 2311.17030).** Subspace activation patching can produce the *right end-to-end effect via a dormant parallel pathway that is causally disconnected* from the output — an "interpretability illusion." Demonstrated on IOI and factual recall; linked to rank-1 fact editing. Directly warns against your "subspace they all write to" object being an artifact.
- **"Circuit Component Reuse Across Tasks" (arXiv 2310.08744)** shows the same heads serve different (even opposite-sign) roles across tasks (GPT-2-medium negative mover head demotes subject not IO), reinforcing that head-level attribution is too coarse and that "the shared thing" must be conditional.

### Your specific object: the shared, conditional, subspace-level mechanism

Works that isolate the mechanism *below* the head level:
- **McDougall et al. — "Copy Suppression" (BlackboxNLP 2024, arXiv 2310.04625).** GPT-2-small head **L10H7**; "at least 76.9% of the role of attention head L10H7 on GPT-2 Small's training distribution is copy suppression," explained via a **rank-1-ish QK/OV copy-suppression mechanism** (weights-based). Unifies "negative name mover" (IOI) and "anti-induction" (Olsson) as one conditional mechanism (copy suppression also "explains 39% of the behavior" of self-repair in the narrow IOI task) — a model example of "the shared thing across heads that also do other stuff."
- **Gould, Ong, Ogden, Conmy — "Successor Heads" (ICLR 2024, arXiv 2312.09230).** Attention heads (31M–12B params: GPT-2, Pythia, Llama-2) that increment ordinals via a shared **"mod-10" feature subspace**; vector arithmetic on these features edits head behavior; they explicitly document *interpretable polysemanticity* in a Pythia successor head (the head does succession AND other things — exactly your case).
- **Todd et al. — "Function Vectors" (ICLR 2024) / Hendel et al. task vectors**, and the 2026 critique **"Function-Vector Heads Are Two Populations: Writers and Cancellers" (arXiv 2606.07560)**, which shows the FV head set splits by *sign* into writers vs cancellers, invisible to magnitude-only ranking — a redundancy/conditional-interaction result across Pythia scales.
- **Anthropic CLT attribution graphs** (above) are the most ambitious feature/subspace-level, cross-layer account in a frontier model.

### Benchmarks / ground-truth infrastructure (for your benchmark design)
- **MIB: Mechanistic Interpretability Benchmark (Mueller et al., ICML 2025, arXiv 2504.13151)** — two tracks (circuit localization; causal-variable localization) over 4 tasks, 5 models (IOI, MCQA, arithmetic, ARC; RAVEL for causal variables). Verdicts: attribution + mask-optimization best for circuits; **supervised DAS best for causal variables; SAEs ≈ neurons**. The BlackboxNLP 2025 Shared Task (arXiv 2511.18409) extends it. The closest existing analog to your "verifiable component replacement" scoring.
- **ACDC (Conmy et al., NeurIPS 2023)** and **EAP / EAP-IG (Syed et al.; Hanna et al.)** — automated circuit discovery + faithfulness eval; EAP faster but linear-approximation faithfulness concerns; "Position-aware Automatic Circuit Discovery" (arXiv 2502.04577) extends to Llama-3-8B.
- **RAVEL (Huang et al., ACL 2024), CausalGym, InterpBench, Tracr** — ground-truth or semi-synthetic testbeds (Tracr/InterpBench are compiled/toy, flagged).

## Recommendations

1. **Anchor your benchmark's "replacement" gold standard on the entity-tracking (Prakash) and binding-subspace (Feng & Steinhardt) results**, not IOI. They are the cleanest real-model cases where a *shared, subspace-level* mechanism is both extracted and shown functionally equivalent across models (0.88–0.97 faithfulness on transfer). IOI should be your "known-fragile" reference, included precisely to test whether a scoring method is fooled by backup/redundant heads.
2. **Adopt IIA (interchange intervention accuracy) as a primary replacement-verification metric, alongside faithfulness/completeness/minimality — but harden both against the three known failure modes**: (a) run zero/mean/resample ablations and report the spread (Miller et al.); (b) test for dormant-pathway illusions by checking the replaced subspace is causally connected downstream, not just end-to-end effective (Makelov et al.); (c) measure self-repair by iterated/joint ablation to detect Hydra compensation (McGrath et al.). A replacement that scores well on only one ablation type should fail your benchmark.
3. **For criterion-1 (prediction) scoring, operationalize "predict OOD/margin behavior" using the greater-than over-application template and the Anthropic addition-feature-transfer paradigm**: a circuit account passes only if it predicts *both* where the mechanism fires when it shouldn't (over-generalization) and where it declines to fire when it "should." A much stronger test than in-distribution faithfulness.
4. **Treat SAE-feature circuits as necessary but not sufficient.** MIB's finding that SAE features ≈ neurons and supervised DAS wins for causal-variable localization means your benchmark should reward *subspace/DAS-style* isolation over raw SAE nodes, while using SAE/transcoder circuits (Marks; Anthropic CLT) for interpretability of the isolated subspace.
5. **Include the negative/critique papers as scored items or adversarial cases**, not just citations: Hydra, faithfulness-non-robustness, subspace-illusion, and writers-vs-cancellers each define a specific way a "verified replacement" can be secretly wrong. A benchmark that can't detect these is measuring the wrong thing.

**Thresholds that would change these recommendations:** if a method demonstrates faithfulness *stable across all three ablation types* (Δ < a few %) AND survives a dormant-pathway causal-connectivity check AND predicts ≥1 OOD/over-application behavior confirmed by intervention, promote it to gold-standard status. If future work (e.g., scaled weight-sparse or CLT circuits) achieves fully reverse-engineered, self-repair-robust replacement on a *dense frontier model*, retire IOI-style head-level scoring entirely.

## Caveats
- **Toy-model flags:** Tracr, InterpBench, docstring circuit (2-layer attention-only), modular-arithmetic/grokking, and the *training* of OpenAI weight-sparse models are not "real larger models" in your sense — included only for ground-truth calibration.
- **Prediction claims are qualitative.** Neither greater-than nor Anthropic's addition study provides a formal held-out predictive-accuracy number; they show same-circuit activation on new contexts + intervention confirmation. Anthropic explicitly reports attribution graphs give satisfying insight on ~a quarter of prompts and that even successful case studies "only capture a small fraction of the mechanisms of the model."
- **"Functional equivalence" faithfulness numbers are ablation-dependent** (Miller et al.) and can be inflated by tiny mean-ablation datasets; cross-model transfer numbers (Prakash 0.88–0.97) should be read with the specific ablation methodology in mind. Note also a minor discrepancy in the greater-than sequence-recovery figure across secondary summaries (~82–90%); consult the primary paper (arXiv 2305.00586) for the exact table.
- **Subspace-level results risk illusion** (Makelov et al.); DAS/Boundless DAS/MDAS high IIA does not by itself rule out non-unique or non-faithful abstractions — see "The Non-Linear Representation Dilemma" (arXiv 2507.08802), which shows >80% IIA achievable with non-linear maps even in *random* models, and "Causality is Key for Interpretability Claims to Generalise" (arXiv 2602.16698).
- **Coverage gaps in this report:** dedicated searches on CausalGym (Arora et al.) details and the "circuits in superposition" theoretical line were cut short by budget; both are relevant follow-ups. Several 2026 arXiv IDs (e.g., 2606.xxxxx) are recent preprints — verify venue/peer-review status before citing as settled.

### Quick reference (author / arXiv or venue)
- IOI — Wang et al., arXiv 2211.00593 (ICLR 2023)
- Greater-than — Hanna, Liu, Variengien, arXiv 2305.00586 (NeurIPS 2023)
- Copy Suppression — McDougall et al., arXiv 2310.04625 (BlackboxNLP 2024)
- Successor Heads — Gould et al., arXiv 2312.09230 (ICLR 2024)
- Circuit Component Reuse — Merullo et al., arXiv 2310.08744
- Function Vectors — Todd et al., arXiv 2310.15213 (ICLR 2024); Writers/Cancellers — arXiv 2606.07560
- Entity binding — Feng & Steinhardt (ICLR 2024); Representational Analysis of Binding — arXiv 2409.05448; Rebinding circuit — arXiv 2606.08644
- Fine-Tuning Enhances Existing Mechanisms (entity tracking) — Prakash et al., arXiv 2402.14811 (ICLR 2024)
- DAS / Causal Abstraction — Geiger et al. (PMLR/JMLR); Boundless DAS (Alpaca) — Wu et al., arXiv 2305.08809 (NeurIPS 2023); RAVEL/MDAS — Huang et al., ACL 2024
- Sparse Feature Circuits / SHIFT — Marks et al., arXiv 2403.19647 (ICLR 2025)
- Edge Pruning — Bhaskar et al., arXiv 2406.16778 (NeurIPS 2024)
- ACDC — Conmy et al. (NeurIPS 2023); EAP — Syed et al., arXiv 2310.10348
- Hydra Effect — McGrath et al., arXiv 2307.15771
- Faithfulness metrics not robust — Miller et al., arXiv 2407.08734 (COLM 2024)
- Subspace illusion — Makelov et al., arXiv 2311.17030 (ICLR 2024)
- Weight-sparse transformers — Gao et al. (OpenAI), arXiv 2511.13653
- Circuit Tracing / Biology of an LLM — Anthropic, transformer-circuits.pub 2025
- MIB — Mueller et al., arXiv 2504.13151 (ICML 2025)

## Bilin18 application update (2026-08-29)

### Short answer

Yes.  The highest-value additional entry point is a **typed sparse-transcoder
graph with causal edge masks**.  This combines three ideas in this literature:

1. sparse feature circuits / cross-layer transcoders supply candidate state variables;
2. edge pruning supplies an explicit pressure for a small wiring diagram;
3. DAS-style interchange tests decide whether the candidate variables and edges are
   the variables and edges the downstream model actually uses.

The important qualification is that this should not be a generic activation SAE.
Bilin18 gives us more structure than an ordinary transformer: every bilinear MLP has
an exact product vector and an exact linear `Down` write, while attention and residual
updates can be retained as typed operators.  The graph should therefore be trained
around these physical interfaces, with reconstruction anchors, rather than being
allowed to replace an arbitrary hidden activation by an unconstrained predictor.

This is not wholly speculative.  Earlier bilin18 experiments already established:

- a hard-top-k weight-action dictionary for MLP1 `Down` recovered `0.870`, `0.938`,
  and `0.951` of its causal CE contribution at 8, 32, and 64 active atoms;
- joint reconstruction-anchored training across MLP0 `Down` and MLP1 `Left`
  preserved `0.945` CE recovery while reducing measured wiring in-degree from 291
  to 70, a 76% reduction;
- the resulting weights-only coupling predicted the sign pattern of downstream
  atom responses above a shuffled null, but weakly: correlation `0.217` versus
  `-0.008` for the null;
- the missing terms were localized: the old graph omitted attention's cross-position
  mixing, RMSNorm's input-dependent scalar, MLP1's `Right` branch, and the bilinear
  product joining `Left` and `Right`.

Thus the literature suggests a concrete repair to an already-positive result, not a
new blind search.

### Entry point 1: typed sparse-transcoder graph (highest expected return)

The first graph should span the physical interface

$$
g_0
\xrightarrow{\mathrm{Down}_0}
x_1
\xrightarrow{\mathrm{RMSNorm}+\mathrm{Attn}_1+\mathrm{residual}}
\bigl(\ell_1,r_1\bigr)
\xrightarrow{\odot}
g_1
\xrightarrow{\mathrm{Down}_1}
w_1.
$$

Here:

- $g_0$ is MLP0's vector of exact scalar products, one per bilinear channel;
- $x_1$ is the residual-stream state read by block 1;
- $\ell_1$ and $r_1$ are MLP1's `Left` and `Right` linear reads;
- $g_1=\ell_1\odot r_1$ is the exact coordinatewise bilinear product;
- $w_1$ is MLP1's residual-stream write.

Sparse dictionaries define candidate nodes at the write/read interfaces.  The exact
RMSNorm, residual, attention, and product operators remain explicit typed nodes.
Learned gates then select a small set of dictionary nodes and cross-node edges.
Training should minimize a combination such as

$$
\mathcal L =
\mathcal L_{\mathrm{teacher\ KL}}
+\alpha\mathcal L_{\mathrm{physical\ response}}
+\beta\mathcal L_{\mathrm{weight/action\ reconstruction}}
+\lambda_n\lVert m_{\mathrm{node}}\rVert_0
+\lambda_e\lVert m_{\mathrm{edge}}\rVert_0.
$$

The reconstruction term is essential: earlier CE-only joint training preserved some
loss while making the fitted weights physically unrelated to the original model.
The anchor prevents that escape.  The teacher-KL term orders equally reconstructive
graphs by what matters to predictions.  The node and edge penalties price an
executable graph instead of merely pricing matrix rank.

The cheapest decisive pilot is MLP0-to-MLP1, with the already-composed C512 MLP0 and
shared-HOSVD copy gate as downstream checks.  Relative to the previous left-only
graph, a useful pilot should simultaneously:

- retain at least about `0.94` of MLP1's held-out causal CE contribution;
- retain the physical response anchor rather than drifting under CE training;
- improve held-out edge-response prediction materially above correlation `0.217`;
- preserve the validated L8 copy state and copy-edge behavior;
- keep roughly the existing 70--80% edge reduction.

If it succeeds, extend the identical typed construction to MLP2.  If it cannot beat
the `0.217` response correlation after the omitted exact operators are included, the
sparse wiring is probably a convenient reparameterization rather than the model's
causal graph, and this branch should be pruned.

### Entry point 2: supervised DAS on downstream-defined state variables

DAS is not new to this repository: class and topic interchange experiments already
showed that a subspace can transport a causal variable even when single-direction
steering is weak.  The new use is to apply it to a variable whose downstream meaning
is now exact.

For the copy circuit, the shared-HOSVD result defines

$$
z_{\mathrm{copy}} = V_{256}^{\top}x^{(8)},
$$

the 256-dimensional part of the layer-8 stream read by the compressed H3/H4 gate.
C512 preserves this state with held-out $R^2=0.9955$.  Instead of asking only which
upstream approximation reconstructs $z_{\mathrm{copy}}$, learn the smallest rotated
subspace whose value can be swapped between prompts and whose swap predicts the
native copy scalar, MLP2 response, and final logits.  This is interchange intervention
accuracy applied to a known consumer interface.

The payoff is a causal-state dimension rather than a variance rank.  It may reveal
that only a much smaller part of the 256-dimensional HOSVD interface is independently
manipulable.  Required controls are a same-rank random rotation, wrong-source swaps,
and a downstream connectivity check; high end-to-end interchange accuracy alone can
be produced through a dormant parallel pathway.

### Entry point 3: edge pruning over exact tensor terms

Ordinary edge pruning learns masks over transformer component edges.  Bilin18 permits
a finer and more meaningful mask: each MLP already has exact scalar product nodes and
exact additive `Down` writes.  Put gates over:

- groups of bilinear product coordinates;
- residual-stream writer-to-reader routes;
- attention source-to-destination edges such as the exact L8 copy edge;
- the low-rank factors already admitted for C512 and the shared-HOSVD gate.

Optimize a stratified teacher-KL objective on natural text plus the existing circuit
slices, with explicit price and complement tests.  This could discover which exact
monomials and routes deserve to survive before we try to name them.  It is a better
fit than head-level edge pruning because it exploits the known polynomial graph.

This ranks below the typed-transcoder pilot because an unconstrained mask can produce
a task-specific sparse subnetwork without yielding a reusable basis.  It becomes
more valuable after the dictionary/interface nodes are fixed.

### Entry point 4: algorithmic-variable seeds

The successor-head, binding-ID, and greater-than studies show that a narrow behavior
can expose a shared variable that cuts across polysemantic components.  Bilin18
already has well-supported copy and successor circuits, so rediscovering those heads
has low information gain.  A genuinely new seed would be entity binding or another
task with a known counterfactual variable: entity identity, binding slot, numerical
residue, or bracket depth.

The purpose would not be to claim that a task circuit explains the whole model.  It
would provide a clean variable on which to calibrate the typed graph and DAS machinery:
can the same learned nodes predict, interchange, extract, and selectively remove the
variable?  Binding is the best candidate because it explicitly tests a shared
continuous code rather than a single token class.

### Entry point 5: weight-sparse teacher distillation (high upside, high cost)

A more radical route is to initialize a second bilin18 from the dense weights and
distill it under global hard-concrete/L0 masks plus teacher KL, while keeping the same
tensor architecture.  If a highly sparse student matches the teacher on held-out and
OOD distributions, its graph could serve as a proposed executable explanation.

This is attractive because weight-sparse transformers demonstrate that very sparse
models can contain legible circuits.  It is not the first experiment to run: a sparse
student can implement the same function by a different internal mechanism, so it
does not automatically explain the dense teacher.  Cross-model interchange and
weight/action anchors would be required to establish correspondence.

### What should not be treated as a new decomposition entry point

- **Plain SAE atoms:** already tested.  They are efficient reconstructors, but most
  atoms are seed-unstable and atom monosemanticity is nearly orthogonal to causal
  importance.  Use their span as a proposal, not the atoms as ground truth.
- **More successor/copy-head localization:** these behaviors already have localized
  circuits.  The open problem is composing their variables with upstream MLPs.
- **Hydra, ablation robustness, and dormant-pathway checks:** these are essential
  validators, not decomposition algorithms.
- **CE-only sparse fitting:** it can preserve predictions while drifting away from
  the physical model.  That may be useful extraction, but it is not weight-faithful
  reverse engineering.

### Recommended order

1. typed sparse-transcoder graph across MLP0 $\rightarrow$ MLP1, including both
   bilinear branches and exact intervening operators;
2. DAS/interchange compression of the validated copy state and the MLP1/MLP2 state;
3. edge-mask optimization over the resulting fixed typed nodes;
4. one binding-variable seed to test whether the machinery transfers beyond copy;
5. only then, a global weight-sparse distilled twin.

This ordering makes each stage falsifiable and composable.  The first three can share
one intervention substrate and directly attack the current MLP0/MLP1/MLP2 bottleneck;
the last two are broader bets.
