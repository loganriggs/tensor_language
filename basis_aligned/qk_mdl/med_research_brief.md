# Research brief: where can exact-mechanistic model extraction have real impact?

*(Prompt for a web-research agent. Self-contained — no external repo needed.)*

## What we have (the capability to match against)

We train small neural networks **that are deliberately built to be exactly
decomposable** (a "foldable" architecture: attention without softmax, bilinear
MLPs). Because of that architecture, we can do something standard interpretability
cannot: **render the exact function a model learned as a human-inspectable object** —
e.g., for an image classifier, the precise pixel-space pattern each internal feature
detects, with zero approximation error — rather than an approximate saliency/heatmap.

We have validated this end-to-end on a histology benchmark (colorectal tissue tiles):
we extracted the classifier's algorithm into a small bank of renderable visual
texture filters, quantified how much of the accuracy is explicit local texture vs
genuinely deep composition, and — the useful part — used the extracted mechanism to
**diagnose and fix a real fragility**: the model leaned on stain color, so we built the
mathematically-correct color invariance and improved BOTH cross-institution accuracy
and robustness to stain shift.

### What is genuinely distinctive (and what is NOT)

- **Distinctive / unique to us:** exact mechanistic *rendering* — the exact learned
  feature as an auditable object. This turns a black box into **specific, testable
  hypotheses** ("the model discriminates cancer using exactly *this* pattern").
- **NOT distinctive:** confounder/shortcut *detection*. We tested this directly and
  standard causal ablation (occlusion) beats our approach — you don't need us for it.
- **Hard constraint:** we must **train our own foldable model**, so the setting must
  have (a) accessible training data, and (b) a task where a *small* model is
  competitive. This biases toward small-image vision and 1-D signals. It is a
  train-your-own-model research tool, not something you point at a deployed system.
- **Crucial caveat:** exactness ≠ truth. The exact feature the model uses may be a
  real biological signal or a spurious correlation. Separating the two requires
  **external validation** — chiefly, does the feature generalize across sites /
  instruments / cohorts. So the full value loop is:
  **(1) fold extracts exact candidate features → (2) cross-setting generalization
  filters true from spurious → (3) survivors become precise biomarker hypotheses for
  domain experts to validate or mechanistically investigate.**

## The question to research

**In which real-world domains, tasks, and model settings would this exact-feature
extraction loop have the most genuine impact?** It does not have to be histology or
even medicine. We want a grounded map of the field and a ranked shortlist of
best-fit opportunities.

### Criteria for a high-fit setting (rank candidates on these)

1. **The scientific question is "what did the model learn," not just "is it
   accurate."** Domains where understanding the discriminative feature has value in
   itself (biomarker discovery, mechanism, regulatory trust) — not domains where a
   black-box accuracy number is sufficient.
2. **Ground-truth features are partly unknown**, so *discovering* a real feature would
   matter — or trust/auditability is the acknowledged bottleneck.
3. **A small model is competitive**, so we can train a foldable version (small images,
   modest resolution, or 1-D signals; NOT tasks that require billion-parameter
   foundation models or gigapixel full-resolution input).
4. **Multi-site / multi-instrument / multi-cohort data exists publicly**, so
   generalization can separate true features from spurious ones.
5. **High stakes / clear real-world impact** if a true feature were found and validated.

### Specific sub-questions to answer

- Which domains have an *acknowledged, active* problem of "we have accurate models but
  don't understand what feature they use"? (medical imaging, but also where else?)
- Concrete candidate task families to assess against the 5 criteria — go beyond the
  obvious: histopathology prognostics; **retinal fundus imaging** (diabetic
  retinopathy, and predicting systemic factors); **ECG / EEG** (1-D, small, easy to
  train); dermatology; chest X-ray / CT; **cell microscopy / morphological profiling**
  (drug mechanism-of-action); **genomic/regulatory-sequence models** (motif discovery);
  materials-science and remote-sensing imaging; anything else strong.
- **Existence proofs** — find documented cases where interpreting a model *revealed a
  real, novel, expert-validated feature* (the canonical example: deep nets predicting
  systemic health signals from retinal fundus photos that clinicians didn't know were
  encoded). These prove the loop can work and indicate the highest-value domains.
- For each candidate: is a small model actually competitive with SOTA (cite numbers)?
  Is there public multi-site data (name datasets, sizes, licensing)? Is
  feature-discovery an explicitly stated goal in that community?
- Where does the field currently rely on unreliable post-hoc saliency, such that an
  *exact* feature rendering would be a real improvement?
- Honest counter-analysis: where would this approach add nothing (e.g., the feature is
  already fully known; only accuracy matters; small models are far from competitive)?

### Requested output

A ranked shortlist of **5–8 settings**, each with: the domain and the specific task;
why it fits (scored against the 5 criteria); public dataset names + sizes + licensing
+ whether multi-site; evidence that a small model is competitive (with numbers); any
existence-proof of model-driven feature discovery in that domain; and a
difficulty-vs-impact assessment. Plus a short section on the 2–3 strongest existence
proofs across all domains, and an honest list of settings to de-prioritize and why.
Prefer primary sources (papers, dataset pages, challenge leaderboards) with links.
