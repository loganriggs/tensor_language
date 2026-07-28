# Foldable models for medical ECG interpretability: extraction, honest baselines, and distilling SOTA models

*Findings consolidation, 2026-07-28. Companion to RESULTS_l0_mdl.md §13–31 and LOG §13–55.*

## Summary

We set out to apply an exactly-foldable model class (no-softmax bilinear attention + bilinear MLP,
so every layer folds to a polynomial with no gauge ambiguity we can't remove) to real ECG
classification, to extract *interpretable* accounts of what the models compute. The arc has three
phases and one honest reversal:

1. **Extraction** — train foldable ECG models, decompose them, render waveform features, validate
   physiologically and across three continents.
2. **Honest baselines** — three independent baselines each deflated a claim: the interpretable
   features are a faithful *rendering* of real diagnostic shapes but **not** a superior classifier
   or a uniquely-necessary mechanism.
3. **Distillation (path 2)** — the impactful capabilities need data we lack, so we interpret
   *released SOTA models* by distilling them into foldable students. Proven on a diagnostic model
   and a mortality biomarker, with a load-bearing lesson about which signals distillation preserves.

## 1. Extraction works, and the fold is exact

A foldable ECG model (12-lead, patched in time, ~0.4M params) reaches macro-AUC ~0.90–0.93 on
PTB-XL — competitive, off-images. Each bilinear MLP folds **exactly** (verified relative error
2.6e-7) to a symmetric third-order tensor; the "neurons" are a CP-factorization gauge, not features.
Refit minimally, the block-0 layer is behaviorally **rank ~32–64** (of 192), and in the correct
interaction basis each diagnosis has a sparse, physiological readout (leads match textbook 10/10),
with shared features explaining correlated diagnoses. The rendered features cosine-match the real
median beat of confirmed cases on **independent US and China cohorts** (complete LBBB 0.968) — a
shape-level external validation.

## 2. The honest baselines (the program's ethos)

Three baselines each deflated an overclaim:

- **Linear (§27):** a fair pooled-amplitude linear model reaches 0.745 vs the model's 0.925. The gap
  is code-type-dependent — amplitude diagnoses (bundle branch block, hypertrophy) are near-linear;
  morphology diagnoses (injury, ischemia, MI) need the nonlinearity.
- **Random-feature (§28):** in the interaction basis, random directions classify almost as well as
  the "chosen" features — the basis is *not* privileged; the model is redundant (behaviorally
  low-rank, fungible features). Readout-sparsity is not causal-sparsity: no small feature set is
  *necessary* (removing the top 10 collapses nothing).
- **Template-match (§42):** matching the aligned average diagnostic beat — **no model at all** —
  beats our single feature cross-cohort on 7 of 10 diagnoses. The model's genuine edge over the
  template is narrow: hypertrophy (+0.22, an *amplitude* signal the scale-invariant template
  discards) and first-degree AV block (+0.23, a *timing* signal a single beat drops).

**Net:** the features faithfully *render* externally-validated shapes, but a trivial baseline
captures the same shape, and the model's advantage is distributed and narrow. This is methodology,
not clinical impact — we were re-deriving known cardiology on a sub-SOTA model.

One constructive counter-result: a small **learned atomic basis** (24 reusable waveform primitives
on aligned beats, sparse readout) *is* privileged (learned 0.848 vs random-atom 0.659), compositional
(~6 atoms/diagnosis, each reused ~7×), and recovers the amplitude/timing the cosine template lost.

## 3. Path 2: interpret SOTA models by distilling them into foldable ones

The impactful capabilities (detecting what humans can't — low ejection fraction, mortality) need
outcome/echo-labeled data we don't have. **The data is the moat, not the architecture.** Path 2
sidesteps it: take a released SOTA model, distill it into a foldable student (the teacher provides
the labels, so we need none of its training data), then apply the toolkit to a *capable* model.

**Tier-1 — a diagnostic model.** Teacher: Ribeiro CODE ResNet (2.3M ECGs, 6 classes incl. rhythm).
Foldable student matches it at **0.991 agreement**, including rhythm classes (atrial fibrillation
0.989) our architecture had never handled. Mechanism extracted: rhythm recruits **attention ~7× more
than morphology** and needs the **whole strip** (atrial fibrillation is chance from one beat, 0.99
from all 20) while morphology is focal; brady/tachy reduce to heart rate, but atrial fibrillation is
a genuine multi-cue computation (0.989 vs RR-irregularity 0.836, P-absence 0.725).

**Tier-2 — an invisible biomarker.** Teacher: the ECG-age model (Lima 2021; the predicted-minus-true
age-gap predicts mortality). Validated on PTB-XL (age r=0.80). Foldable student inherits it (r=0.75).
ECG-age is **~70% novel morphology** (known measures explain only R²=0.30).

## 4. The load-bearing lesson: distillation preserves the dominant signal, not the subtle residual

Distilling the biomarker's **raw age** matched the teacher at r=0.91 but **reversed** the
mortality-relevant age-gap (pathology read *younger*) — the ~1-year mortality signal was rounding
error on a 16-year prediction. Distilling the **age-gap directly** recovered it: pathology reads +3.0
years older than normal (correct, stronger than the teacher). Generalizes: *to interpret a
biomarker's valuable signal via distillation, target that signal, not the raw output.*

## 5. The payoff decomposition

The foldable age-gap student gives a clinically coherent, decomposable account of the mortality
biomarker: premature ECG-aging is driven by **atrial fibrillation (+8.7y), complete RBBB (+8.3),
first-degree AV block (+7.5), MI and ischemia (+6.7/+6.6)** — the excess-mortality conditions — read
mostly from **novel precordial morphology** (leads aVR/V3/V1/V4), only weakly explained by heart rate
or intervals. Age-controlled contrast (matched true age ~63) isolates a ~15-year ECG-age difference
as genuine premature aging. And it is **subclinical**: within pure-normal ECGs it still tracks true
age (r=0.73), and 28% of normal-looking ECGs read ≥5 years prematurely old — the genuinely
"invisible" signal.

## 6. Honest limitations and what's data-gated

- The mortality *direction* is captured, but not validated against actual death outcomes — that needs
  outcome-linked data (the CODE mortality set, access-gated).
- Low-ejection-fraction (the other flagship invisible capability) has no released fine-tuned
  checkpoint and needs echo-EF labels we lack.
- The foldable students are sub-teacher in absolute fidelity (age r=0.75 vs 0.80) — decomposability
  costs some capability, though the dominant behavior transfers.

## 7. Contribution

A demonstrated, architecture-agnostic-in-principle method: **interpret any released clinical model by
distilling it into an exactly-decomposable foldable one, and — for a subtle biomarker — target the
valuable signal directly.** Proven end-to-end on a Tier-1 diagnostic model and a Tier-2 mortality
biomarker, with the honest boundary of when naive distillation suffices and when it must be targeted,
and with the baselines that keep the interpretability claims honest.

## Artifacts
- Feature atlas (Tier-1): https://claude.ai/code/artifact/49397032-b01d-47f6-8f76-e6033b7523b8
- Prematurely-aged ECG (Tier-2): https://claude.ai/code/artifact/339f6dfc-cc7a-4640-9e0b-0f5bfe0a4c96
