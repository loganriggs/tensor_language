# Preregistration: MLP1 same-context split-probe response-bundle discriminator

Date: 2026-08-28

Status: the outcome-blind CPU row/probe plan and analysis kernel are frozen. This file
does not authorize a GPU measurement. A create-only collector must independently bind
the source, rank-640 program, parent geometry, rows, and both probe halves before any
response is sampled.

## Motivation

The completed 96-row tangent pilot rejected a compressed shared-linear early-MLP
interface, but its document-split difference conflates context variation with
Monte-Carlo error from only 16 Fisher probes. Almost every per-context response also
saturated rank 16. The positive localization was that MLP1 supplied 94.9--95.4% of
cut-3 response energy under equal-RMS site directions. The cheapest next experiment is
therefore MLP1-only and repeats independent probes on the same contexts.

## Frozen object

Select 16 distinct documents without looking at model outcomes by the smallest
SHA-256 values of `2026082804:document_id`. Within each selected document, choose the
row with smallest SHA-256 of `2026082804:document_id:row_id`. Use one context per
document, position 128 in every context, the already frozen 32-direction MLP1 bank,
and the summed categorical score over every output position 128 through 255.

These documents appeared in the parent tangent result. The code-level row choice is
outcome-independent, but the rule and seed were chosen after that parent outcome was
known. This is therefore a **conditional historical-row follow-up**, not fresh-document
confirmation. The first 12 contexts in frozen document-hash order are prospectively
the promotion cohort; the last four are diagnostic only. Cohort membership never
depends on paired-probe outcomes.

For every context collect two independent halves

\[
H_c^{(A)},H_c^{(B)}\in\mathbb{R}^{32\times32}
\]

using disjoint consecutive seeds beginning at 2026082901 and 2026083001. Every
direction is evaluated in every context. At batch size four this is 256 backward
passes, two thirds of the previous three-site response collection.

The coefficient directions are nonorthogonal. Factor the frozen MLP1 direction matrix
as

\[
D_1^\top=QR,
\]

convert the measured responses to orthonormal physical coordinates

\[
\widetilde H_c=H_cR^{-1},
\]

and compare physical frames

\[
U_{c,r}=Q V_{c,r}(\widetilde H_c)
\]

at fixed ranks 8, 16, and 24. Taking an ordinary SVD of \(H_c\) before mapping its
frame through \(D_1^\top\) is forbidden: that construction changes under a
nonorthogonal reparameterization of the same physical directions.

## Frozen decisions

1. **Probe-limited high rank.** Both halves have numerical support rank at least 24
   and \(r_{95}>16\) in at least 75% of contexts. This prunes every local tangent-state
   story of dimension at most 16 in the registered bank.
2. **Stable local low rank.** In every one of the 12 prospectively fixed promotion
   contexts, both halves have an energy-plus-gap selected rank at most 16, differ by at
   most two, have numerical support at least 16, and have same-context
   **fixed-rank-16** physical projector distance at most 0.15. A fixed-rank projector
   above either half's numerical support is unidentified and not evaluable. Checking
   only the smaller selected frame is also forbidden because unstable
   below-energy-tail directions could otherwise drive the rank-16 promotion contrast.
3. **Context-varying response bundle.** Gate 2 passes and, on exactly the same fixed
   12-document promotion cohort, the 95% document-paired bootstrap lower bound for mean
   rank-16 cross-context minus same-context physical distance is at least 0.05. The
   cohort is not selected by the same outcomes entering the contrast. The four
   diagnostic contexts cannot supply promotion heterogeneity.
4. Otherwise report **no admitted local bundle**. In particular, unstable
   same-context halves mean the prior document difference is measurement-limited; they
   do not license a context-conditioned compiler.

The bootstrap uses 1,000 repetitions and seed 20260828. The implementation returns
only ranks, spectra, distances, and scalar uncertainty summaries; raw logits,
responses, frames, and projectors remain forbidden.

## Scope and consequences

This experiment identifies response geometry only. Since

\[
H_c=D_cE_c,
\]

variation in \(H_c\) does not identify whether the encoder \(E_c\), decoder \(D_c\),
or both vary. A positive result is a response bundle, not an encoder gauge. No outcome
authorizes a finite replacement, selective removal, CE claim, or context-to-chart fit
without a separately preregistered consequence stage.
