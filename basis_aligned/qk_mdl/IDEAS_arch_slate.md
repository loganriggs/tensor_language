# Architecture-idea slate (2026-08-06, local) — answering Logan's two framings

Framing 1: what does each new structure newly enable? Framing 2: what else can
be pinned, typed, bounded, or made human-legible by construction? Ideas are
ordered by (evidence behind them) x (cheapness to test at w264 fresh).

## From new structure (framing 1)

**S1. Closed-form bigram path from the remnant (enabled by E16).** The remnant
is a pure per-token linear function of the embedding. Give the readout an
explicit remnant->logits term: its composition with the embedding is a single
vocab x vocab bigram table computable in closed form from weights alone —
the model's entire token-conditional prediction becomes a printable artifact,
and the module slots only have to carry the *contextual* correction. Prediction:
small CE gain (the direct-unembedding family showed the path has value) plus a
strictly cleaner division of labor. Cost: one 264 x d_remnant matrix.

**S2. Position remnant (enabled by E16).** Same trick for position: a second
shrinking channel carrying a pure per-position function. Stream dims then come
in three types — token remnant (closed-form in token), position remnant
(closed-form in position), module slots (contextual). Every non-module
dimension of the stream is a known function of (token, position) BY
CONSTRUCTION; circuit analysis starts from two fully-solved channels.
Cheap: reuse the E16 machinery verbatim.

**S3. Slot lifetimes (enabled by slots + certified lasso zeros).** The lasso
already certifies which (reader, writer) edges are zero. If ALL readers of
writer j at blocks > k are certified zero, slot j is dead after block k —
reclaim those dims for the remnant floor or late slots. Generalizes E16's
"embedding retires on a schedule" to "every module retires when its audience
ends". Gives each slot a typed lifetime [birth, death] in the datasheet, and
converts certified sparsity into free bandwidth (which E15c says is the
binding resource).

**S4. Attention-targeted shared values (enabled by the E12aqk mechanism
answer).** Constant-width shared values failed (+0.071 at w1152) because there
is no bandwidth bottleneck to repair; the neck spectra say the squeeze lands on
ATTENTION reads (P_a half-rank, P_m near-full). So share/copy values only on
the attention path (or only in a narrowed attention sub-stream), not globally.
Test at w264: recipe + attention-only value sharing, param-matched.

**S5. Bandwidth-first recipe (enabled by E15c).** The single strongest lever of
the batch: true-small decoders + slot width 15 collapsed the partition cost to
+0.0525 vs vanilla. Compose it with E16b's floor (the two are orthogonal:
one reinvests decoder waste into slots, the other schedules the embedding out
of them). Predicted stack at w264: vanilla + ~0.03-0.05 with full slots,
per-slot norm, lasso, remnant channel. This is the leading retrain candidate
and needs the E15c readability number (running now) before enshrinement.

## Circuits-grade assets (framing 2)

**A1. Covariance-composed wiring as the standard metric.** Done (E17): +0.086
Spearman, top-10 precision 0.5 -> 0.7 over plain reader-norms, one cached
forward pass. Adopt in the light probe; re-score all frontier arms.

**A2. The model datasheet.** One machine-readable JSON per trained model:
edges (2,016 groups) with covariance-composed strengths, certified zeros from
anneal, per-slot content bases + effective ranks (census eigenvectors), slot
lifetimes (S3), naming-pass labels where recovery >= 0.75. Everything circuit
analysis needs before ever running the model. Cheap: all pieces exist; this is
a packaging job with a schema.

**A3. Fixed per-slot bases with named coordinates.** Census says many slots
carry rank 2-4 content. After training, rotate each slot to its content
covariance eigenbasis (a gauge choice — free), so "slot j, coordinate 2" is a
stable, nameable direction across analyses. Message between modules = k
numbers with persistent meaning.

**A4. Binary wiring after anneal.** Per-slot RMSNorm makes unit-strength
meaningful; after certifying zeros, quantize surviving read groups toward a
{0, unit} wiring and fine-tune. If cheap (predict +0.02-0.05 based on anneal
prices), the wiring diagram becomes literally boolean — the strongest possible
"which modules talk" statement.

**A5. Naming-pass regression harness.** The level-5 naming pass re-run on every
retrain, diffing which modules stay nameable and at what recovery — makes
interpretability a tracked metric with history, not a one-off audit.

## Registered predictions
- S1 small win (+0.00 to -0.03 CE) and exact bigram table; S2 near-free.
- S4 recovers a piece of the sv win at constant width where global sv failed.
- S5 stack lands within +0.05 of vanilla at w264 with Spearman >= 0.75 under
  the covariance-composed metric.
- A4 costs < +0.05 from an annealed checkpoint.
