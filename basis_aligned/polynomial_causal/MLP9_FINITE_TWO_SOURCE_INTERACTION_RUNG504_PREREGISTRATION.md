# Rung504 preregistration: finite two-source interactions in the attention8-driven MLP9 response

Registered 2026-09-02 19:48 UTC, after scoring rung503 and before computing any rung504 pair-removal outcome.

## Question and claim boundary

Rung503 physically removed each of 18 explicit upstream sources from the BF16 residual entering MLP9. The instrument
passed, but no singleton simultaneously supported the MLP9 response and the copy task across both backgrounds and
document halves. The null was cancellation-heavy: several sources had large negative effects, while M8 removed about
25--27% of the MLP9 response but had negative or unstable first-order copy-loss effect.

Rung504 asks whether **two sources jointly form a stable finite interaction even though neither source passed alone**.
It removes every unordered pair among the same 18 sources, recomputes the real RMS normalization and BF16 MLP9, and
then recomputes the entire suffix through layers 10--17. Pair selection uses actual finite cross-entropy changes, not
a loss gradient. This is the finite-pair route licensed by rung503's A-true/B-false branch.

A pass identifies a finite two-source interaction candidate at the MLP9 boundary. It does not identify the two whole
upstream modules as a circuit, establish portability, or claim compression. A separately registered held-out edit or
removal must still test the frozen candidate as an executable circuit.

## Frozen parents, data, and source vocabulary

Pin rung503's preregistration, source, result, and sufficient-statistics bundle. Require its literal A=true, B=false,
C/D/E=false, strong-null verdict and `finite_pair_removal_screen_or_float32_control` route. Rung503's measured source
effects may be used only as registered singleton controls; rung502b's selected pair names and their ordering are not
loaded into selection.

Use the same hash-pinned checkpoint, 1,000 FineWeb rows, nearest-predecessor copy mask, score and payload scales,
`L5H5 score -> L8H4` action, and early-present/early-absent backgrounds. Discovery uses documents `0:248`, with fixed
halves `0:124` and `124:248`. Confirmation uses `248:500`, with fixed halves `248:374` and `374:500`. Documents
`500:1000` and the 30 validation circuit tags remain closed.

The source vocabulary is unchanged:

`A0,A1,A2,A3,A4,A5,A6,A7,A9,M0,M1,M2,M3,M4,M5,M6,M7,M8`.

`A8` is the score-changing carrier and `E` is the BF16 skip/rounding complement, so neither may be selected as a
partner. Test all `18 choose 2 = 153` unordered pairs. There are no self-pairs, source subdivisions, top-k ranking,
norm ranking, or outcome-dependent additions.

## Exact local and suffix computations

For action `a` in `{recipient_absent, score_donor, payload_donor}`, background `b`, and source set `S`, use the exact
captured BF16 pre-MLP9 residual `x[b,a]` and explicit float32 source contributions `r_s[b,a]`:

`x_without_S = BF16(float32(x) - sum_(s in S) r_s)`

`W_without_S = MLP9(RMSNorm(x_without_S))`.

Install `W_without_S` in place of the native MLP9 write and recompute attention/MLP layers 10--17 and the output
logits. Earlier layers, the stored block-0 value input used by later attention, and the residual entering MLP9 remain
the action trajectory's real values. Every edited BF16 input must differ from its unedited input. An unchanged MLP9
write or suffix output is an allowed scientific zero.

For the score action, define the complete local response and the response after removing `S`:

`Delta = W_absent - W_score`

`Delta_without_S = W_absent_without_S - W_score_without_S`

`J_S = Delta - Delta_without_S`.

For pair `{s,t}`, define the finite mixed local interaction

`K_st = J_{s,t} - J_s - J_t`.

This inclusion--exclusion difference is exactly zero when the two removals affect the score response additively. It
contains the interaction created by RMS normalization and MLP9 under this finite edit; it is not a Hessian estimate.

On copy positions, let `d = CE_absent - CE_score`, so positive `d` means the score action repairs copy loss. After a
source removal, define `d_without_S` from the recomputed suffix and

`C_S = d - d_without_S`,  `Q_st = C_{s,t} - C_s - C_t`.

`C_S` is the actual portion of the score action's copy-loss benefit removed by deleting `S`; `Q_st` is its finite
two-source interaction. Define the same quantities with `payload_donor` replacing `score_donor`. No gradient enters
selection or confirmation.

## Frozen pair-selection rule

For each pair and each early background, pool the two discovery halves only after also storing every half separately.
Retain the pair if all of the following hold independently in both backgrounds:

1. `J_st` has cosine at least `.75` with `Delta`, positive-scale residual at most `.70`, and response fraction
   `<J_st,Delta>/||Delta||^2` between `.20` and `1.50`.
2. The mixed local interaction fraction `<K_st,Delta>/||Delta||^2` is at least `.10`.
3. The finite copy-loss fractions `sum(C_st)/sum(d)` and `sum(Q_st)/sum(d)` are respectively in `[.20,1.50]` and at
   least `.10`.
4. The raw local-response, mixed-local, finite-copy, and mixed-finite-copy numerators are positive in both discovery
   halves.
5. For both the complete pair and its mixed term, the absolute payload projection is at most half the corresponding
   score projection in local write space and finite copy-loss space.

Retain every passing pair. Do not choose top-k. B requires the complete set to be nonempty and contain at most 10
pairs; more than 10 is a diffuse description and fails rather than being truncated.

The `.20` complete-effect and `.10` mixed-effect floors are frozen before pair outcomes. They require a selected pair
to explain a material part of the MLP9 response and the actual copy benefit, rather than turning 153 comparisons into
a hunt for tiny nonzero interactions.

## Frozen predictions

### A — exact finite instrument and parent reproduction

All hashes, rows, masks, source names, 153 pairs, intervals, actions, edits, and calls match. Native replays are exact;
the 20-source raw recurrence remains within rung503's BF16 bound; all edited inputs are live; the unedited parent
reproduces rung503's score/payload measurements within absolute `.01`; and all 18 remeasured singleton local-response
and payload sufficient statistics reproduce rung503 within relative `1e-6` or absolute `1e-8`. Rung503's gradient
statistics are not remeasured or used.

Discovery costs exactly 496 full prefix/model forwards and 63,612 batch-sized MLP9-plus-suffix evaluations:
`62 batches * 2 backgrounds * 3 action states * (18 singletons + 153 pairs)`. It performs no backwards. If B opens
confirmation, the ordinary-arm totals are 1,000 full forwards and 128,250 MLP9-plus-suffix evaluations. If the
selection contains `k` pairs, the 16 shifted-position controls add exactly `63 * 2 * 3 * 16 * k = 6,048k`
MLP9-plus-suffix evaluations, so the conditional total is `128,250 + 6,048k`. Every call count is asserted literally.
Zero deployed parameters are added or saved.

### B — a compact finite pair set is selected

The literal rule returns a nonempty complete set of at most 10 pairs. All 153 pairs, all component measurements, and
all failure reasons are stored.

### C — exact pair identity and finite effects confirm

Applying the unchanged rule to confirmation documents returns exactly the same complete pair set. Every selected pair
passes every local and finite-copy bar independently in both confirmation halves and backgrounds. No favorable
intersection, union, pair deletion, or threshold change is allowed.

### D — selected interactions separate known downstream circuits

Only if B and C hold, use the 32 fixed discovery circuit tags on confirmation documents. For each selected pair,
compute the finite member-minus-control fingerprint from its recomputed suffix losses for `C_st`, `Q_st`, and their
payload counterparts. (`J` and `K` name the corresponding local MLP9 write-space quantities; `C` and `Q` name the
finite loss-space quantities.) All tags must have support in both halves. In every half/background, the score pair's fingerprint
must have cosine at least `.75` with the complete score-response fingerprint, positive-scale residual at most `.70`,
and norm at least `.25` of complete. Its cosine must exceed the payload fingerprint by `.20` and the 95th percentile
of 16 fixed position-roll controls by `.10`. The mixed `Q_st` fingerprint must have cosine at least `.65` with the
complete fingerprint and norm at least `.10` of complete. Every selected pair must pass; no best-pair reporting can
turn D true.

### E — interpretation

E is true only if A--D hold. Each frozen pair is then a **finite two-source interaction candidate at the MLP9
boundary**. The next rung must test the same candidate on documents `500:1000` with an executable suffix intervention,
held-out copy behavior, unrelated-circuit preservation, and composition with the sign-gauged upstream score action.
Rung504 alone cannot call the pair a circuit or claim compression.

## Nulls and routing

- A false repairs only the instrument.
- A true/B false means no material two-source interaction survives the direct local-plus-finite-copy rule. Run the
  already named float32 explanatory control or change the downstream observation; do not search triples or lower bars.
- A/B true/C false means pair identity is document-dependent. Do not keep a favorable intersection; use the float32
  control or change observation.
- A--C true/D false means a stable local pair exists but the current downstream circuit measurements do not identify
  its use. Preserve the local fact and change the downstream observation.
- A--D true licenses only the separately registered held-out executable test in E.

Rank reduction, quantization, reconstruction error, gradients, another normalized-source allocation, and post-outcome
pair ranking cannot pass any clause.
