# Preregistration — attention0 cross-head sparse Q/K token vocabulary (rung 426)

Date: 2026-09-01 20:13 UTC  
Claim level: held-out sparse-function and physical-generator screen; not adoption

## Question and relation to prior work

Rung 418 compared pairs of complete 128-dimensional folded token-function subspaces. It did not train a global
sparse dictionary. A pairwise shared-block null does not exclude a collection of atoms reused by different subsets
of heads.

The older layer-0 QK MDL program is the required baseline. It established that 18 independent dictionaries—one for
each `(head, score branch)` entry, with rows `[q | k]`—Pareto-dominated SVD. It also rejected one global hard token
partition and rejected regrouping `[q1 | q2]` versus `[k1 | k2]` within each head. Its own open-work list explicitly
left a cross-head shared-atom dictionary unexecuted. This rung tests that different object.

## Exact object

For every real token `t`, head `h`, and score branch `b`, layer 0 gives exact unit-root-mean-square factors

`q_b(t,h), k_b(t,h) in R^128`.

One entry row is

`x_(h,b)(t) = [q_b(t,h) | k_b(t,h)] in R^256`.

Concatenating all nine heads and both branches gives

`X(t) = concat_(h,b) x_(h,b)(t) in R^4608`.

The fold must reproduce native layer-0 scores to maximum absolute error at most `1e-10`. Token IDs with
`t mod 5 != 4` are FIT (40,206 types); `t mod 5 = 4` are SELECT (10,051 types). Atoms and encoders are optimized
only on FIT. Frozen encoders then produce a code table for all 50,257 tokens; the deployed artifact stores the code
table and decoder, not the encoder or native layer-0 Q/K maps.

Every decoded `[q | k]` slice is root-mean-square normalized separately on its query and key halves before scores
are computed. At relative offset `delta`, reconstructed branch scores are

`s_hat_(h,b)(u,v,delta) = q_hat_b(u,h)^T R_delta k_hat_b(v,h) / 128`,

and the complete attention weight is `P_hat_h = s_hat_(h,1) s_hat_(h,2)`.

## Frozen arms

All learned arms use signed top-k codes, 512 atoms, the same FIT token types, optimizer family, step count, and
per-coordinate squared-error objective.

- **I72 (independent baseline):** 18 independent `512 x 256` decoders and one k=4 code per entry and token. Total
  stored nonzeros per token: `18 x 4 = 72`.
- **G54 (saving candidate):** one `512 x 4608` decoder and one k=54 code per token. A coefficient can jointly
  generate slices for several heads/branches. It stores 25% fewer code entries than I72.
- **G72 (equal-code diagnostic):** the same global-family training and decoder, decoded with the leading 72 signed
  encoder coefficients. It has exactly the same decoder size and stored nonzero count as I72. It is diagnostic and
  cannot be selected in place of G54 after the outcome.
- **P54 (token-alignment null):** independently permute token rows inside every entry, separately within FIT and
  SELECT, then train/evaluate the same global k54 architecture. Every entry marginal and all dimensions are
  preserved; only the fact that entry slices belong to the same token is destroyed. P54 is a structural null, not
  a deployable language-model arm.
- **D54 (atom-coupling null):** keep G54's stored codes, bias, and per-entry decoder-slice marginals, but independently
  permute the 512 atom identities of every entry slice before reconstruction. It has identical stored shape and
  destroys which head/branch slices share an atom.

## Literal stored price

Decoders, biases, and coefficients are physically stored as FP16 (two bytes/value); sparse indices are physically
stored as uint16 (two bytes/index). No entropy-coded or theoretical index bill is used.

`decoder bytes = 512 x 4608 x 2 = 4,718,592`  
`bias bytes = 4608 x 2 = 9,216`

Therefore:

- I72: `4,718,592 + 9,216 + 50,257 x 72 x 4 = 19,201,824 bytes`;
- G72: exactly `19,201,824 bytes`;
- G54: `4,718,592 + 9,216 + 50,257 x 54 x 4 = 15,583,320 bytes`.

G54 is 3,618,504 bytes, or 18.8446%, smaller than I72. Native layer-0 Q/K/Q2/K2 contain
`4 x 1,152 x 1,152 = 5,308,416` values and are not counted as retained by a passing physical candidate. Native
layer-0 V/O and every later module remain.

## Binding measurements

1. SELECT factor fraction of variance unexplained (FVU), balanced equally over the 18 entries.
2. SELECT random token-pair score error at rotary offsets `{1,2,4,8,16,32,64,128}`: branch relative squared
   error and complete-product relative squared error.
3. On the frozen 96-document SELECT role: full attention0-write relative squared error, immediate MLP0 and all
   attention1 Q/K/Q2/K2/value consumer relative squared errors, and document-mean cross-entropy damage. Damage is
   cross-entropy added above the real model; lower is better.
4. Global atom head-service participation. Decoder energy is summed over both branches and q/k coordinates within
   each head, normalized over nine heads, and converted to participation rank `1 / sum_h p_h^2`.
5. A no-native-QK execution check: with the sparse score patch active, zero all four native layer-0 Q/K maps and
   require candidate logits to reproduce. This demonstrates that the stored sparse generator, rather than a
   projection of native scores, determines the result.

## Frozen predictions

**A — exact and physical instrument.** Fold error `<=1e-10`; FIT/SELECT cover exactly 50,257 token IDs with zero
overlap; all learned losses decrease by at least 20%; stored tensor dtypes/shapes and the two literal byte bills are
exact; and patched logits before versus after zeroing native layer-0 Q/K have relative squared error `<=1e-12`.

**B — same-token sparse sharing is real.** G54 SELECT factor FVU is at least 10% lower than P54 SELECT FVU; median
global-atom head participation rank is at least 3; and at least 25% of global atoms have head participation rank at
least 3. P54 is scored only against G54 on the permuted SELECT object.

**C — sharing survives composition at equal or lower price.** G72 complete-pattern relative squared error is no
more than `1.10 x` I72 and its CE damage no more than I72 `+0.002 nat`. G54 complete-pattern error is no more than
`1.35 x` I72, its full attention0-write relative squared error no more than `1.50 x` I72, and its CE damage no more
than I72 `+0.005 nat`.

**D — the learned cross-head atom identity matters.** D54 complete-pattern error is at least 25% worse than G54;
D54 full-write error is at least 25% worse than G54; and D54 CE damage is at least `0.01 nat` worse than G54.

## Strong null and routing

The strong null fires if A fails; if G54 has less than 2% SELECT-FVU advantage over P54; if G72 complete-pattern
error is at least `1.25 x` I72 or its CE damage is more than `0.01 nat` worse; if G54 full-write error is at least
`2 x` I72; or if D54 is within 2% of G54 on complete-pattern error or full-write error.

- A/B/C/D pass without the null: the global sparse token vocabulary is identified as a lower-byte physical Q/K
  generator. Next compare a product/downstream-finetuned sparse generator directly with the rung-424 continuous
  physical generator and ordinary rank at matched total price.
- B passes but C fails: shared atoms exist descriptively but are not an efficient generator; keep them for
  interpretation only.
- C passes but B fails: any gain is generic global capacity, not same-token cross-head vocabulary; do not attach
  semantic sharing language.
- The strong null: close global sparse Q/K atoms at this budget and continue with the direct continuous composite
  generator. Do not tune atom count, k, thresholds, or permutations after reading SELECT.

Even a full pass is not whole-model adoption. It still requires fresh documents, shifted text, 62-behavior
certificates, composition with the current frontier, and signed interventions.
