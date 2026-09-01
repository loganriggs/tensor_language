# Attention0 cross-head QK shared-half preregistration

**Registered:** 2026-09-01 18:39 UTC

**Owner / rung:** Codex / 418

**Parents:** exact layer0 QK fold (`qk_mdl/tier2_folding.py`); rung417 finite-response head-service null

**Claim level:** gauge-invariant weight/folded-function screen only; no mechanism identification, compression, or adoption

## Dossier-controlled question

Attention0 has nine heads and two multiplicative query/key score branches per head. Existing work already proves:

- the normalized layer0 token factors reproduce both QK score branches exactly with rotary position encoding;
- the two branches inside a head are usually complementary and their product is a selective signed conjunction;
- one global token partition across all head-branches is worse than independent partitions;
- shared atoms across head-branches and a joint product decomposition remain open.

Do **not** repeat global token clustering or the within-head complementarity census. Test the user's distinct proposal:
one complete or partial QK branch vocabulary may be reused across several heads, while each head composes it with a
different companion branch.

## Exact object and gauge

For real tokenizer token `t`, head `h`, and score branch `b in {1,2}`, the established fold gives unit-RMS vectors

`q[t,h,b], k[t,h,b] in R^128`.

At relative offset `delta`, that branch score is exactly

`s[h,b](t_query,t_key,delta) = q[t_query,h,b]^T R_delta k[t_key,h,b] / 128`,

where `R_delta` is the model's known rotary rotation. Each `(h,b)` therefore supplies a pair of 50,257-by-128
token-function tables and a whole relative-offset score family. Rotating the private 128 coordinates while applying
the inverse compatible transformation does not change the function. Individual coordinates are not named. The
column subspaces in token-function space, principal angles between them, held-out mapped factors, and complete scores
are the reported objects.

## Frozen split and controls

- Use all 50,257 real tokenizer IDs. FIT tokens have `token_id mod 5 != 4`; SELECT tokens have `mod 5 == 4`.
- Select candidate relations on FIT only. SELECT evaluates unseen token identities.
- Evaluate exact scores at causal offsets `{-1,-2,-4,-8,-16,-32,-64,-128}` and on the frozen natural SELECT
  document rows used by rungs401–417. FINAL stays sealed.
- For each source table, a seeded independent permutation of token rows is the main null. It preserves factor norms,
  singular values, coordinate dimension, and every token marginal while destroying token-to-function alignment.
- Seeded Haar token-function subspaces of the same rank are the geometric null. An exact independent target table is
  the no-sharing ceiling and is priced explicitly.

## Computations

For each of the 18 entries `(head,branch)` and each query/key side, compute FIT QR bases for both raw unit-RMS factors
and factors after subtracting their FIT token mean. For every pair from different heads, report:

- normalized projector overlap `trace(P_i P_j)/128`;
- the number of squared principal cosines at least `.50`;
- the matched row-permutation and Haar distributions.

Build a graph whose edges require shared centered query **and** key structure. Choose the largest/highest-overlap
FIT component before looking at SELECT. For every selected directed relation source→target, fit two 128-by-128 linear
maps on FIT tokens, one for queries and one for keys. On SELECT tokens report factor `R2` and exact relative-offset
branch-score `R2`. Fit and score the identical maps with permuted source-token rows as a negative control.

For a selected relation `(source head,source branch) -> (target head,target branch)`, also test the two companion
branches and the products of both branches on natural held-out examples. A shared half with different partners should
transfer the selected branch but not the companion, and the complete head products should be less similar than the
shared branch.

## Frozen predictions

1. **A — exact and auditable.** Folded scores reproduce native attention0 branch scores with maximum absolute error
   `<=1e-10`; every factor row has RMS one within `1e-6`; token roles are disjoint and contain exactly 50,257 IDs;
   all eight offset families and natural-score hooks are live.
2. **B — a multi-head shared-half subspace exists.** The FIT graph contains at least three distinct architectural
   heads in one component. Every edge in a spanning tree has at least 16 centered shared dimensions on both query
   and key sides, centered projector overlap at least `.15` on both sides, and each overlap exceeds its matched
   row-permutation 99th percentile by at least `.08`.
3. **C — the vocabulary transports to unseen tokens and complete scores.** Every spanning-tree relation has SELECT
   query and key factor `R2>=.50`, exact branch-score `R2>=.60`, and branch-score margin at least `.40` over its
   permuted-source control. The selected branch relation is stable across the two offset halves with `R2` difference
   at most `.15`.
4. **D — shared one half, different companion.** For every spanning-tree relation, companion-branch score `R2<=.35`
   or selected-minus-companion `R2>=.25`; and natural complete-product correlation is at least `.20` lower than the
   selected branch correlation. Thus the result cannot be “both halves and the whole head are simply copies.”

**Strong null:** A fails; no cross-head pair has even 8 centered shared dimensions on both query and key sides; best
SELECT branch-score `R2<=.25`; best real score transfer is within `.10` of its permutation control; or every candidate
shares both companion and selected branches equally within `.10`.

## Decision and literal price

- A+B+C+D with no null licenses a finite natural-score intervention: share the source half plus two 128-by-128 maps,
  retain each head's distinct companion, and compare against independent/same-price score approximations before CE.
- A+B/C with D failure identifies broader whole-head similarity, not the user's compositional shared-half object.
- A with B/C failure closes complete and at-least-16-dimensional cross-head half reuse at attention0 under this
  exact token-function metric. It does not close tiny sparse atoms. Move the downstream-response test to attention1,
  where the dossier gives a stronger redundancy/copy prior.
- A failure repairs the fold/harness only.

One proposed relation adds `2*128*128 = 32,768` map values but reuses a full source factor generator. The native
target query/key projection pair contains `2*128*1152 = 294,912` weights. These numbers are only a prospective price:
no saving exists until exact factor generation, score transport, physical CE, and all shared-source costs are counted.
