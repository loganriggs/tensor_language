# MLP0 centered interaction head-carrier preregistration

**Registered:** 2026-09-01 15:07 UTC

**Owner / rung:** Codex / 402

**Parent:** rung401 exact centered token/context attribution

**Claim level:** mechanism identification only; no compression or adoption

## Decision

Rung401 identified the centered token×context interaction `I` as MLP0's largest causal role on both FIT and SELECT.
The next uncertainty is whether `I` is carried by a small, stable subset of attention0 heads or is collective across
the nine heads. This is not answered by the old attention-head map: that assay removed complete attention-head
writes, whereas this assay changes only the head's contribution to MLP0's centered interaction while retaining the
token main, context main, normalization modulation, native attention write, and every downstream component.

The historical direct-head control is frozen from
`attn_head_map_results.json` (SHA-256
`f54b26891cc75e5feaeadd6bb19defb68fd8be0af535e3aabbaa1c6f7522358b`). Its layer0 removal-cost vector for heads
0–8 is `[.0020,.0001,.0007,.0877,.0004,.0008,.0039,.0074,.0046]`: head3 is largest and the top two account for
.883 of positive direct cost. Rung402 asks whether that sparsity survives at the isolated MLP0 interaction port.

## Exact head decomposition

Let attention0's pre-output-projection head state be `y=[y_0,...,y_8]` and split the output projection by input
columns as `O=[O_0,...,O_8]`. In real arithmetic,

`a = O y = sum_h O_h y_h = sum_h a_h`.

Capture `y` from the live attention0 projection and compute each `a_h` with the corresponding frozen column block.
Because the deployed projection is BF16, retain the observed arithmetic remainder

`a_eps = a_native - sum_h a_h`

in every arm. Estimate `mu_h=E_FIT[a_h]` and `mu_eps=E_FIT[a_eps]`. Then

`a_native-mu_a = sum_h(a_h-mu_h) + (a_eps-mu_eps)`.

Rung401's centered interaction is linear in this context deviation:

`I = sbar^2 Down( Left(delta_e)*Right(delta_a) + Left(delta_a)*Right(delta_e) )`.

Therefore it splits exactly into nine semantic head terms `I_h` plus an always-retained numerical term `I_eps`.
The explicit vector normalization residual `R` from rung401 also remains in every arm. No rounding remainder is
assigned to a semantic head.

## Physical arms and scores

**Pre-execution amendment, 15:10 UTC:** retaining `I_eps` means a no-semantic-head arm is not algebraically identical
to rung401's arm that removes all of `I`. Add the latter as a separate control before implementation. This makes the
parent-boundary check exact and tests that the BF16 remainder is causally negligible without assigning it to head8.
No model/data execution preceded this amendment.

On the same frozen 96 FIT and 96 SELECT documents and scored positions 64:256, always retain rung401's constant,
`T`, `C`, `S`, native bias, `R`, and `I_eps`. Change only which semantic `I_h` terms are present:

- `ZERO_I`: neither semantic heads nor `I_eps`, exactly rung401's no-interaction boundary;
- `NUMERIC`: `I_eps` only and no semantic interaction head;
- `FULL`: all nine interaction heads;
- `SINGLE_h`: only head `h`'s interaction term;
- `DROP_h`: all interaction heads except `h`.

This is 21 arms. For each head define

- singleton benefit `u_h = CE(NUMERIC)-CE(SINGLE_h)`;
- removal benefit `n_h = CE(DROP_h)-CE(FULL)`;
- endpoint-average benefit `b_h=(u_h+n_h)/2`.

`u_h` measures sufficiency from the no-interaction boundary; `n_h` measures necessity at the full-interaction
boundary. `b_h` is explicitly not a Shapley value. Report all three, the positive top-two share of `b`, FIT/SELECT
rank correlation, and correlation with the frozen historical direct-head removal vector.

## Frozen predictions

1. **A — exact and live.** In both roles, the nine `I_h` plus retained `I_eps` reconstruct rung401 `I` at relative
   MSE `<=1e-8`; `FULL` and `ZERO_I` reproduce rung401's `T+C+I+S` and `T+C+S` pooled CE within `1e-6`; every arm has
   the expected 24 forwards, 432 attention calls, 24 MLP0 calls, and 408 later-MLP calls. The BF16 head-write
   remainder has relative squared energy `<=1e-4`, and `|CE(NUMERIC)-CE(ZERO_I)|<=.002` nat.
2. **B — old head map predicts the isolated interaction.** On SELECT, head3 has the largest `b_h`, and Spearman
   correlation between `b_h` and the frozen direct-head removal costs is at least `.50`.
3. **C — sparse carrier.** In each role, the two largest positive `b_h` values account for at least `.65` of total
   positive endpoint-average benefit. FIT and SELECT have the same top head and `Spearman(b_FIT,b_SELECT)>=.75`.
4. **D — an individually material route exists.** On SELECT, some head has singleton benefit `>=.05` nat, some
   head has removal benefit `>=.02` nat, and at least one head has both endpoint benefits positive.

**Strong null:** A fails; `Spearman(b_FIT,b_SELECT)<=0`; SELECT direct-map Spearman `<=0`; both roles have positive
top-two share `<=.35`; or every SELECT `|b_h|<.01` nat.

## Decision rule and price

- A+B+C+D with no null: interaction is sparse and aligned with the old head map; resolve the top fixed head set by
  source-position/distance contribution and physical interchange.
- A+C+D but B fails: interaction is sparse, but MLP0 changes which attention heads matter; follow the new ranking,
  not the historical direct-cost ranking.
- A holds but C or D fails: the head-level interaction is distributed/redundant; do not enumerate source positions.
  Instead audit which `T/C/I/S` branch the already adopted rank448 MLP0 context projection damages.
- A fails: repair the instrument only; publish no head attribution.

The assay reuses all 15,926,400 native MLP0 weights and all native attention0 weights. Diagnostic reference storage
adds nine 1,152-vectors for head means plus the rung401 moments; it does not remove a single deployed value and is not
a compression candidate. FINAL remains unopened. The parent receipt SHA-256 is
`6650b97c9f5b53714d29f999eff6653bdbc9273c9238e4c10ce607d8d5728277`; the frozen row receipt SHA-256 is
`ce4a6f8eeb20840711bb20677ff8310f1a39db55b50106face1157cd2feeef7f`.
