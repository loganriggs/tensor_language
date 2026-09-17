# 17 September — The aspectual has/had readout component, taken through the definition of done

**Path now.** The since/by cue token's block-0 value is read by head 8.1 wherever its attention lands
(the final query and the `last`/period/`the` bank); heads 9.1 and 9.4 relay the bank copy; all three write
along one weight-defined direction per head, `O_h^T(u_has − u_had)`, which the unembedding reads directly
(57%) and MLPs 9–15 amplify (43%). **Delta today:** this component did not exist in the released program
(which named block-9 H1/H4 and MLP4). Twenty-one preregistered runs (edits with nulls, folds, frozen
predictions) establish it against `better_circuits.md` §1; the claim table and every receipt are in
`basis_aligned/claude_hourly_review/ASPECTUAL_DOD_SCORECARD.md`.

**Most instructive result (a failure).** Zeroing whole head writes ranked attention5's four heads as the
strongest component (0.94 logits). Sixteen random directions of the same norm did the same damage (median
1.05): the number was norm, not aspect. Every later removal carries that null. Second failure: random
coordinate pieces of a cue-defined removal are individually "selective" too, so per-piece selectivity does
not certify a module partition; the partition is supported by damage allocation and the blind head sweep.

**Metrics box.** *Damage*: native has−had margin minus the edited margin, oriented to the native answer,
mean over rows (logits); *fraction* = damage / native mean margin (2.07 on the discovery rows). *Null*: 16
random directions of the removed norm at the same slice; max reported. *Retention* = 1 − damage(arm) /
damage(zero). *Fresh* = rows unused at any prior stage; *opened* = used by an earlier removal.

## Claim table (abridged; full table in the scorecard)

| claim | tag | rows | numbers | status |
|---|---|---|---|---|
| Removing the weight-only readout direction at {8.1, 9.1, 9.4} damages has/had | edit | fresh lexicon, 4 templates | 50–66% of the margin; null max ≤ 0.02; positive 160/160 | passes |
| Three unrelated readers (was−were, who−which, night−day) stay at null level | edit | same | excess over null ≤ +0.09 | passes |
| Only 3 of 162 heads carry that removal (blind sweep) | edit | opened | 9.1 0.36, 8.1 0.34, 9.4 0.33; next 15.5 0.18; rest < 0.04 | passes; 8.1 was unnamed |
| The three heads compose additively | edit | 4 templates | gap ≤ 0.06 vs bars ≥ 0.08 | passes |
| Keeping only the readout projection at the three heads keeps their whole has/had service | edit | opened | retention 1.01; random direction 0.015 | passes |
| Head 8.1 = per-cue constant × λ × block-0 value of the cue token (token IDs in) | edit | cross-construction, new templates | retention 0.83 / 0.93 | passes |
| Frozen prediction 0.54 ± 0.15 on a third lexicon and a new construction | edit | fresh | 0.43 / 0.53 / 0.54 | passes |
| The bank 9.1/9.4 relay: ¼ is 8.1's cue write (edit-confirmed), the rest a diffuse MLP5–8 polynomial | fold + edit | opened | mediated 0.27; top MLP8 pair 10–14% | open port |
| Natural rows: shift scales with 8.1's attention on the cue; since −0.24 and temporal-by +0.36 (FineWeb, in-distribution) reproduce on the Pile OOD corpus (−0.24 / +0.27) | fold + edit | natural in-distribution + OOD corpus | r −0.89 (since), +0.63 (by); temporal-by panel +0.36 | passes; any-sense `by` fails |

## Five properties

Simple: 3 heads + the has/had contrast + 2 constants (8.1). Predicts OOD: held on new lexicon and new
templates; natural FineWeb text (in-distribution) and the Pile OOD corpus transfer when the cue is temporal (since
rows −0.24 logits on both corpora; temporal-`by` +0.36 / +0.27), with the same frozen bars. Extracted: held at the head boundary; 8.1 closed;
two declared open ports (MLP5–8 bank polynomial, block-9 patterns). Selective: held (random null, three
readers). Composes: held at head grain; one serial pair (mlp4→attn5, from the older path) above gate.

## Limitations that change the reading

- Every removal direction is a weight object, but the *sites* were chosen by a sweep on the discovery rows
  (opened); the numbers on fresh panels are the evidence.
- The natural-corpus `by` result depends on the cue being temporal; the block-0 value is sense-blind and the
  attention pattern carries the sense. That pattern's key is built by mid-block writers (MLP5–7, attention
  5–7; diffuse, panel-dependent), so "token-only" is exact for 8.1's value branch and a template fact for
  its pattern; the cue key is a declared open port.
- The older MLP4 → attention5 route explains ~5% and is not part of this component.

Receipts: `bilinear_quotient/circuits/followups/aspectual_anchor_dod_*`. Board entries 20:56–22:52 UTC.
