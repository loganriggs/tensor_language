# Task-circuit decomposition: memorized-successor-sequence in bilin18

Independent sanity-check decomposition of the successor behavior (weekdays /
months / alphabet, "X, Y, Z," -> " succ(Z)") in the 18-layer bilinear
no-softmax model, run against the full-model atlas. All scripts, JSONs and
subspace tensors in `basis_aligned/qk_mdl/algo_tasks/successor/`.

Stimuli: 60 clean/corrupted pairs (20 per family, starts varied cyclically;
45 analysis / 15 held out). The corrupted twin replaces the LAST element with
a different same-family member, so the correct successor changes while the
prompt's position structure does not. Prompts are 6 tokens
(`[e0][,][ e1][,][ e2][,]`); every family member is a single GPT-2 token with
and without leading space, so clean/corrupted twins are length-matched and
differ only at position 4.

## 1. Does the model follow the last element or the position?

It follows the **last element**, essentially never the position — but the
corrupted design exposed a third mode. On corrupted prompts
("Wednesday, Thursday, Monday," style) the prediction falls into three modes
(n=20 per family):

| mode | weekday | month | alphabet |
|---|---|---|---|
| successor of the NEW last element (successor lookup) | 45% | 80% | 50% |
| successor of the original third element (position counting) | 0% | 5% | 10% |
| the original third element itself, i.e. ignore the intruder and continue the coherent prefix | 45% | 0% | 25% |
| other (mostly " and", i.e. list-end) | 10% | 15% | 15% |

So the behavior is successor-lookup, not position-counting (0-10% position
mode); the main competing behavior is treating the out-of-order intruder as
noise and continuing the coherent run — strongest for weekdays. Clean
accuracy: weekday 85% (all misses are cyclic wraps "…Saturday, Sunday," ->
" and" instead of " Monday"; the model does not wrap weekdays), month 100%,
alphabet 65% — matching the briefed rates on non-wrap prompts. The logit
margin `logit[clean successor] - logit[corrupted successor]` separates cleanly
(family means +2.6 to +6.2 clean vs -1.1 to -2.6 corrupted; per-pair
denominator median 7.9, min 0.75), so margin recovery is a well-posed patching
metric even on pairs whose argmax lands in the ignore-the-intruder mode.

Files: `s1_stimuli.py`, `stimuli.json`.

## 2. Component patching: one circuit, not three

Activation patching (clean -> corrupted, full component output, 45 analysis
pairs), metric = recovered fraction of the summed margin
`(m_patched - m_corr) / (m_clean - m_corr)`, over all 180 components
(162 heads + 18 MLPs).

Top-10 overall: **head L8.H3 alone recovers 0.66** of the margin; everything
else is small: MLP8 0.17, MLP9 0.16, MLP10 0.13, MLP11 0.11, L8.H7 0.09,
L13.H4 0.07, MLP14 0.07, MLP12 0.07, L6.H3 0.06. Cumulative (patching top-k
together): top-1 0.66, top-3 0.77, top-5 0.82, top-8 0.96, **top-10 0.99**.
The circuit is one dominant mover head (L8.H3) plus the layer 8-14 MLP stack
that amplifies and cleans up.

**One circuit or three?** One circuit at the component level. L8.H3 is the #1
component for every family individually (weekday 0.60, month 0.76, alphabet
0.59), and the per-family top-5 lists are near-identical (L8.H3 + MLP8/9/10 +
{L13.H4 or MLP11}). Spearman rank correlations of per-family importance
vectors over all 180 components: weekday~month 0.57, weekday~alphabet 0.50,
month~alphabet 0.42 (all p < 1e-8); restricted to the union of per-family
top-20 components (where the signal lives): weekday~month **0.84**,
weekday~alphabet 0.63, month~alphabet 0.66. (Steps 3-4 add the nuance: the
shared circuit routes family-specific payload codes.)

**Mechanistic side-findings.** (a) At the prediction position, L8.H3's
pattern weight concentrates on the last-element position (mean |pattern| 0.27
at pos 4 vs 0.11 for the next key; the sign is negative, which is legal in
this unnormalized bilinear attention and composes with the OV sign). (b) The
value payload the head routes is NOT its own layer-8 value projection: lamb
at layer 8 is **4.0**, so the head's value is `-3*v_L8 + 4*v1_L0` — dominated
by the layer-0 value stream. Zeroing the L8 c_v head-3 slice keeps 98.7% of
task score, while zeroing the layer-0 c_v head-3 slice drops it to 34% and
zeroing the L8 c_proj head-3 columns to 40%. The circuit: token identity is
encoded into the layer-0 value stream (the v1 skip), and L8.H3's QK pattern
routes it from the last-element position to the prediction position, where
MLPs 8-11 map it to the successor logit.

**Atlas agreement.** The suggested nearest atlas tasks are only weakly
related: Spearman of our overall importance vector vs atlas 'capital' 0.30,
'funcword' 0.28 (significant but low; top-10 overlap: 0 components — atlas
capital/funcword importance is dominated by the MLP0-3 early stack, which is
irrelevant here). The real nearest atlas tasks are **'digit' (Spearman
0.35)** and, for the mover head specifically, **'induction': L8.H3 is atlas
rank 4 of 180 for induction and rank 7 for digit** (vs rank 108 for capital,
87 for funcword). The whole-vector induction correlation is negative (-0.17)
only because the successor circuit's MLP stack (MLP8-11) actively hurts
atlas-induction (ranks 167-179 there). Reading: the successor circuit reuses
a sequence-continuation/induction-family head for memorized successor lookup
— consistent with the corrupted-behavior modes, where "continue the coherent
run" competes with intruder lookup.

Files: `s2_patching.py`, `patching.json`.

## 3. DAS-lite subspace

**Literal spec site fails, diagnostically.** An r-dim orthonormal subspace
(QR-parameterized, trained with Adam on the 45 analysis pairs, base = clean
run, source = corrupted twin, success = argmax moves to the source's
successor) was first trained at the top component's layer: the residual
stream entering block 8, last-element position. The **full-vector-swap
ceiling at that site is 0% flips** (still-clean 80%): swapping the entire
pos-4 residual entering L8 changes L8.H3's keys but cannot touch the payload,
because the payload is the v1 skip stream computed at layer 0 (lamb finding
above). Learned subspaces r=1/4/16 and random controls all flip 0%. This is
a positive control of the v1 mechanism, not a training failure.
(`s3_das.py`, `das.json`, `das_Q_all_r*.pt`.)

**Corrected site: embedding stream (input to block 0), pos 4** — upstream of
where the payload is created. A full swap here equals the corrupted run, so
the ceiling is the model's intruder-following rate.

| r | held-out flip rate | still-clean | random control (5 seeds) |
|---|---|---|---|
| full swap (ceiling) | 53% | 0% | — |
| 1 | 0% | 67% | 0% |
| 4 | 7% | 20% | 0% |
| 16 | **47%** | 7% | 0% |

r=16 reaches 88% of the behavioral ceiling; r=1 is insufficient; the
functional dimension of last-element identity at this site is between 4
and 16. Random subspaces never flip anything.

**Cross-family transfer: none.** Family-trained subspaces (400 epochs,
r=4/16): the weekday-trained subspace flips weekday held-out pairs at 80%
(4/5) but months and alphabet at **0%** (0/20 each); month-trained flips
months 40-60% and others 0%; alphabet-trained fails even in-family (0%,
consistent with alphabet's small margins). So at this site "successor-ness"
is NOT a shared movable subspace — what moves behavior is family-specific
token-identity content. Geometrically the family subspaces DO overlap (r=16
principal-angle cosines: first ~8 values 0.7-0.98 for every family pair),
but the overlap is not functionally sufficient — a shared "ordered-sequence
token" component with family-specific payload dimensions carrying the actual
lookup key.

Files: `s3b_das_l0.py`, `das_l0.json`, `das_l0_Q_all_r*.pt`.

## 4. Ethan's weight reduction (data-conditioned rank)

**First target choice was wrong and the run said so.** Taking "the most
important W" as the L8 c_v head-3 rows (the head's own value matrix) gives
rank-1 retention 99.7% — with even data-free SVD rank-1 at 98.8% — because
that matrix is nearly null in the circuit (lamb=4.0, see above). Kept for
the record in `s4_weightred.py` / `weightred.json`; superseded after
zero-ablation triage of candidate matrices.

**Corrected target: L8 c_proj columns for head 3** (1152 x 128 — the head's
output channel; zeroing it drops the task to 40%). X = head-3
pattern-weighted value outputs at all positions of an extended battery
(sequences of 3-10 elements, all valid starts, all three families; n=3920
>= 3000). Y = W X^T, SVD, W'_r = Y_r pinv(X^T, rcond=1e-4), evaluated on
held-out stimuli (margin-score retention), model otherwise intact:

| r | data-conditioned retention | data-free SVD retention | FineWeb dCE (data-cond) |
|---|---|---|---|
| 1 | 0.37 | 0.42 | +0.0032 |
| 4 | 0.48 | 0.47 | +0.0030 |
| 8 | 0.80 | 0.53 | +0.0019 |
| **16** | **0.96** | 0.61 | **+0.0006** |
| 32 | 1.00 | 0.73 | +0.0006 |
| 64 | 1.00 | 0.86 | +0.0005 |
| 128 | 1.00 | 1.00 | -0.0000 |

**Minimal rank for >=90% retention: 16 data-conditioned vs 128 (full rank)
data-free** — an 8x saving from conditioning on task inputs. General damage
at r=16 is negligible: FineWeb (rows 500-519, length 128) CE 3.4281 ->
3.4287 (+0.0006; for scale, zeroing the whole matrix costs only +0.0038 —
this head barely matters off-task).

**Weight-level sharing test.** W'_16 fit on weekday-only inputs (n=728):
weekday retention 0.999, but month 0.42 and alphabet 0.47 (at r=32: 1.00 /
0.46 / 0.56). Although the three families share the same components (step 2)
and rank 16 fully suffices for weekdays, the 16 head-output directions that
carry weekday successors do NOT span the month/alphabet payloads. Same
conclusion as DAS: a shared routing circuit with family-specific payload
codes.

Files: `s4b_weightred_cproj.py`, `weightred_cproj.json`.

## 5. Honest failures / caveats

- The briefed "weekday 100%" holds only for non-wrapping starts; cyclic
  wraps fail (-> " and"), so clean weekday accuracy over cyclic starts is 85%.
- The "position vs last element" dichotomy in the design was incomplete: the
  model has a third mode (ignore the intruder, continue the coherent prefix)
  that accounts for most non-lookup responses, especially weekdays (45%).
- DAS at the literal spec site (the top component's layer) is impossible for
  this circuit in this architecture — the payload rides the v1 skip stream
  from layer 0 past the patch site. Reported as a 0%-ceiling negative result
  and re-run at the corrected upstream site.
- The alphabet-trained DAS subspace failed to flip even alphabet pairs;
  alphabet margins are small (mean clean margin 2.6; one pair with
  denominator 0.75), so all alphabet-specific numbers are noisier.
- Per-family held-out n=5, so family-level flip rates are coarse (80% = 4/5).
- The first weight-reduction target (L8 c_v) was a near-null matrix; its
  "rank-1 works" result is vacuous and was replaced by the c_proj-column
  analysis.
- Atlas comparison caveat: atlas importances are mean-ablation CE deltas on
  FineWeb text; ours are patching margin recoveries on 6-token prompts — the
  Spearman comparisons span those two different metrics.

## Bottom line

The memorized-successor behavior is a single, sharply localized circuit:
layer-0 value embedding (v1 skip) -> L8.H3 pattern routes the last element's
payload to the prediction position (0.66 of the effect alone; top-10
components 0.99) -> MLP8-11 write the successor. The same components serve
all three families, but every payload-level probe (DAS interchange at the
embedding, data-conditioned weight rank) shows family-specific codes inside
the shared machinery: successor lookup is one algorithm with three lookup
tables, not three circuits and not one shared "successor-ness" direction.
