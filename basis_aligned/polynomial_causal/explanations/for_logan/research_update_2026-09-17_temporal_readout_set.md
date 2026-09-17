# 17 September — The temporal will/had readout set, through the same battery

**Path now.** Adverb cue (tomorrow/earlier) → mid-block writers (mlp8, attn8, mlp9, attn9, mlp10) build a
subject-NP state at `the`/agent → head 11.3 reads it at the final query; heads 9.1, 9.4 and 15.5 read the
cue and the NP → all four write along `O_h^T(u_will − u_had)` → will/had. **Delta:** opened at 23:09 UTC
after the aspectual component reached its declared limit; nine preregistered runs (v27–v36) in 70 minutes
with the same battery, code and nulls. Claim table: `basis_aligned/claude_hourly_review/TEMPORAL_DOD_SCORECARD.md`.

**Most instructive results.** (1) The four-head set removes 81% of the will/had margin on fresh rows and is
selective on was−were even though 11.3 is the subject-number head. (2) The strict four-way additivity bar fails
twice by 0.02 while all six pairwise Möbius terms are below gate and the head split beats random splits — the
overshoot is a real 3% superadditivity, recorded as a failure. (3) On natural rows (FineWeb and the Pile OOD
corpus) the removal shifts will−had by −1.3 to −1.6 logits on tomorrow-cued rows and +0.5 to +0.6 on
earlier-cued rows; head 11.3 alone carries only the tomorrow side.

**Metrics box.** As in the aspectual report: damage (logits, oriented to the native answer), fraction of the
native margin, norm-matched random null (16 seeds, max reported), retention, fresh/opened labels.

## Claim table (abridged)

| claim | tag | rows | numbers | status |
|---|---|---|---|---|
| S = {11.3, 9.1, 15.5, 9.4} readout removal damages will/had | edit | fresh lexicon, 2 constructions | 81%; null max 0.05; positive 64/64 | passes |
| S is selective (was−were, who−which, night−day) | edit | fresh + 3 templates | within null + 0.25×damage everywhere | passes |
| Blind sweep and random four-head-set null | edit | opened | 11.3 1.48, 9.1 1.05, 15.5 0.41, 9.4 0.33; random max 0.02 | passes |
| Keep-only readout at the four heads retains their service | edit | opened | retention ≥ 1.0; random keep ≤ 0.02 | passes |
| Transfer to 3 new templates | edit | fresh | 82% / 78% / 36% (comma-final below the registered band) | 2 of 3 |
| Frozen 0.80 ± 0.15 on a 4th lexicon + comma-adverb construction | edit, frozen | fresh | 0.79 / 0.83 / 0.81 | passes |
| Natural FineWeb and Pile rows shift as predicted | edit, frozen | in-distribution + OOD corpus | tomorrow −1.34/−1.57, earlier +0.54/+0.59 | passes (11.3-alone sub-claim fails on earlier rows) |
| 11.3 reads the subject NP; 15.5 reads the cue | fold | opened | 11.3: the 0.47, agent 0.31; 15.5: cue 0.46 | established |
| NP state written by blocks 8–10; relay causal | fold + edit | opened | top-4 writers 0.70; mediated share 0.56 | passes |

## Five properties
Simple: held (4 heads + contrast; sweep + random null). Predicts OOD: held (authored, natural, Pile).
Extracted: held at the head boundary; ports = NP-state inputs and the cue reads (contextual, not token-only).
Selective: held. Composes: pairwise held, four-way strict bar fails by 3%.

## Limitations
- The set S was selected by a sweep on the line's opened authored rows; all numbers above are from rows that
  sweep never saw, except the fold/mediation rows (opened).
- No token-only generator here: unlike the aspectual head 8.1, 11.3's port is a contextual NP state.

Receipts: `bilinear_quotient/circuits/followups/temporal_auxiliary_dod_*` and `..._reuse_temporal_v27`.
