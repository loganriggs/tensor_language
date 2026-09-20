---
name: aspectual-readout-component
description: Key 2026-09-17 finding on the bilin18 aspectual has/had path — the readout component is heads {8.1, 9.1, 9.4} along O_h^T(u_has−u_had); weight-only removal is selective, additive, template-stable; 8.1 was missed by the released path
metadata:
  type: project
---

On the bilin18 aspectual has/had path, a weight-only removal (subtract the projection of each
head's pre-c_proj slice onto `O_h^T (u_has − u_had)` at the final query) identified the readout
component as heads **8.1, 9.1, 9.4** (blind 162-head sweep; nothing else above 0.04 logits except
15.5 = 0.18 and 11.3 = 0.12). The triple removes 50–66% of the has/had margin on four
constructions, beats a norm-matched random null, passes was−were / who−which / night−day gates,
and is additive. Response census: 57% direct readout, 43% via MLPs 9–15 (mlp11 largest), mlp17
opposes; attention 10–17 inert. Whole-write zeroing had wrongly ranked attention5 first (norm
effect). Receipts: `bilinear_quotient/circuits/followups/aspectual_anchor_dod_*`; scorecard in
`basis_aligned/claude_hourly_review/ASPECTUAL_DOD_SCORECARD.md`.

**Why:** the released program (v1–v12) named block9 H1/H4 and never 8.1; 8.1 is also the
temporal will/had line's subject-onset writer, so it is a shared head.

**How to apply:** treat {8.1, 9.1, 9.4} as the aspect readout component; remaining unheld
properties are Extracted (standalone) and strong-form OOD (token IDs). See
[[claude-circuit-lane-2026-09-17]].

Update 2026-09-17 22:47 UTC: 19 receipts (v1–v19). Head 8.1 is a token-only generator (per-cue
constant pattern × λ × block-0 value of the since/by token; 2 constants transfer to new templates at
0.93 retention); it also writes the bank at `last`/period/`the` that 9.1/9.4 relay (27% of their
readout effect, edit-confirmed). The MLP5–8 part of that bank is a diffuse polynomial (kill criterion
tripped; declared open port). Frozen prediction 0.54 ± 0.15 of the margin held on a third lexicon and
a new construction. Helpers: `ops/board_append.sh`, `ops/dod_record.sh`.

Update 23:02 UTC: v20–v25 done. Natural FineWeb rows: since-shift transfers; temporal-by panel +0.36; the
cue KEY is contextual (mid-block MLP5–7/attn5–7, diffuse — open port), only the value is token-only. Random
three-head-set null passes; lib refactored onto `attention_factors` (bit-exact, v24). Folding on this path
is stopped at two open ports; scorecard has 37 claim rows.

Update 23:05 UTC: v26 Pile OOD panel passed the frozen bars (since −0.24, temporal-by +0.27, r −0.84).
All five properties held at head grain with two declared open ports. Next candidate move: reuse test —
does head 8.1's token-only cue table also serve the temporal will/had line (8.1 is its subject-onset
writer)? FineWeb = training corpus (in-distribution), Pile = OOD; label rows accordingly.

Temporal will/had line (opened 23:09 UTC after review 3; scorecard TEMPORAL_DOD_SCORECARD.md): readout set
S = {11.3, 9.1, 15.5, 9.4} on O_h^T(u_will−u_had): 81% of the margin on fresh rows, selective, keep-only
sufficient, frozen 0.80±0.15 held on a 4th lexicon + new construction, natural FineWeb/Pile rows pass
(tomorrow −1.3/−1.6, earlier +0.5/+0.6). 11.3 reads the subject NP (contextual, few mid-block writers);
15.5 reads the cue. Four-way additivity bar fails by 0.02 (pairwise fine).

Cross-line finding 23:23 UTC (v37): on the temporal will/had line, attention8's write into the subject-NP
state that head 11.3 reads is 99.6% head 8.1 and 98% sourced from the tomorrow/earlier token — the same
head 8.1 that reads since/by on the aspectual line. 8.1 = a general temporal-cue token reader (block-0 value
of the cue, written wherever its pattern lands); block-9 pair 9.1/9.4 relays; 11.3 reads the NP state.

23:28 UTC: temporal line done-at-grain (v28–v41): 8.1's NP port closed with 8 constants (v40); MLP8–10 NP part
diffuse (v41, kill). Both lines now have two declared open ports each. Registry cross-links added for both.

Third line (narrative tense was/is, v42–v47, 23:30–23:59 UTC): blind sweep on fresh rows returns the SAME four
heads as the temporal set {15.5, 11.3, 9.4, 9.1}; 72–75% of the margin; frozen 0.74±0.15 held; keep-only,
random-set null, three new tense frames all pass; readout heads read the second sentence's contextual state
(tail), not the tensed verb; 8.1 absent (no single cue token). Cross-line summary in
claude_hourly_review/SHARED_READOUT_COMPONENTS.md. Lesson: single-token occupational nouns are nearly exhausted
across corpus lists + my panels; vary places/adjectives and declare subject reuse.

00:30 UTC 18 Sep: nine decisions censused → two readout families at the auxiliary slot: temporal {9.1, 9.4, 15.5}+11.3
(tense/mood/aspect) and number {5.7, 7.8, 9.7}+11.3 (agreement); 11.3 in both; preposition on/of uses neither.
Number family: readout pair {11.3, 7.8} one-directional; 5.7 upstream, off-direction, not token-only; number
readout direction entangled with has−had at the number heads (cos −0.4..−0.6) — orthogonalized weight-only
direction restores selectivity. Report: for_logan/research_update_2026-09-18_readout_families.md.
Protocol lesson: replacing slices by projections of native captures ≠ in-forward keep-only (later heads see the
edited stream); use keep_span/keep_only modes.

00:45 UTC: number family scope — head readout {11.3, 7.8 (+5.7, 9.7)} carries number only when the subject is not
adjacent to the auxiliary; adjacent-subject number is MLP-borne (MLP16 0.40, MLP0 0.22, MLP17 opposing; no head
beats a norm-matched null). Scorecard NUMBER_DOD_SCORECARD.md rows 1–17.

00:52 UTC: five lines through the battery (aspectual, temporal, narrative, number [scoped], modal); readout atlas
v68 (226 families × census) queued — receipts atlas_<family>_v68_result.json; summary readout_atlas_v68_result.json.
Generic tools: dod_reuse_census.py (sweep+set), dod_battery.py (fresh-row battery). Reviews written in-turn ~hourly.
