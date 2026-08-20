# Circuit SOP -- step-by-step procedure for one circuit (swarm-runnable)
# v4 (2026-08-20): STEP 3M IS RETIRED (writeup 510) -- the damage-cluster
# mechanism screen produced no causally real writer mechanism in 60 leaves
# on either decomposition. Leaf work is now MAINTENANCE ONLY. New circuit
# work starts from a BEHAVIOUR: see BEHAVIOUR-FIRST below. Supersedes v3.
# v4.1 adds UNITS OF ANALYSIS -- decompose by subspace channel, not by head.

Written for a FRESH stateless agent (Sonnet/Opus class) holding only this
file + census_lib. The procedure is verification-driven: every judgment an
agent makes is checked by a computed bar, so a weaker model can fail to
find things but cannot certify junk. GPU steps go through queue.txt
(absolute paths only!) or direct `python -u` if the queue is idle.

SETUP (required, first lines of every session):
    import census_lib as cl
    cl.use_state(cl.PT+'census_state_diverse.pt')   # 1000-row FineWeb tree
Grid is 1000x256 (flat 256000). All cl.* functions then target the
diverse tree. Forgetting use_state() silently runs the OLD 212-row tree.
Standalone scripts run from another directory need PYTHONPATH=<workdir>
(python puts the script's dir, not cwd, on sys.path). Both-halves
identity gate: cl.sign_stats_half(tag, d, 0, 500) and (tag, d, 500,
1000) -- both concentrations must clear >=3.

Deliverable: one merged circuit record via census_lib.write_circuit(tag, ...)
conforming to CIRCUIT_SCHEMA.md. Do the steps IN ORDER; record every number.

## What counts as a result (user standard, 2026-08-20)
Compression facts ("k reads suffice", "top-N units survive") and surface
programs are NOT results -- they are starting points. A result names the
MECHANISM in plain language: what variable the machinery computes, who
WRITES its inputs, what MOVES them, and what downstream reads them --
each edge tested by intervention. The finished induction record is the
template: identity code (wte + MLP-chain enrichment), writer (mlp0),
couriers (a6.h3, a4.h7 -- prev-token heads), computation (double-QK
coincidence), every arrow causally verified (BILIN18_CONNECTION 393-408).

## UNITS OF ANALYSIS -- heads and layers are not the natural ones
(user correction, 2026-08-20, writeup 527)

Every localization in this program has split the model along
architectural seams: whole layers, then whole heads, then whole
MLPs. Those are the units the code is written in, not the units
the computation uses, and two results show the cost.
  521 concluded the digit subspace is "distributed" because no
  head carried more than a third of it. That conclusion is only
  forced if heads are the units.
  524 found that layer 10 does three unrelated jobs in three
  different heads, so a LAYER is a shared address rather than a
  shared computation.
Finer and coarser units are both available exactly, and cost
nothing extra to compute:
  BELOW A HEAD. A head writes into the residual through its own
    128 columns of c_proj. If the object of interest is a
    subspace P (rank r), that head's entire possible contribution
    to it passes through the r x 128 matrix M_h = P^T W_h. Its
    singular values give the width of the head's CHANNEL into
    that subspace, from weights alone, with no data. Ablate the
    channel rather than the head.
  ACROSS HEADS. Several heads writing into one subspace can be
    compared by the principal angles between their channels'
    column spaces. Shared directions mean cooperation (and
    explain superadditivity); orthogonal ones mean the heads
    partition the subspace. Again weights only.
  INSIDE A CHANNEL. Channel content is SUM_k score(q,k) M v(k),
    so it decomposes exactly by SOURCE POSITION. That is where a
    semantic name comes from -- "the content arrives from earlier
    digits" is a mechanism; "head 8.3 matters" is a coordinate.
  RECURSING. v(k) is itself a linear map of the residual at k,
    which has its own exact writer decomposition
    (cl.writer_parts), so the same treatment applies one layer
    down. This is tier 5 of 512.
Rule: before reporting that an effect is distributed, check
whether it is distributed in the units the model uses. A negative
at head level is not a negative at channel level.

## BEHAVIOUR-FIRST (v4 default for NEW circuit work, writeups 495-506)
Three circuits in this program worked -- induction, the position-0
bias, the newline head -- and all three started from a behaviour or
a striking observation, never from a damage cluster. The newline
head reduced the first step to one cheap statistic, and it is the
one to reuse:
  1. Define a target class of POSITIONS (what token is being
     predicted), plus a POSITION-MATCHED control (the same
     positions jittered a few tokens) and a FULLY RANDOM set of
     the same size. Both controls are mandatory: the first catches
     a component that merely matters at certain sequence
     positions, the second catches a screen that fires on any
     position set.
  2. Mean-ablate each of the 36 components once and rank them by
     CONCENTRATION -- damage at the target class divided by that
     same component's damage elsewhere -- NEVER by absolute damage
     at the target. Ranking by magnitude finds the front of the
     model for every behaviour ever tried (writeup 494).
  3. Decompose the winning component into heads and repeat. If one
     head carries most of it, you have a candidate.
  4. Characterize the head causally: what token does deleting it
     move, at which positions, and by how much against the best
     competing token.
  5. Cross-check against head_atlas_results.json, which profiles
     all 162 heads with no knowledge of your behaviour. If the
     atlas independently singles out your head, say so -- that is
     an advance prediction that could have failed.
Reporting rules that came out of these runs, and are not optional:
report the PAIR (target and elsewhere) alongside every ratio, and
never score a bar on a quotient whose denominator is near zero or
whose comparison class the sample never populated. cl.score_bar()
enforces both and returns UNEVALUABLE rather than a verdict.

## Identity rule (2026-08-20 revision, from 381/404/406)
Leaf member-sets are ONE SAMPLE from a family of valid partitions. A
record's identity is its MACHINERY (probe bundles) + CAUSAL PROFILE.
Certification gate = SELECTIVITY REPLICATED ON BOTH CORPUS HALVES
(concentration >= 3 on rows 0-499 and 500-999 independently). Do NOT
use raw damage-profile cosine (the 395 gate) -- it certifies magnitude
stability, which anti-predicts selectivity (404). Surface token-class
programs are NOT identity either: only 1/72 leaves earned one under
strict doc-disjoint splits (410).

## Step 0 -- claim a leaf
Claim from swarm_shortlist.json (199 certified-selective tags; skip
depth-0 tags like "r.16" -- their concentration is ill-defined, 406).
Prefer tags with a pre-computed pack (sop_packs_certified.json /
sop_packs_shortlist.json = steps 1-2 already done; verify step 1
reproduces, then continue from step 3). PACK CAVEATS (dry-run 412):
packs were computed on a 60-row subsample -- only CONCENTRATION is
expected to reproduce, and only loosely (deltas up to ~0.5 observed;
the operative check is that YOUR full-grid value clears the >=3
gate); n_pos/n_neg/dce_members will differ by design against your
full-grid rerun. Pack 'examples' are
NOT canonical -- always regenerate with cl.examples(tag, d).
write_circuit now stamps tree.instance from the active state
('diverse-1000row-v1'); records with '212row-v1' on a diverse tag
predate the fix. Check circuits/registry.json for
a tag with no file yet. Never edit another leaf's file.
PARTIAL RECORDS: batch scripts may have pre-written steps 1-2 (causal +
examples, no story/program). That leaf is still claimable: re-run step 1
to verify the recorded numbers reproduce (report both), then continue
from step 3. write_circuit deep-merges: certification entries are
appended (never lost), dict fields merge, so just pass your new fields.

## Step 1 -- causal footprint (GPU; ~45s alone, MINUTES under swarm
## load -- run inline with a generous timeout, never park on a monitor)
    import census_lib as cl
    d = cl.leaf_ablate(tag)                # dCE under the leaf's own probes
    s = cl.sign_stats(tag, d)
GATE: s['concentration'] >= 3 (the field is provided).
THREAD `d` THROUGH the whole task -- examples, story_test_class and
your merge all take it; recomputing leaf_ablate per step wasted a
full GPU pass for a wave-4 agent. Capture `git rev-parse HEAD`
into a variable NOW: read at merge time it has already drifted
from concurrent commits.
CALIBRATION (430, provisional): a rank-matched RANDOM subspace in
the same components already scores ~2.4-2.7 concentration, so a
gate pass near 3 means little; quote your leaf's ratio to that
baseline when it matters. If FAILED, record
{'causal': s, 'certification': [gate FAILED]} and STOP -- the leaf is not
locally selective; do not write a story for it.

## Step 2 -- examples (CPU, instant)
    exs = cl.examples(tag, d)              # mechanical: top-3 + 3 random
Record verbatim. NEVER swap examples for prettier ones.

## Step 3M -- MECHANISM SCREEN [RETIRED 2026-08-20, writeup 510]
DO NOT run this step as a deliverable and do not certify a
mechanism from it. History, so nobody re-derives it: the screen
asks whether a writer's SHARE of a component's input is larger at
a leaf's member positions. Writeups 503-510 rebuilt it on a
corrected residual decomposition (the old one was wrong by up to
4,242x for early writers) and then tested every positive causally.
All failed. At the two leaves the corrected screen newly flagged,
matched RANDOM directions did 3-7x more damage than silencing the
writer the screen named; at r.3.0.2, the oldest positive,
silencing the named writer made the leaf's own positions BETTER.
The correction moved 60 of 167 top writers without changing that
conclusion, which is why the statistic and not its inputs is the
problem.
ALSO RETIRED with it: the member-vs-off-slice contrast as a bar.
Member positions were SELECTED for damage-sensitivity, so any
perturbation lands on them harder -- random directions score 8.5
and 18.0 on a statistic whose registered bar was 1.5. If you need
a localization readout, compare against MATCHED RANDOM DIRECTIONS
on the member positions, which is a comparison that can fail.
If you are reviewing an OLD record that cites 3M, the negative
still stands (a larger ablation failing to show enrichment is not
rescued by shrinking it); the positives do not.
    python leaf_input_decomp.py <tag>      # writes leaf_mech/<tag>.json
Decomposes the residual entering each of your leaf's machinery
components into exact writer contributions, member vs off-slice.
Record, verbatim, per component: the top writers with their
member/off-slice ratio and the ENRICHED / BEATS_NULL verdicts.
WHEN TO QUOTE RATIOS AT ALL (470): the table's ordering predicts
causal selectivity when there IS structure -- Spearman 0.84 on a
leaf with a positive, 0.76 with that leaf's own machinery excluded
-- but on NEGATIVE leaves it is unreliable (0.91 on one, 0.12 on
another) and the underlying spread is tiny (concentration range
~2 against ~7). So: if ENRICHED_STABLE2 is true anywhere, quote
the ratios and chase them. If it is false everywhere, report the
flat profile and the threshold and move on -- do NOT rank writers
by ratio in that case, the ordering is a coin flip.
SPECIFICITY IS REQUIRED (wave-3 reviewer catch, 425): a ratio
above 1 is measured against random positions in the SAME leaf, so
an adjacent-layer writer can look enriched simply because it is
adjacent. Before claiming a writer is this circuit's mechanism,
run `python leaf_input_decomp.py <tag> --baseline`, then run the
tool on the peer tags it names and compare: a ratio that
reproduces on unrelated leaves is a LAYER property, and the record
must say so.
Two honest outcomes, both publishable:
  ENRICHED true  -> mechanism lead: "this machinery acts where
                    writer X's contribution to C's input is
                    enriched (ratio r, null n)".
  ENRICHED false -> write "no STRONG single-writer mechanism
                    (top ratio r, threshold t)" -- NOT a blanket
                    absence claim. A wave-4 reviewer showed the
                    gate has little power against weak effects
                    (ratio 1.1-1.25) on leaves whose null noise is
                    wide, so quote `threshold_v2` and `headroom`
                    from the table and say what the test could and
                    could not have detected. Then look downstream
                    (leaf_output_decomp).
NEIGHBOUR CONTROL IS MANDATORY BEFORE ANY WRITER CLAIM (473): a
positive screen plus a peer-specificity pass is still not enough.
On the census's only surviving positive, ablating the enriched
writer m14 gave concentration 4.33/4.29 while the ADJACENT
unflagged m13 gave 4.44/4.89 -- the neighbour was more damaging.
The same happened for a14 versus a13. If you claim a writer,
ablate the component one layer either side and report all three
numbers; if the neighbour matches or beats it, the claim is a
BAND-level statement, not a writer-level one.
Escalation is LADDER-TIER, never blocking the merge: if ENRICHED,
note the target in the record and (only if you have time after
step 6) copy qk_writer_decomp.py's pattern to name what an
attention component's score compares -- that script is hard-coded
to the induction heads, so adapting it is a new-script job, not a
5-second step. For an MLP, check whether the enriched writer is m0
(the universal identity code, 411/415).

## Step 3 -- surface program (CPU, ~15s) [DEMOTED; context only]
    p = cl.leaf_program(tag)               # heldout + null
On the diverse tree use docid-parity splits (rows of one document are
adjacent; row parity leaks -- see sop_program_batch.py for the pattern).
Record bacc/null either way, but a PASS is a convenience label, NOT a
mechanism grade: surface programs passed only 1/72 on the diverse tree,
and per the user standard they are description, not mechanism. Do not
stop here on a FAIL; the mechanism steps below are the actual work.

## KNOWN GENERAL EFFECTS -- do not report these as discoveries
(added 2026-08-20 after four leaves independently "found" the
same population-level phenomenon)
PUNCTUATION AND DIGIT SPARING (writeups 462-464): ablating almost
any 16-dim probe bundle on this model spares punctuation targets
(dissociation about -0.025) and digits (-0.018) while damaging
space-initial words (+0.014), capitalised words (+0.006) and
NEWLINES most of all (+0.027). It is not leaf-specific, not a
frequency effect, and not explained by predictability. If your
leaf shows a punctuation claim, it is almost certainly this
effect. Report it as "consistent with the known general sparing
(464)" and, if you want to claim anything leaf-specific, show
your leaf's dissociation EXCEEDS the population value of -0.025
by a stated margin. Otherwise spend your step-5 budget on the
other classes.

## Step 4 -- claims, written blind-ish
Look ONLY at exs + s + your step-3M table. Write TWO lines:
  MECHANISM (required): <=25 words naming components and writers
    from step 3M with their numbers. This is the record.
  BEHAVIOR (optional): <=25 words on what members share and what
    the machinery pushes. WAVE-2 EVIDENCE (414/420): behavioral
    stories on these leaves usually reduce to the leaf's base rate
    or to one tokenizer bit, and three of four were WEAKENed by
    review. Write one only if step 5 clears its bar; otherwise
    record "no behavioral claim survives base-rate testing". If s shows a
two-signed split (minority_share >= 0.15), the story MUST say what the
push is and where it is wrong -- "helps X" alone is incomplete.

## Step 5 -- red-team your own behavioral claim (CPU) [v3 bars]
Mechanical, no judgment calls (wave-2 reviewers hand-built class
draws and varied):
    f = cl.examples_filtered(tag, d, kind, n=5)   # kind: subword,
        # space_word, digit, punct, capitalized, newline
    cl.story_test(tag, d, [x['gi'] for x in f['draw']], preds)
A behavioral claim is KEPT only if cl.story_test_class(tag, d,
kind, pred_help)['ROBUST_V2'] is true on the class it names
(population-level significance over EVERY member of that class,
n_available >= 10; the v1 'ROBUST' seed gate is deprecated -- it
was underpowered and demoted a real effect, wave-3 reviewer
catch, writeup 424) -- a
single seed-11 draw passes by chance about 2 times in 5 (wave-3
seed sweep), so ROBUST demands seed stability AND whole-population
significance. Also state how many (class, direction) pairs you
tested: with all 12 pairs searched, require p <= 0.10/12 on the
population, or pre-declare the pair before looking. COUNTING RULE
(wave-5 question): n_tests is every (class, direction) pair you
actually EVALUATE. Testing one class in both directions is
n_tests=2; sweeping all six classes both ways is n_tests=12.
Choosing a direction by eyeballing the data first and then
testing only that one is circular -- if you looked, count all the
pairs you could have chosen from (these leaves are ~50/50 two-signed,
so raw hit counts prove nothing). Otherwise record the numbers and
write "no behavioral claim survives base-rate testing".

## Step 6 -- merge
    cl.write_circuit(tag, {'causal': s, 'examples': exs,
        'story': {'blind_name': ..., 'program': p['program'],
                  'program_bacc': p['bacc'], 'program_null': p['null'],
                  'mechanism_level': 'surface' if PASS else 'none'},
        'certification': [ ...every gate with verdict... ],
        'provenance': {'scripts': ['SOP-v1'], 'agent': '<model name>',
                       'lib_rev': <git rev-parse HEAD at task START>}})
GIT: agents DO NOT commit or push. The repo has concurrent writers, and
a directory-wide `git add` sweeps other agents' in-progress files into
your commit (this happened in the dry run). A dedicated consolidator
(the wake loop) commits and pushes all records periodically -- your
record is durable once write_circuit returns; report its path and stop.
Record `git rev-parse HEAD` in provenance at task START so version skew
from mid-wave infra edits is detectable.

## Escalation ladder (only after steps 1-6 are merged)
- bundle split: ablate each probe SINGLY (cl.proj_hooks([probe])), record
  the per-wing damage profile; dissociated bundles = sub-circuit structure.
- tension scan: while the leaf's hooks are installed, other leaves' member
  dCE is free -- record any leaf whose members IMPROVE by <= -0.3 as a
  tension edge on both records.
- mechanism (THE ACTUAL GOAL -- promoted 2026-08-20): identify what the
  machinery's probes disrupt in named terms. Template scripts to copy:
  qk_writer_decomp.py (decompose an attention score into writer pairs:
  which components' writes the score compares), mlp_ladder_code.py /
  relay_heads.py (is the content token-computable / MLP-written /
  attention-relayed, and by WHICH head), relay_edge_causal.py (delete
  the named carrier vs matched control, verify selectivity). A record
  that names writers + couriers + computation with one causal test
  beats any number of k-curves or surface programs.

## Known traps
- The repo has CONCURRENT writers (other agents, the queue runner).
  Before reporting, `git diff` the specific files you touched -- never
  assume `git status` reflects only your work.
- queue.txt requires ABSOLUTE paths; bare filenames are silently dropped.
- tags are tree-instance-local; identity across instances = member overlap.
- basev/base CE is fit-window; fresh-data claims need fresh rows.
- Do not edit census_lib semantics; add functions if needed.
