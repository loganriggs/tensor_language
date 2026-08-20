# Circuit SOP -- step-by-step procedure for one circuit (swarm-runnable)
# v2 (2026-08-20): DIVERSE TREE + MECHANISM-FIRST. Supersedes v1.

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

## Step 1 -- causal footprint (GPU, ~45s on the diverse grid)
    import census_lib as cl
    d = cl.leaf_ablate(tag)                # dCE under the leaf's own probes
    s = cl.sign_stats(tag, d)
GATE: s['concentration'] >= 3 (the field is provided). If FAILED, record
{'causal': s, 'certification': [gate FAILED]} and STOP -- the leaf is not
locally selective; do not write a story for it.

## Step 2 -- examples (CPU, instant)
    exs = cl.examples(tag, d)              # mechanical: top-3 + 3 random
Record verbatim. NEVER swap examples for prettier ones.

## Step 3 -- program (CPU, ~15s) [DEMOTED 2026-08-20]
    p = cl.leaf_program(tag)               # heldout + null
On the diverse tree use docid-parity splits (rows of one document are
adjacent; row parity leaks -- see sop_program_batch.py for the pattern).
Record bacc/null either way, but a PASS is a convenience label, NOT a
mechanism grade: surface programs passed only 1/72 on the diverse tree,
and per the user standard they are description, not mechanism. Do not
stop here on a FAIL; the mechanism steps below are the actual work.

## Step 4 -- story, written blind-ish
Look ONLY at exs + s (not at other circuits' stories). Write <=25 words:
what the members have in common + what the machinery pushes. If s shows a
two-signed split (minority_share >= 0.15), the story MUST say what the
push is and where it is wrong -- "helps X" alone is incomplete.

## Step 5 -- red-team your own story (CPU)
Take the 3 RANDOM examples. For each, test the story's CAUSAL claim:
given the example's context and dCE sign, does the story correctly
predict whether the machinery helps or hurts there? (All examples are
members by construction -- membership itself is not the test.)
Count hits. <=1/3 -> mark story 'weak', keep it, flag for revision.

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
