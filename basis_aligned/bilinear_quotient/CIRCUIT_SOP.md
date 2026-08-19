# Circuit SOP -- step-by-step procedure for one circuit (swarm-runnable)

Written for a FRESH stateless agent (Sonnet/Opus class) holding only this
file + census_lib. The procedure is verification-driven: every judgment an
agent makes is checked by a computed bar, so a weaker model can fail to
find things but cannot certify junk. GPU steps go through queue.txt
(absolute paths only!) or direct `python -u` if the queue is idle.

Deliverable: one merged circuit record via census_lib.write_circuit(tag, ...)
conforming to CIRCUIT_SCHEMA.md. Do the steps IN ORDER; record every number.

## Identity rule (2026-08-19 revision, from 381)
Leaf member-sets are ONE SAMPLE from a family of valid partitions --
clustering is context-dependent (5% same-data identity across tree
builds). A record's identity is its MACHINERY (probe bundles) +
PROGRAM + CAUSAL PROFILE (sign split, concentration, class profile).
Member-sets are evidence. Certification of a new record requires its
machinery's causal profile to REPLICATE on a disjoint window.

## Step 0 -- claim a leaf
Read circuits/registry.json; pick a tag from census_lib.all_tags() with no
file yet (or the assignment given to you). Never edit another leaf's file.
PARTIAL RECORDS: batch scripts may have pre-written steps 1-2 (causal +
examples, no story/program). That leaf is still claimable: re-run step 1
to verify the recorded numbers reproduce (report both), then continue
from step 3. write_circuit deep-merges: certification entries are
appended (never lost), dict fields merge, so just pass your new fields.

## Step 1 -- causal footprint (GPU, ~15s)
    import census_lib as cl
    d = cl.leaf_ablate(tag)                # dCE under the leaf's own probes
    s = cl.sign_stats(tag, d)
GATE: s['concentration'] >= 3 (the field is provided). If FAILED, record
{'causal': s, 'certification': [gate FAILED]} and STOP -- the leaf is not
locally selective; do not write a story for it.

## Step 2 -- examples (CPU, instant)
    exs = cl.examples(tag, d)              # mechanical: top-3 + 3 random
Record verbatim. NEVER swap examples for prettier ones.

## Step 3 -- program (CPU, ~15s)
    p = cl.leaf_program(tag)               # doc-disjoint heldout + null
PASS if p['bacc'] >= 0.75 and p['null'] <= 0.6. Record either way.
If PASS: append the program to features.json as circ_<tag> (kind expr,
provenance "SOP step 3", cert f"heldout {p['bacc']}") so later circuits can
compose on it. Name collision = someone else did this leaf; STOP and check.

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
- mechanism (bigram fold, interchange/DAS): design-heavy; leave for the
  consolidation model unless you have a template script.

## Known traps
- The repo has CONCURRENT writers (other agents, the queue runner).
  Before reporting, `git diff` the specific files you touched -- never
  assume `git status` reflects only your work.
- queue.txt requires ABSOLUTE paths; bare filenames are silently dropped.
- tags are tree-instance-local; identity across instances = member overlap.
- basev/base CE is fit-window; fresh-data claims need fresh rows.
- Do not edit census_lib semantics; add functions if needed.
