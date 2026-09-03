# Circuit record schema (v1) -- decide the format BEFORE 1k circuits exist

One JSON file per circuit in `circuits/`, indexed by `circuits/registry.json`.
Write/merge ONLY through `census_lib.write_circuit(tag, updates)` -- it keeps
the registry in sync. Fields are additive; never repurpose a field (bump
schema_version instead).

```json
{
  "schema_version": 1,
  "tag": "r.0.0.1",                 // tree-instance-local name
  "tree": {"instance": "212row-v1", "n_rows": 212},
  // IDENTITY IS MEMBER OVERLAP, NOT TAG: tags do not transfer across tree
  // rebuilds. Cross-instance matching = Jaccard on member sets.
  "components": ["('pca','m0','r.0.0',(12,16))", ...],
  "members": {"n": 184, "indices": [...]},      // census-grid flat indices
  "base_ce": {"member_mean": 15.2, "frac_lt3": 0.07},
  "causal": {                        // from leaf_ablate + sign_stats
    "abs_dce_members": 2.674, "abs_dce_offslice": 0.348,
    "dce_pos": 1.007, "dce_neg": -2.734, "n_pos": 96, "n_neg": 88,
    "minority_share": 0.478
  },
  "certification": [                 // one entry per test EVER run; append-only
    {"test": "selectivity", "bar": ">=2x matched controls",
     "value": 4.1, "verdict": "HELD", "source": "census_redteam3", "date": "2026-08-19"}
  ],
  "story": {
    "blind_name": "line-break policy in list layouts",
    "name_score": 6,                 // true_pos from the blind quiz, if taken
    "program": [["class_newline"]], "program_bacc": 0.80, "program_null": 0.49,
    "mechanism_level": "none|surface|unigram|bigram|induction",
    "mechanism": {"trigger_top": [...], "lift": 35.5}   // if induction-grade
  },
  "relations": {
    "parent": "r.0.0", "siblings_shared": {"r.0.0.0": 0.71},
    "tension": [{"tag": "r.0.0.3", "evidence": "ablating this leaf's
                 machinery IMPROVES that leaf's members by -0.4", "value": -0.4}]
  },
  "examples": {"top": [...], "random": [...],
               "rule": "top-3 by |score| + 3 seed-0 random"},  // MECHANICAL ONLY
  "provenance": {"scripts": ["circuit_explainer2.py"], "sections": ["344"]}
}
```

Hard rules:
1. Examples are mechanically selected (fixed rule recorded in the record).
   Cherry-picked examples are a schema violation.
2. Every number carries its source (script/results file) in provenance.
3. Verdicts are verbatim HELD/FAILED against pre-registered bars; a FAILED
   stays in the record forever.
4. Sign-mixedness fields are mandatory once causal data exists: a circuit
   is a two-signed policy by default in a bilinear model; records that only
   say "damage" are incomplete.
5. Tension edges (anti-correlated circuit pairs) are first-class relations,
   discovered whenever one circuit's ablation improves another's members.

# Feature registry (features.json)

The compositional variable store: how downstream searches FIND what upstream
work certified. `census_lib.surface_features()` loads it; shifted copies
(prev1_/prev2_) are generated automatically for every entry.

```json
{"features": {
  "class_newline":  {"kind": "expr", "expr": "L0['is_newline']",
                     "provenance": "circuit_dictionary CLS", "cert": "corpus-general"},
  "circ_r_3_0":     {"kind": "expr", "expr": "<program mask expr>",
                     "provenance": "cl2 round 1", "cert": "heldout 0.77"},
  "fold_m0_d3":     {"kind": "expr", "expr": "<fold score>=median>",
                     "provenance": "fold_basis", "cert": "weights-derived"}
}}
```
Append-only; name collisions are an error, not an overwrite. `kind: expr`
entries are evaluated with `torch, F, flat, tok2d, roll, L0, d1` in scope.

# Circuit evidence schema (v2)

Version 2 adds behavior circuits and append-only causal evidence without changing any
version-1 census record. The authoritative object remains one tagged JSON file in
`circuits/`; `registry.json`, `CIRCUITS_INDEX.md`, `DOSSIER.md`, campaign reports, and
`REPERTOIRE.json` are generated views or historical snapshots.

Behavior records use namespaced tags such as `task.bracket.pending_opener`. They do not
claim a census-tree identity. A relation to a census slice is evidence that must name an
identity/member-map artifact; matching tag text is never sufficient.

Required top-level fields are:

```json
{
  "schema_version": 2,
  "tag": "task.bracket.pending_opener",
  "identity": {
    "kind": "census_slice|behavior_circuit|shared_subroutine",
    "instance": null,
    "identity_artifact_id": "task_definition",
    "aliases": []
  },
  "claims": [],
  "split_plans": [],
  "evidence_events": [],
  "translations": [],
  "artifacts": {}
}
```

Each claim contains a stable `claim_id`, increasing `revision`, a `status`, the declared
causal variable, counterfactual families, candidate sites, and artifact/event IDs.
Allowed statuses are `proposed`, `specified`, `site_live`, `activation_identified`,
`weights_translated`, `adopted`, `rejected`, and `superseded`. Revisions are append-only;
do not mutate an old claim after evidence has used it.

The causal variable explicitly states its value domain, what information is read, the
operation/composition, the quantity written, and the behavioral endpoint. A
counterfactual family records:

- role: `interchange`, `necessity`, or `invariance`;
- the facts it changes and holds fixed;
- builder, controls, and split-plan IDs; and
- status: `proposed`, `frozen`, `validated`, or `failed`.

An interchange changes the declared variable and correct answer. A necessity test
changes/removes the variable while keeping the original answer fixed. An invariance test
changes nuisance information while preserving both. These roles are not interchangeable.

Every evidence event is immutable and includes a controlled enum verdict. Its
`test_type` is one of capability, full-swap ceiling, DAS interchange, cross-family
transfer, removal, invariance, composition, OOD, seed stability, compiled equivalence,
or null/control. Its `stage` is `preregistered`, `complete`, or `invalid`; its verdict is
`held`, `failed`, `null`, `inconclusive`, or `invalid`; and failures distinguish
`scientific_null`, `insufficient_power`, `invalid_instrument`, and
`implementation_failure`. Historical free-form certifications remain verbatim in v1;
do not rewrite them into fake precision.

`design_key` hashes the claim/variable, family change-and-hold contract, controls, site,
test type, and metric/bar. `execution_key` additionally binds the split, seed,
checkpoint, preregistration, and result hashes. Renaming an experiment therefore cannot
silently repeat it. A deliberate successor names `supersedes_event_id`; a replication
names `replicates_event_id` and must have a distinct execution key.

`make_circuit_experiment_index.py` turns these hashes into the generated
`CIRCUIT_EXPERIMENT_INDEX.md` and `circuits/experiment_index.json` views. Check that view
before claiming new work: it lists the currently open preregistrations, exact execution-key
duplicates, and repeated scientific protocols that lack an explicit supersession or
replication link. Its `protocol_key` deliberately ignores claim revision and execution
details, so changing a filename, seed, or split cannot disguise a repeated question.

Artifacts use repository-relative paths and SHA-256 hashes. `frozen` requires a hash;
unavailable historical inputs are `legacy_unhashed`, never guessed. Split plans group the
underlying document/template/entity across all families so one unit cannot appear in FIT
for one family and FINAL_TEST for another.

Write v2 behavior records with `census_lib.write_behavior_circuit()` and append results
with `census_lib.append_evidence_event()`. Both are implemented in the lightweight
`circuit_registry_v2.py`, so registry maintenance itself never loads the model or GPU.
