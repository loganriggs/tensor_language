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
