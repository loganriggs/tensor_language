# Directory index (post-cleanup 2026-08-19)

## Top level = live surface only
- **Docs (active, standing loop writes here):** BILIN18_CONNECTION.md
  (master record, numbered sections), RESULTS.md (appendices),
  CIRCUITS.md (the standing loop), BENCHMARK.md + BENCHMARK_BACKLOG.md,
  CIRCUIT_SOP.md (swarm procedure), CIRCUIT_SCHEMA.md (record format),
  LESSONS.md (distilled rules w/ examples), MANIFEST.md (script ->
  results -> sections map), PREREGISTRATION.md (larger-checkpoint
  bundle, frozen), THEORY.md / HARNESS.md / LAYER_PROGRAM.md (early
  program docs).
- **Toolchain (imported / rerun; never move):** census_lib.py (THE
  api -- state, sweeps, hooks, features, rule search, registry,
  canonical head/fresh/ioi helpers), bilin18_joint_removal.py (model
  loader: fwd, orth, m, FW, DEV), circuit_dictionary.py (10-class
  labeler), make_circuit_viewer.py -> circuits.html,
  make_benchmark_figs.py / make_arc_explainer.py -> figures,
  b_common.py, bilin18_canary2.py (queue idle canary).
- **Data:** *_results.json stay top-level (hundreds of scripts write/
  read PT+'<name>_results.json'; migrating them breaks reruns -- defer
  until after the swarm weekend). census_state.pt (cached 212-row
  census tree), circuit_tree4_packs.json (118-leaf packs),
  features.json (compositional feature registry, append-only via
  census_lib.register_feature), das_basis.pt.
- **Registry:** circuits/ (one JSON per circuit + registry.json; write
  ONLY via census_lib.write_circuit). Viewer: circuits.html.
- **Queue infra:** queue.txt (ABSOLUTE paths), backlog.txt, runlogs/
  (_completed.txt = ledger). Runner/feeder: /opt/supervisor-scripts/
  bqrunner.sh, bqfeeder.sh.
- **Pages (published artifacts):** bilin18_report.html (main report),
  circuit_r001_explainer.html, circuit_r310_explainer.html,
  circuits.html.
- **Transient top-level .py:** ONLY scripts currently queued/running.
  After completion + writeup, a wake moves them into experiments/.

## experiments/ -- completed experiment scripts by arc
  assembly/          component stand-ins, assembled_v*, dictionaries,
                     fold tables, absorbers, frontier configs
  early_structure/   a*_ / b*_ series: front-of-model exact structure
  structure_mapping/ bilin18_* wiring era: bus, blind-*, junctions,
                     handoffs, families, content/constraint tests
  census/            damage-tree census, packs, explainer batches,
                     SOP populate, dependency graph
  motifs/            attention motif census, OV, pattern dictionaries,
                     head_lowrank
  mechanisms/        mechanism bootstraps, program/description ladders
                     (compositional_*, fold_*, transported, pairs)
  gating/            gated assemblies, deploy gates, probe gates,
                     slack harvest
  interventions/     interchange, DAS, projection-law, bundle splits
  ioi/               IOI task-circuit arc
  redteam/           fake batteries, blind quizzes, red-team audits
  crossmodel/        bilin12 / swiglu18 / sqrd12 comparisons
  infra/             harness checks, prereg runners, fig utils

## Conventions
1. Find anything: MANIFEST.md maps script -> arc -> results json ->
   BILIN18_CONNECTION sections. Regenerate: python experiments/infra/
   (see make_manifest note in MANIFEST.md header).
2. New scripts import census_lib; re-implementing head recompute,
   fresh rows, IOI prompts, mean-head hooks, feature libraries, or
   rule search is a cleanup violation (they live in census_lib as
   canonical copies).
3. Moved scripts still run (they sys.path-insert this directory
   absolutely and write results to PT). Transform-lineage: read
   sources from their experiments/<arc>/ path per MANIFEST.
4. Registered predictions in docstrings; results written to
   PT+'<name>_results.json'; every run writes a runlog and a
   _completed.txt line.
