# MLP4 evidence adapter versus TheseusBench submission

`mlp4_theseus_evidence.json` is a provenance-preserving handoff artifact. It is
not a TheseusBench submission, verifier transcript, or frontier entry.

The sibling `theseus-bench/SPEC.md` imposes two relevant requirements:

1. the harness, rather than a replacement, computes fidelity and complexity;
2. frontier admission requires verifier-backed statistical dominance, with private
   and OOD evaluation handled by the benchmark workflow.

The MLP4 adapter consequently fails closed:

- `artifact_role` identifies it as an evidence adapter;
- `theseus_harness_verified` is always false;
- `self_reported_scores_frontier_admissible` is always false;
- opening the prospective MLP4 held-out file can populate only the held-out lane;
- composite, extraction, removal, and OOD remain unmeasured;
- no candidate can become `frontier_eligible` through this exporter;
- structural tensor spectra and composition bounds are explicitly nonbehavioral;
- signed-square scores are not inherited from the product-factor program.

The proper integration path is to turn a selected decoded program into a
`Replacement` plus `Manifest`, have the Theseus harness independently inventory
its constants and source structure, and retain the resulting verifier transcript.
The five-lane evidence file can accompany that submission as provenance, but cannot
substitute for it.
