# Continue the same matched sparse path objective to support stability

11September2026, before continuation. The first pilot's completed arms still change support after four outer cycles, even where their fixed-support gradients converge. Preserve that miss. Do not reinterpret the first pilot as absent structure.

Use completed pilotV2 programs only after its instrument predicate passes and its final artifact is hash-bound. Same full-U source tensor, gain adapter, rank16joint/rank8independent,96edges, seeds and147456fitted-float budget. No initialization restart, text fitting, new penalty or changed comparison threshold.

Use the verified active-edge kernel in coupled_sparse_path_v2.py for fixed-support optimization, and the original all-column helper for exact support selection. CPU value/gradient checks hold within2.59e-16. Recompute each saved initial objective and coefficient writer; require relative replay<=1e-8. Benchmark old/new gradients at the same native points, with synchronized warmups and three timed evaluations.

Per arm allow up to20additional support cycles,60seconds maximum fixed-support CG per cycle. Stop only after two consecutive unchanged selected supports, tangent norm<=1e-7 and capture-relative tangent norm<=1e-4. These are the original convergence bars. Save intermediate program after each arm and compact scalar progress each cycle. If max budget ends unconverged, preserve that and choose a discriminating solver/topology check; do not declare structure absent.

A instrument: pilotA, bound sources/artifact, initial objective+writer replay<=1e-8relative, old/new gradient<=1e-8relative, orthogonality<=1e-9, support selection nondecreasing capture within1e-10, finite outputs and matched costs.
B convergence: all four final arms meet the unchanged criteria above.
C original structural advantage: jointcapture>=1.10independent EACHseed. Interpret as an optimization-controlled comparison only if A/B hold.
D speed: active-edge native gradient evaluation at least1.5x faster EACHjointseed. This is an engineering hypothesis, not circuit evidence.

Also preserve original incidence test (>=8mixededges and>=4reusedfeatures eachsource) descriptively; final edge/node interfaces must use the saved selected support, not an unoptimized reselection. Zero body forwards/text, at most4800fitseconds total with5200second alarm, managedlane1 only. Native sample/uncertainty check may run ahead in its existing queue position; do not reorder it.
