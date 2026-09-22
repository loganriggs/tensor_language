# Frozen larger residual evaluation

2026-09-22 01:04 UTC. Registered before model capture/scoring. Exact five artifact hashes in RESIDUAL_FRESH_CANDIDATES_V1.json; no fitting on new examples. Larger shared-direction fit and CP-parent refit remain separate experiments.

Target: last twoMLPs16->17 purequartic selfterm projected onto the original16fixed writers. Actual normalized x16 inputs; omit residual, attention, bias and cross terms from this polynomial. QR-derived vocabulary geometry is fixed. Tests concern prediction and stability of the candidate computation; no selective causal or full-model adoption claim.

Data:256 distinct FineWeb sample-10BT source documents, first65tokens; scorefirst64,16384states. Exclude all current local FineWeb rowcache first32token excerpts anywhere in a candidate document, earlier direct_tensor_match token-panel first32tokens, and prior document IDs/text hashes. One prefix per document; distinct documents can remain related. Not a guarantee against pretraining or all repository exposures. Store exact exclusion list and identities. Dataset selection uses no candidate responses. No OOD claim from another FineWeb sample.

Opposing predictions and bars, evaluated for BOTH CP seeds:
1. Instrument: repeat4oldprefixes, input/target norm replay<1e-4; finite16384x1152inputs and16384x16targets; all candidate hashes unchanged.
2. Aggregate transfer: new pooledrelativeerror<=1.25times old6.322183414/6.587528291%; feature0-2relativeerrors<=10%each. Failure falsifies adequate transfer even for dominant writers.
3. Feature weakness replication: at least8of12features4-15 retainrelativeerror>30%. Passing is evidence AGAINST uniform component recovery, not a success/adoption gate.
4. Concentration replication: worst10%states carry>=30%total squared error. Report target energy in exactly those states, denominator-stable norm quantiles, maximum and per-prefix errors.

Always report all16feature errors and energy shares, all five frozen candidates, bias and centered residual spectrum. Bootstrap entire prefixes (2000resamples, fixedseed220922) for pooled and feature errors; these are conditional panel uncertainty estimates, not universal generalization certificates. No threshold tuning after seeing results. Never select top contexts as proof of a semantic category. No fitting, no repaired program export.

Capture runner batches4prefixes; CPU dryrun uses actual256x64x1152/16output shapes plus independent projected bilinear teacher identity. Existing fourprefix input/label replay is the decisive native instrumentation check. Save reusable float32inputs/labels (~73MiB) plus token IDs/provenance. Measure capture and scoring times separately. Compare unchanged programs: original384products, CP1536products each, shared1088products each. Evaluation has no added deployment cost; any later correction must be priced independently.

Managed GPU capture only via ops/enqueue.sh after tokens ready. Existing live/queued jobs preserved. CPU scorer follows capture; this preregistration is not a completed result.
