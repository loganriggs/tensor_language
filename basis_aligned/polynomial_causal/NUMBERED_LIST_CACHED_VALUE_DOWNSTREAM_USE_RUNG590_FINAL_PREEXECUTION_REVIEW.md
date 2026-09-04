# R590 final independent exact-byte pre-execution review

Date: 2026-09-04 UTC  
Reviewed commit: `3eb52938b3641f067d8f8eb9e654f461cbd61ad0`  
Verdict: **APPROVED for the exact reviewed bytes**

This was a CPU-only, model-free, outcome-blind review. I loaded the candidate
from immutable Git blobs and confirmed that the working files used for the
model-free tests were byte-identical. I did not load the model, open CUDA, use a
GPU, enqueue a job, or open an R584 or R590 scientific result. The R590 result,
receipt, and evidence namespaces were absent before and after review.

## Exact approved packet

- producer: `c38654506f36fcf111f3a34f356893240548c3cfbf4eded58efb04d31fdb2e36`
- owner test: `49f6f7a998bfb69331c36391f5d3c16d9b702c1fd60b4da5b09c920f3832e5b0`
- dry run: `3ebada19f74906ba3e7cd1637fc1cd6cdff84936124dee01cb058875432d3b95`
- managed adapter: `c525cad078935ef0552214fba13c16a5d56483c8e3048bbec4d6ab9ef3f17885`
- adapter test: `17d51c8e7df667ecf1cc146b1ac00e34f658e97759ee149ddb254f7d9317f07e`
- prospective contract note: `dae72b4aee35030f31ce42674d9535d6bff6c857b9beb8633a8ac809edaf031b`
- repaired immutable-closure regression: `0a1cefe529971e3533f8750f47f665163ebdbf5c4782c510c0a524edb9d806ba`
- this independent planted test: `de7b31ecfe997e8d956dcc1a4ede9505b765456e980f806053726a2d523e1d91`

Approval is conditional on rechecking these exact hashes at dispatch. A change
to any approved byte requires a new review.

## Independent reconstruction

The outcome-free R582 authority regenerates exactly 1,440 rows: 576 FIT and 288
each SELECT, FINAL_TEST, and OOD. Every split has 36
`condition × representation × source_level` cells; FIT has 16 rows per cell and
each later split has eight. The opened panel must equal the authority's ordered
row IDs exactly. Internally rehashed reorder, missing-row, duplicate,
replacement, cross-split-borrowing, and cell-shrink attacks fail.

Independent length counting gives 27 FIT batches, 14 SELECT batches, 20
FIT-null batches, and 10 SELECT-null batches at batch size at most 24. The
literal conditional paths therefore reconstruct as:

- no provisional FIT candidate: `2×27 + 1 + 12×27 = 379` forwards;
- provisional candidate but failed FIT nulls: `379 + 2×20 = 419`;
- selected candidate and SELECT: `419 + (2×14 + 1 + 3×14 + 2×10) = 510`.

The saved manifest contains exactly these 510 possible call IDs: 82 trajectory,
two native-smoke, 366 real component-suffix, and 60 null-suffix calls. Every
call records its exact row IDs, batch size, token length, split, guard, shape
mode, and checkpoint/model-structure validation. FIT-first precedence is
preserved; FINAL_TEST and OOD cannot open; backwards and weight updates remain
zero.

## Scientific and evidence contract

R590 preserves the frozen R582/R584 intervention. For the cached-value contrast
`delta = x_with - x_without` at MLP sites 8, 10, 12, and 14, it removes the
gauge-invariant background-cross term `C`, contrast-self term `Q`, or `C+Q`.
The first candidate in the frozen 12-term FIT order that passes all real gates
is provisional. Its two active deterministic nulls must pass before SELECT can
open. If selected, SELECT scores the three terms at that one site using FIT-
frozen scales. This review changes no row, term, threshold, null, bootstrap,
selection order, or terminal meaning.

The producer now derives every report, bootstrap trace, interaction, predicate,
selected component, split opening, forward count, decision, and scalar
`next_step` from primitive evidence. It verifies exact row/arm/site identity,
semantic token coordinates, recipient capture joins, endpoint/logit/CE and RMS
sufficient-statistic identities, deterministic null donors, active-null norms,
and the fixed conservative null comparison. Planted held, no-candidate, and
active-null-failure packages reconstruct the 510, 379, and 419 paths
respectively.

A value at the next representable float above the frozen `1e-10` native-replay
limit raises `UnretainedInstrumentError` before any publishable scientific
terminal. Non-finite or malformed evidence also fails closed. A coordinated
rewrite of result decision and next step, followed by recomputing the outer
receipt hashes, still fails because the result is rederived from evidence.

Evidence, result, and receipt are written in a unique same-filesystem stage,
validated as strict finite JSON, and mutually hash-bound. Publication atomically
renames evidence, result, and receipt in that order, with receipt last as the
commit marker. Injected crashes after each final rename roll all moved paths
back into the recognizable stage; the managed preflight quarantines only the
recognized incomplete package and refuses complete or arbitrary occupied
namespaces.

## Immutable execution closure

The two previous R590 blocks are closed. Before importing project code, the
managed adapter captures and verifies the producer plus this complete executable
project closure:

- R584 producer: `50609756d97de2f13f717774f13d72b1c743f38a172375e9b08efc2b055336c7`;
- R588 auditor/scorer: `b4acebb23bff71c7dc11beec95ff83f5490a86971787bce5930351cfb4572115`;
- generic result contract: `af8fb9557dcb77e038319b0fffa919927f3925497a0edafe27fc951125dfb272`;
- model facade: `b62947f772c807259890a9d09dfcbe5e91ad339a0bffa867ab99177fde4c728c`;
- R576: `91db3a2a9210aef915ce2e4f0a62253274e0b5470cbfaa05a95d50a3c0cf985a`;
- R573: `5723e42e2a5f72a4ddab7a20b631e18e0b6d28875ff53f3db2d37d1845d6e076`;
- R582 helper: `b0d99eeeef834091cf9ddfe77b58682f0e9a7e101e143a18570808dacb57bc1c`;
- `jacclust/__init__.py`: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`;
- model definition: `49ecdbd6c060ff5b3e57f3134d87ba32841390c891c42e6ae23b71d8627612b2`.

All are compiled from captured bytes. R588's recursive helper lookup is bound to
the captured R582 module rather than reopening a path. R590 provenance and its
two AST call-site censuses read the captured snapshot, so swapping a pathname
after capture cannot change either execution or the digest later claimed by a
result. A read trap confirms that the dry-run call graph never opens R576/R579
outcomes or calls R588's broader outcome authority loader.

## Gates run

- candidate, adapter, repaired-closure, and independent suites: **51 passed**;
- new independent planted suite alone: **7 passed**;
- Python compilation: pass;
- repository static gate, producer and adapter: `GATE: PASS`;
- advisory preflight: `no findings`;
- managed no-model dry run: pass, 510 possible calls, zero actual forwards,
  zero backwards, no updates;
- `ops/test_fast.py`: 0 failures.

The old `cf00f555d` adversarial file remains intentionally bound to that obsolete
blocked commit and is not a candidate acceptance test; its exactness, support,
terminal-rewrite, and dependency-closure attacks are preserved in the repaired
and new independent suites.

## Scope of approval

This approval licenses one managed execution of these exact bytes under the
already frozen R590 contract. It does not promote the historical R584 result,
open FINAL_TEST/OOD, authorize threshold changes, or establish that any
downstream component is scientifically held. The future R590 evidence package
must still derive its own held/null decision and receive the registered
post-execution audit.

