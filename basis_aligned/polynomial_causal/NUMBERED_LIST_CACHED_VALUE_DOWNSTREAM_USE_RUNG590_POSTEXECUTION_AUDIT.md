# R590 independent postexecution audit

Date: 2026-09-04 UTC  
Approved producer commit: `3eb52938b3641f067d8f8eb9e654f461cbd61ad0`  
Verdict: **scientific null independently held; instrument valid; managed wrapper false-failed after publication**

## Boundary

This audit began only after the managed R590 process terminated. It used the
primitive evidence and the exact immutable bytes approved in
`NUMBERED_LIST_CACHED_VALUE_DOWNSTREAM_USE_RUNG590_FINAL_PREEXECUTION_REVIEW.md`.
It loaded no model, opened no CUDA context, made no model call, changed no
weight, used no queue, and did not open FINAL_TEST or OOD.

The managed process published a complete evidence/result/receipt package and
then exited 1 because the adapter unconditionally raised
`R590 verified scientific entry point unexpectedly returned` after the producer
returned normally from its successful publication path. The scientific package
is independently valid. The nonzero managed exit is a real wrapper bug, but it
occurred after the receipt commit marker and is not an instrument-exactness
failure or a reason to rerun the science.

## Exact bytes

Approved execution packet:

- producer: `c38654506f36fcf111f3a34f356893240548c3cfbf4eded58efb04d31fdb2e36`;
- owner test: `49f6f7a998bfb69331c36391f5d3c16d9b702c1fd60b4da5b09c920f3832e5b0`;
- dry run: `3ebada19f74906ba3e7cd1637fc1cd6cdff84936124dee01cb058875432d3b95`;
- managed adapter: `c525cad078935ef0552214fba13c16a5d56483c8e3048bbec4d6ab9ef3f17885`;
- prospective contract: `dae72b4aee35030f31ce42674d9535d6bff6c857b9beb8633a8ac809edaf031b`.

Landed package:

- result: `e868a0e67aa9e4d3251deccbf625b7708ee1f1ce9070afbdfce2357a9f0bed24`;
- evidence: `3025923441b40d7ced3a0d9b8277ade3639d87deb4b8fe2c6a00438c9fcf4815`;
- receipt: `2789d205c311ffaa8401edd761a787b8061a138ad297ab5c7f4b67b45ba3b20d`;
- managed run log: `182554a1824fbd7254893d5df28704fb891fc6f78d22d75519f8192c1b657456`;
- package ID: `f6955310fae86f5b77e1005ee3f571715a72bc5d66bd400b9e437c7126777c92`.

Audit packet:

- independent audit script: `7a0de3fbf1cf431115c98439d76d14c706b45ed2a3a7c4318b43a769444386db`;
- six planted tests: `bc783b213dd89613298f067dcb9658f065fb983a53c889a56153d883d7c38bff`;
- audit JSON: `931c72c2fe7a24b63d073aa0337946c3c1d78e6325b2cbacace941710c0b437f`.

## Independent reconstruction

The audit regenerated the exact 1,440-row authority and required the full
ordered 576-row FIT panel: 36 condition-by-representation-by-source cells with
16 rows each. It validated all 576 native/source-deleted capture records and all
12 real arms, each with 576 rows. It checked token IDs, source and query
positions, source values, answer IDs, arm/site/component identities, endpoint
margin and CE identities, vocabulary-RMS sufficient statistics, and every
arm-to-native-capture join.

The exact FIT instrument checks all passed:

- cached-bus error: `0`;
- native end-to-end smoke error: `0`;
- native replay error: `0`;
- value split: `5.6360e-16`;
- source-head sum: `6.1073e-15`;
- projected cached term: `4.3846e-14`;
- maximum bilinear response reconstruction error: `3.0813e-11`, below the
  frozen `1e-10` limit.

The audit independently reran all 432 group-bootstrap cells at 2,000 resamples,
obtaining trace hash
`6d6d47369f7234ee27b202da6147f6685515d034b00e15fca8c50ede7d7abb57`.
No one of the 12 candidates passed all non-null FIT gates. Therefore there was
no provisional candidate, the two active scientific null interventions were
correctly not run, SELECT stayed closed, and the exact price was the registered
379 forwards. Backwards and parameter updates were zero.

The result and receipt exactly match that reconstruction. The result binds the
12,250,616 evidence bytes; the receipt binds both evidence and result bytes;
all three are canonical finite JSON. There was no incomplete stage left after
the receipt-last publication.

## Scientific result in normal terms

The experiment asked whether the known cached numeric-value signal becomes a
selective “advance this value” computation inside any one of MLPs 8, 10, 12, or
14. At each MLP it removed one of three exact pieces:

- `background_cross`: the cached-value change interacting with the rest of the
  MLP input;
- `contrast_self`: the cached-value change interacting with itself;
- `joint_response`: their exact sum.

All interventions were active. For every candidate, all 12 relation/conflict
activity cells and all six conflict-sign preservation cells passed. This rules
out the explanation that the experiment returned null merely because the edits
were zero or the controls were unavailable.

Some terms substantially damaged successor behavior, but none separated it
cleanly from copy behavior and remained stable everywhere. The strongest target
counts were 10/12 cells for MLP8 background-cross, MLP8 joint, and MLP10
background-cross. The strongest action-gap counts were 9/12 for MLP8 self and
MLP8 joint. But all three MLP8 terms passed 0/12 copy-preservation cells;
the best copy count anywhere was only 5/12 for MLP12 background-cross. No
representation passed the whole gate for any candidate.

Therefore the independently reconstructed scientific terminal is:

`downstream_use_decomposition_null`

This is a valid negative result, not an invalid instrument and not evidence that
the cached-value signal has no effect. It says only that none of these 12 coarse
single-MLP `C`, `Q`, or `C+Q` terms is the selective, reusable successor-use
circuit defined by the preregistration. The established R576 broad carrier
remains; it should not be promoted to a selective R582/R590 component.

## Managed wrapper defect

The producer printed its complete result summary and returned after atomic
publication. The adapter's `dispatch()` then deliberately raised because its
test fixture had encoded successful science as a non-returning function. Thus:

- scientific terminal valid: **yes**;
- instrument invalid: **no**;
- complete receipt-committed package: **yes**;
- managed wrapper clean: **no**, exit code 1 after publication.

The minimal prospective repair is adapter-only: a normal return from the exact
producer after receipt publication must become managed success, not an
exception. This repair must be tested with a returning fake science function.
It does not justify changing the science or rerunning R590, because rerunning
would duplicate an already complete receipt-bound outcome.

## Tests and next gate

The independent audit suite reports `6 passed`. It includes a valid planted
null, an internally consistent result-plus-receipt terminal rewrite, a missing
primitive row with recomputed outer hashes, strict non-finite JSON, exact-byte
tampering, and the observed postpublication wrapper ordering.

The durable next scientific conclusion is the registered null: retain the broad
cached-value carrier and seek a finer decomposition by visible label identity,
successor action, and copy/conflict downstream use rather than repeating these
12 coarse MLP terms. Separately repair the adapter success sentinel before any
future reuse of this execution pattern.

