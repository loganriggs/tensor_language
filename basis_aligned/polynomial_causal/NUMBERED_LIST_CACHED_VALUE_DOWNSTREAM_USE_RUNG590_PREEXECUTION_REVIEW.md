# R590 independent pre-execution review

Date: 2026-09-04 UTC  
Reviewed commit: `cf00f555d`  
Verdict: **BLOCKED**

This review was performed from immutable Git blobs before any R590 model run. It
used CPU-only planted evidence. It did not load a model, open CUDA, enqueue work,
or inspect an R584/R590 scientific outcome.

## Exact reviewed bytes

- producer: `74b565fe835ee69a73ed1bdcdc103df3b2f4aa94931796ca1b96a4080639062e`
- owner test: `037c7b7368fd2ca1f2d4656b75fd4e97e96e71c9c4f7679730ab94442fa6cee2`
- dry run: `fb0b65d32be3422440602ae6458a39c357a1c83d5d180d0142f9f453edae3ad9`
- managed adapter: `34899a771279cda55e674df2da3de7cf8321a787b26e986d2601d8bbdd6b3479`
- adapter test: `3eec8628a2b1ef05ef5ff29c107b5db6e958d172d132f8476884379f3983b2fd`
- prospective note: `8b4019b2da24ee8a6acf73cf1cb35b157e3feece713ca9e90698a0801cf15ab5`

## What passed

1. The saved reports, selected component, terminal decision, scalar next step,
   interaction identities, bootstrap traces, and conditional forward count are
   recomputed from primitive evidence. An internally re-hashed change to the
   decision and next step is rejected.
2. Every retained replay and bilinear error is checked against the frozen
   `1e-10` limit. A planted value at the next representable float above `1e-10`
   hard-aborts rather than becoming a scientific null.
3. The staged evidence/result/receipt are strict finite JSON, mutually bound by
   hashes, validated before publication, and published receipt-last. Planted
   crashes after each rename leave no apparently complete final package and a
   recognizable stage for conservative recovery.
4. The phase-local support census is exact: FIT has 576 rows (36 cells times 16)
   and each later split has 288 rows (36 cells times 8). Borrowing, duplication,
   replacement, shrinking, and reordering are rejected even if the attacker
   recomputes the census hashes.
5. The dynamic call-shape manifest contains exactly 510 possible calls. The
   realized paths are exactly 379 (FIT null), 419 (FIT candidate followed by an
   active-null failure), or 510 (FIT plus SELECT), with FINAL_TEST and OOD closed,
   zero backwards, and zero parameter updates.
6. The producer and adapter pass the repository static gate and preflight. The
   managed dry run is model-free and reports 510 possible calls and zero model
   forwards.

## Execution blocker

The managed adapter verifies the R590 producer itself and several data/method
documents before import, but it does **not** verify the producer's executable
dependency closure before importing the producer. The producer imports the R584
runner, R588 auditor, and `result_contract.py` at module scope. R584 then imports
the model facade, R576, R582, and further executable helpers at module scope.
Those files can therefore execute changed top-level code before the producer's
later `validate_authorities()` call notices their hashes.

This is a path-byte race at the reviewed-code-to-managed-dispatch boundary. The
scientific contract inside the exact producer bytes is sound, but the adapter
does not yet guarantee that those are the only executable bytes in force when a
queued no-argument process starts.

Required repair: make the adapter hash-check the full executable transitive
dependency set before importing any of it, then execute verified immutable bytes
(or use an equivalently strong mechanism that cannot re-read changed path bytes
between verification and import). At minimum this includes the frozen R584
runner, R588 auditor, result contract, model facade, R576 runner, R573 runner,
and R582 helper. Re-run this review against the new adapter and adapter-test
hashes. No scientific formula or producer evidence contract needs to change.

## Tests

- Candidate acceptance/adversarial set: `38 passed`.
- Independent review attacks: `3 passed, 1 strict xfailed`; the strict expected
  failure is exactly the missing pre-import executable-dependency pin.
- Static gate: producer and adapter pass.
- Preflight: producer and adapter pass.
- Managed model-free dry run: pass.

The independent planted test is
`test_numbered_list_cached_value_downstream_use_rung590_preexecution_review_adversarial.py`.

