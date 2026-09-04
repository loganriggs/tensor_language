# Rung 592 diagnostic-prefix amendment

Date: 2026-09-04 00:51 UTC  
Status: prospective, outcome-blind, CPU-only specification; independent approval required before implementation  
Parent preregistration amendment SHA-256: `5e9fe2bcf41b88c199b5dfab2ba3ec7d0fa8f4b4b2952173c1984391e4d53094`  
Blocking independent review SHA-256: `21bdc310b4798d3ae6d47fc2ed7dfee969afd871bc90db381db634e2c4cae2f5`

## Scope

This amendment closes only the remaining ambiguity in invalid-run evidence. It changes no authority row, split,
site, role, arm, intervention, tensor width, numerical or scientific threshold, bootstrap identity, claim boundary,
complete-phase evidence, or 639/322/961 call price in the parent specification.

The parent allowed an invalid diagnostic after a completed call but described partial evidence using rectangular
four-arm arrays. A failure after `score` but before `payload` therefore had no unique unpadded representation. R592
now uses the exact per-call diagnostic format below. The complete rectangular arrays remain mandatory for every normal
scientific result and are never inferred from diagnostic files.

## Completed-call rule and immediate stopping

A model call is **completed** only when its forward has returned and every observation required for that call has been
materialized in host memory. After each completed call, applicable validity predicates are evaluated in this fixed
order:

1. `nonfinite_observation`;
2. `fixed_width_token_manifest_failed`;
3. `native_full_write_reconstruction_failed`;
4. `native_equality_remainder_reconstruction_failed`;
5. `factor_transport_failed`;
6. `centered_hook_delta_failed`;
7. `directed_native_zero_replay_failed`;
8. `structural_output_identity_failed`.

Predicates not yet evaluable at that call are skipped. If several newly evaluable predicates fail, the first in this
list is the recorded failure. No later call is executed. The failing completed call is included as the last member of
the executed prefix.

If a forward raises, the process is killed, or required observations cannot be materialized safely, that call is not
completed. This is an **unpublishable hard abort**: no R592 result, diagnostic, evidence, or receipt is created. Existing
temporary files are not renamed into the public namespace. A hard abort can be investigated from the managed run log,
but it is not an R592 diagnostic terminal.

## Exact call-prefix diagnostic evidence

The only public invalid namespace is:

```text
induction_centered_fixed_geometry_rung592_invalid_evidence/
induction_centered_fixed_geometry_rung592_invalid_diagnostic.json
induction_centered_fixed_geometry_rung592_invalid_receipt.json
```

For every completed call, including the failing call, the temporary invalid-evidence directory contains one
subdirectory named by its zero-based manifest index and bound call ID:

```text
calls/{manifest_index:04d}_{call_id}/
```

`call_prefix.jsonl` has exactly one line per completed call in manifest order. Each line contains the manifest index,
call ID, phase, call kind, chunk index, machine arm or `null`, token-record ID, token SHA-256, batch size, physical width
30, ordered authority-row IDs, ordered direction IDs when applicable, query positions, and a lexicographically sorted
map from evidence filename to dtype, shape, byte length, and SHA-256. Its last line is the failing completed call. The
receipt independently hashes `call_prefix.jsonl`, lists the same call IDs, and proves they equal an exact prefix of the
frozen phase manifest.

Each call directory uses its actual batch size $b\in\{16,32\}$ and contains no placeholder for an unexecuted call or
arm:

| completed call kind | mandatory raw arrays |
|---|---|
| endpoint capture | `tokens.npy` `[b,30]` int64; `logits.npy` `[b,50257]` float32; `factor_e.npy` `[b,4,2]` float32; `factor_u.npy` `[b,4,2,1152]` float32; `support.npy` `[b,4,2]` bool; the five parent native-write arrays, each `[b,4,1152]` float32 where applicable |
| directed native | `tokens.npy` `[b,30]` int64; `logits.npy` `[b,50257]` float32; `live_e.npy` `[b,4,2]` float32; `live_u.npy` `[b,4,2,1152]` float32; the parent native reconstruction arrays needed by every currently evaluable predicate |
| directed replay, score, payload, or joint | `tokens.npy` `[b,30]` int64; `logits.npy` `[b,50257]` float32; `hook_deltas.npy` `[b,4,1152]` float32; `planned_hook_deltas.npy` `[b,4,1152]` float32 |

Cached endpoint factors needed to reconstruct a directed call are referenced by content hash from already completed
endpoint-call directories; they are not silently copied from a mutable cache. The replay, score, payload, and joint
calls are separate call directories. Thus a prefix ending after score contains exactly the native, replay, and score
directories and has no payload or joint slot to pad.

`invalid_diagnostic.json` contains only provenance, the exact failure predicate, its maximum/error count and affected
array coordinates, the executed-call prefix, and diagnostic status. It contains no cell score, bootstrap interval,
`split_scores`, scientific terminal, held/null statement, or claim about the selector/content factor.

## Nonfinite evidence exception

All arrays in every successful earlier call must be finite. If and only if the terminal predicate is
`nonfinite_observation`, raw arrays in the final failing-call directory may contain IEEE NaN or infinity. The diagnostic
stores a packed boolean `nonfinite_mask.npy` for each affected raw array, with the same shape and C-order, plus exact
nonfinite counts and first lexicographic coordinates. The raw float bytes and masks are hashed without normalization.
No aggregate scientific statistic is computed from those arrays. A nonfinite value in an earlier call, or a mismatch
between a mask and `isfinite` applied to the hashed raw array, makes the whole publication an integrity failure.

This is the sole exception to the parent's `all_observed_values_finite` rule, and it applies only to an invalid
diagnostic namespace that can never be promoted into a normal result.

## Diagnostic auditor and atomic publication

The diagnostic auditor must reconstruct, without loading any normal-result array:

1. that the call list is an exact frozen manifest prefix and ends at the first failing completed call;
2. every token tensor, authority row, direction row, arm label, dtype, shape, byte length, and content hash;
3. every applicable validity predicate from the per-call raw arrays and frozen cached-factor references;
4. the fixed failure precedence and the absence of an earlier failure; and
5. the absence of directories or rectangular slots for unexecuted calls.

Publication remains receipt-last. The producer writes and fsyncs a temporary invalid-evidence tree, diagnostic, and
receipt; atomically renames the evidence directory; atomically renames the diagnostic; and renames the receipt last.
An absent receipt means no recognized diagnostic. Normal R592 result/evidence/receipt paths must remain absent.

For a complete valid FIT or SELECT phase, none of this per-call diagnostic namespace is published. The parent complete
rectangular evidence and normal scientific terminal rules apply unchanged.

## Review and implementation gate

An independent reviewer must plant at least these cases before implementation begins:

- failure after directed native, replay, score, payload, and joint respectively, with no padded later arm;
- a missing or extra call directory and a non-prefix call ID;
- a wrong batch-size-16 terminal chunk shape;
- nonfinite raw bytes with a correct and an incorrect mask;
- a failure whose recorded predicate violates the fixed precedence;
- a hard-abort simulation that leaves no recognized public namespace; and
- an attempt to treat diagnostic per-call evidence as a normal scientific result.

Only a committed APPROVE of this exact amendment permits the R592 implementation to resume. The completed
implementation bytes then still require a different-agent audit before any managed GPU enqueue.
