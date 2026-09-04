# R593 independent post-execution audit: valid invalid terminal, numerically ill-posed hook predicate

**Audit time:** 2026-09-04 04:25 UTC

**Managed run:** 2026-09-04 04:17:59--04:18:17 UTC

**Scope:** CPU-only, read-only inspection of preserved R593 diagnostic, receipt, evidence, source, and managed-run
records. I did not import Torch, open a model or checkpoint, use CUDA/GPU, inspect or alter the queue, rerun any model
call, delete evidence, or read outcome namespaces from another rung.

## Verdict

R593 is terminally **instrument-invalid**, not a scientific null or hold. The recorded predicate is exactly
`centered_hook_delta_failed`. The producer compared the receipt-bound `hook_deltas.npy` and
`planned_hook_deltas.npy` arrays and obtained exactly `2.288818359375e-05`, above the frozen absolute tolerance
`1e-5`. It therefore stopped after 57 FIT calls and correctly published only the invalid namespace.

That invalid classification must stand. A separate reconstruction from the raw first-layer pre-hook write and planned
delta confirms that the runtime's transient check really did exceed `1e-5`: all 32 layer-5 rows fail, with per-row
maximum errors from `2.7179718017578125e-05` through `6.103515625e-05`.

There is also a narrower diagnostic defect. The saved `hook_deltas.npy` is not the measured hook change. The runtime
initializes it from the planned array and, when its transient aggregate comparison fails, adds `2e-5` to one arbitrary
coordinate as a sentinel. The published `2.288818359375e-05` is therefore the largest FP32-rounded sentinel-minus-plan
difference, not the maximum physical planned-versus-applied discrepancy. The pass/fail branch is corroborated; the
headline scalar is not a direct measurement of the quantity its name implies.

No selector/content factorization score is licensed. FIT did not finish and was not scientifically scored. SELECT,
FINAL, and OOD remained closed. The registered scientific predictions, bars, nulls, and claim boundary remain frozen
and unresolved.

## Managed execution and namespace closure

The managed runner records

```text
[bqrunner] 04:17:59 running execute_induction_centered_fixed_geometry_rung593
[bqrunner] 04:18:17 execute_induction_centered_fixed_geometry_rung593 exit=0
```

The 61-byte per-run log is exactly

```json
{
  "model_forwards": 57,
  "status": "invalid_diagnostic"
}
```

The zero exit status means that the adapter successfully recognized and published the fail-closed diagnostic; it is
not a successful scientific result. At audit time the three normal namespaces
`induction_centered_fixed_geometry_rung593_{results.json,receipt.json,evidence/}` are absent. Exactly the three invalid
namespaces are present. The diagnostic says `final_opened=false`, `ood_opened=false`, zero backwards, and no weight
updates. SELECT closure follows independently from all 57 prefix records having phase `FIT`, the absence of a SELECT
evidence directory, and termination 582 calls before the complete 639-call FIT manifest.

## Receipt and provenance verification

I recomputed SHA-256 from the preserved bytes, without importing the producer or runtime.

| object | recomputed SHA-256 | verdict |
|---|---|---|
| invalid diagnostic | `cf8887eea206fc6139ebb5eb4fea7a7b3daacd445c8935991753cfe3011324f8` | equals receipt |
| `call_prefix.jsonl` | `299d99658818a214220e100d53cca5b69a90a2cf118e1710ab8d23289060c97e` | equals diagnostic and receipt |
| producer | `193013a0c0cf1bec19be4843dee751c355d56f69fbf2d761df57baaa86c6024a` | equals diagnostic/receipt and approved commit |
| runtime | `768c0ed002f107c7549070a0c162552a0e1825ed3de411ff85987a79a8165777` | equals diagnostic/receipt and approved commit |
| sealed-memfd adapter | `1333608efcdb2c1662fe9024aa034d1a805fe342dabc2cda1fdd475f491bb9c5` | equals diagnostic/receipt and approved commit |
| R593 repair amendment | `df0ceebf57818534a9b4ac5de4cd82ca64f2c1228cdfd476e350e62e5707729c` | equals source closure and approved commit |

All four code/document hashes also equal the Git blobs in different-agent-approved candidate
`e2663f0a6fa4e08bb18ba7bc37ef084de48c914d`; the current worktree has no changes to those files. Every one of the 19
entries in `provenance.source_sha256` matches the current source byte-for-byte. The recorded checkpoint hash
`680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3` equals the runtime constant and the approved
pre-execution lineage. In accordance with this audit's no-checkpoint boundary, I did not reopen the checkpoint to hash
it independently.

The receipt names exactly 31 evidence files totaling 727,259,821 bytes. The actual file set is identical: there are no
missing or extra files, and every byte length and SHA-256 matches. I also independently reconstructed the canonical
slice hashing scheme—canonical JSON descriptor plus newline plus C-order slice bytes—and verified all 594 slice
descriptors in the fsynced ledger with zero mismatches.

## Exact prefix and canonical written bounds

The prefix contains exactly these 57 sequential manifest records:

1. manifest indices 0--53: `FIT:endpoint:0000` through `FIT:endpoint:0053`, each batch 32;
2. index 54: `FIT:directed:0000:native`;
3. index 55: `FIT:directed:0000:replay`;
4. index 56: `FIT:directed:0000:score`.

`payload` and `joint` were never called. The first 54 records are canonical slices; the last three are the raw current
chunk. The canonical ledger has exactly 54 records, indices 0--53, and eleven contiguous slices per record:
`[0,32]`, ..., `[1696,1728]`. Its call IDs exactly equal the first 54 prefix IDs.

The diagnostic and receipt agree on the exact written bounds:

```json
{
  "phase": "FIT",
  "endpoint_axis0": [0, 1728],
  "directed_axis0": [0, 0],
  "ledger_records": 54
}
```

Every one of the eleven canonical endpoint arrays has axis 0 exactly 1,728. The endpoint-token width is 30, logit
width 50,304, there are four sites and two roles, and the three decomposition arrays are float64 as amended. A
directed axis bound of zero is correct: canonical directed ingestion occurs only after all five calls in a chunk, and
this chunk stopped on `score` before `payload` and `joint`. The three completed directed calls remain raw and are all
receipt-bound.

The repaired R593 endpoint instruments did work before the later failure. The canonical support array has exactly
5,760 true and 8,064 false entries, matching the frozen FIT census; all 54 per-call support hashes and counts match.
Across all 1,728 endpoints, the independently contracted float64 equality-plus-complement reconstruction has maximum
error `1.7053025658242404e-13`, and the independent full-write reconstruction is bitwise equal to the saved native
write. The directed-native call gives `1.1368683772161603e-13` and bitwise equality respectively. The zero-replay
planned and actual hook arrays are both bitwise zero, and native versus replay logits are bitwise equal. Thus the run
passed the repaired support and decomposition gates and failed at the first score intervention exactly where the
prefix says it did.

## What the planned and saved hook arrays show

Both score arrays have shape `[32,4,1152]`, dtype float32, with site order
`(L5H5,L7H3,L8H3,L8H4)`. They differ in exactly 55 coordinates, all at residual coordinate zero:

| sentinel site | differing rows | interpretation from code path |
|---|---:|---|
| `L5H5` | 32 | layer-5 aggregate comparison failed on every row |
| `L7H3` | 7 | layer-7 aggregate comparison failed on seven rows |
| `L8H3` | 16 | layer-8 aggregate comparison failed on sixteen rows; `L8H3+L8H4` were applied jointly |
| `L8H4` | 0 | same layer-8 transaction; the sentinel is written only into the first same-layer site |

The 55 nonzero saved differences take six FP32 values from `1.9073486328125e-05` through
`2.288818359375e-05`. This pattern follows exactly from the runtime, not from a physical per-site observation:

```python
actual = torch.zeros_like(planned_gpu)
...
observed = modified[local, query].float() - before
for index in indices:
    actual[local, index] = planned_gpu[local, index]
if max_abs(observed - total) > 1e-5:
    actual[local, indices[0], 0] += 2e-5
```

The producer later evaluates `max(abs(actual - planned))`. At planned values near 83 or 123, adding the literal
`2e-5` in float32 moves three ULPs, producing exactly `2.288818359375e-05`. Thus the saved arrays are a lossy boolean
encoding of which row/layer branches failed. They neither preserve `observed` nor its maximum error, and their
per-site shape is especially misleading for the two layer-8 sites, which are summed and applied in one transaction.

## Mathematical and numerical cause

For a float32 pre-hook write coordinate `b` and float32 planned layer-total `d`, the runtime computes

\[
m=\operatorname{fl}_{32}(b+d),\qquad
\widehat d=\operatorname{fl}_{32}(m-b),
\]

then asks for `|d_hat-d| <= 1e-5`. This is not merely a check that the hook executed the requested floating-point add.
It tries to recover a small delta by subtracting two much larger float32 numbers and holds the recovered value to an
absolute tolerance smaller than the output grid at observed write magnitudes.

The earliest intervention is layer 5, so its `before` value is reconstructible from the raw directed-native write;
there is no preceding intervention that could change its state. A concrete receipt-bound coordinate is score row 24,
residual coordinate 645:

| quantity | value |
|---|---:|
| `b` | `1447.919189453125` |
| planned `d` | `75.30926513671875` |
| correctly rounded float32 `m` | `1523.228515625` |
| recovered `d_hat` | `75.309326171875` |
| `d_hat-d` | `6.103515625e-05` |
| FP32 ULP at `m` | `0.0001220703125` |

The error is exactly half an ULP at the modified write and more than six times the frozen tolerance. Across layer 5,
the native write reaches absolute magnitude `1690.3765869140625`, while planned score deltas reach
`99.97101593017578`; all 32 rows necessarily trip this particular absolute comparison somewhere. This is expected
representational rounding, not evidence that the assignment, site, sign, or centered-factor formula was wrong.

Later-layer physical errors cannot be reconstructed exactly from the separate native call because earlier score
interventions alter their inputs. The saved sentinels prove that their transient comparisons branched 7 and 16 times,
but do not preserve the compared values. No later-layer magnitude is needed for the verdict because the first-layer raw
reconstruction already establishes a genuine frozen-threshold breach.

## Was the predicate evaluated fairly?

There are two distinct answers.

1. **The terminal was fair under the frozen contract.** The registered predicate required the applied centered change
   to equal the planned change within absolute `1e-5`. The runtime evaluated that condition before inserting its
   sentinel, the branch fired, and receipt-bound layer-5 inputs independently reproduce errors above the threshold.
   R593 therefore cannot be reinterpreted, promoted, or scored.
2. **The instrument definition and evidence representation were not numerically fair to a correct FP32 add.** At the
   observed scale, `1e-5` is below half an FP32 ULP. A correctly rounded application can fail solely because
   `fl(fl(b+d)-b)` cannot equal `d` that closely. Moreover, the named `hook_deltas.npy` stores planned values plus a
   sentinel rather than actual deltas, so the published maximum overstates or understates the physical maximum
   depending on the planned coordinate's ULP. It is valid evidence of an instrument failure, but not a calibrated
   measurement of hook error.

This is an instrument-design outcome. It provides no evidence for or against the scientific selector/content
factorization hypotheses.

## Narrow prospective repair boundary

Do not rerun or amend R593 in place. Preserve its invalid diagnostic, receipt, evidence, and unresolved nulls. A fresh
successor with new create-only namespaces may change only the hook-fidelity instrument, prospectively and after
independent review:

1. Preserve the same authority rows, machine arms, centered bilinear formulas, scientific scores, scientific bars,
   bootstrap, FIT-to-SELECT rule, FINAL/OOD closure, and null hypotheses.
2. At each intervention layer, save receipt-bound FP32 `before`, theoretical planned layer-total, and actual FP32
   `after` arrays. Derive `actual_layer_delta = float64(after)-float64(before)` from those exact stored values; do not
   synthesize or mutate an “actual” array as a flag.
3. Validate application against an independently constructed correctly rounded target,
   `round32(float64(before)+float64(planned_total))`, preferably by exact FP32 bits. This detects a missing, wrong-site,
   wrong-sign, or corrupted add without requiring an unrepresentable delta.
4. Report `actual_layer_delta-planned_total` as the FP32 representability residual and bind it in evidence, but judge it
   against a precomputed IEEE-754 rounding envelope based on the local ULP, not the impossible global `1e-5` absolute
   bound. Treat the two layer-8 sites as one aggregate transaction unless the implementation applies and observes them
   separately.
5. Add observed-scale model-free fixtures around writes of magnitude 1,000--1,700, including the exact coordinate
   above, plus planted skipped-site, wrong-sign, and one-ULP post-state corruptions. The correctly rounded add must pass;
   each planted implementation error must fail.

This repairs only whether the hook implementation is faithfully observed. It does not weaken or alter the scientific
hypotheses. It also prevents a future diagnostic from presenting a sentinel amplitude as a measured causal quantity.

## Scientific disposition

- **Instrument:** invalid, conclusively.
- **Scientific outcome:** unopened; neither held nor null.
- **Partial score logits:** diagnostic evidence only and prohibited from selecting tasks, thresholds, directions, or
  later claims.
- **R593 rerun authority:** none.
- **Frozen nulls:** preserved unchanged for a separately preregistered successor, if one is authorized.
