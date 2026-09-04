# Post-hoc audit — numbered-list cached-value downstream readers

Date: 2026-09-04 03:48 UTC

Producer commit: `25b9dd1e7`

Result SHA256: `1c6dc2fbe838c410a79f33d7fee1392d85877a0fcebdfb6d18ced0927ff657f1`

This is a model-free audit of the completed `numbered_list_cached_value_read_split_probe` result. It does not
change the registered predictions, rerun the model, or promote the result into the canonical circuit record.

## Computation being tested

The upstream object is the exact layer-8 attention contribution

$$
T_q=\sum_{h\in\{3,7\}}p^{(8)}_{h,q,k}\,W^{O,(8)}_h
\left(\lambda_8 W^{V,(0)}_h z_k^{(0)}\right),
$$

written at the final query position $q$ from the final visible list-label position $k$. The experiment carries
$T$ alongside the residual stream. For a selected downstream component, it subtracts the appropriately skip-scaled
$T$ from that component's normalized input while leaving the ordinary residual path intact. The arms are:

- `DIRECT`: prevent only the final normalization and unembedding from reading $T$;
- `READS`: prevent MLP8 and every attention/MLP component in blocks 9–17 from reading $T$, but leave the final direct
  read intact;
- `FULL`: prevent both sets of reads, which reconstructs deletion of $T$ from the residual path;
- one arm for each downstream component, plus block and FIT-selected joint arms.

The outcome is damage to the correct numeric logit margin,

$$
d_m=m_{\text{native}}-m_{\text{arm}},
$$

so positive values mean that preventing a read harms the correct answer.

## Independent group-level reanalysis

The producer's bootstrap resampled individual row endpoints. Six row families and their two directions share the
same generated `group_id`, so the independent statistical unit is the group, not each row endpoint. I mapped every
saved `row_id` back to its frozen `group_id`, averaged all five answer-changing list families and both directions
inside each group, then resampled the 32 FIT groups or 16 SELECT groups 20,000 times with seed 5932808.

| SELECT arm | mean margin damage | group-bootstrap 95% interval | fraction of reference |
|---|---:|---:|---:|
| `FULL` | 2.121 | [2.024, 2.215] | 1.000 |
| `READS` | 1.914 | [1.788, 2.041] | 0.903 of `FULL` [0.879, 0.924] |
| `DIRECT` | 0.218 | [0.211, 0.223] | 0.103 of `FULL` [0.099, 0.107] |
| `TOP2_JOINT` = MLP8+MLP10 | 0.701 | [0.671, 0.732] | 0.366 of `READS` [0.341, 0.392] |

The direct and downstream-read effects are almost additive on SELECT:

$$
d_m(\mathrm{FULL})-d_m(\mathrm{READS})-d_m(\mathrm{DIRECT})
=-0.011,
$$

with a group-bootstrap interval of $[-0.053,0.033]$. FIT has a small positive interaction of $0.095$
[$0.082,0.109$], but the main routing conclusion is stable on both splits.

For the repeated-label copy control on SELECT, deleting all edges of $T$ changes cross-entropy by $-0.352$ nat
[$-0.426,-0.282$], while preventing only MLP8 and MLP10 from reading it changes cross-entropy by $-0.267$ nat
[$-0.287,-0.246$]. Thus the two selected MLPs account for about 76% [64%, 93%] of this *collateral* improvement.
That does not make their removal selective: it confirms that the same path substantially interferes with copying.

## Verdict and corrections

The strongest licensed conclusion is narrower than the producer headline:

1. About 90% of the correct-list-successor effect of this particular $T$ reaches the logits through downstream
   component reads; only about 10% is its direct final read. This survives group-level resampling.
2. MLP8 is the largest single measured reader, but the effect is distributed. The physically executed MLP8+MLP10
   joint arm carries 36.6% of the collective downstream-read effect, far below the registered 50% concentration bar.
   The two-reader concentration claim therefore fails.
3. The registered `pred_c` used the *sum of two singleton removal effects* rather than the physically executed joint
   intervention. This does not flip the registered result—both fail the 50% bar—but future reusable tooling must use
   the joint arm for a joint-reader claim and report the interaction explicitly.
4. The direct unembedding calculation is a descriptive linear decomposition at the native normalization scale. Its
   52.1% copy-over-successor fraction and median difference 0.00245 do not identify $T$ as either a copy vector or a
   successor vector. The evidence supports “the successor effect is computed downstream,” not the stronger statement
   “$T$ is a context-blind copy representation.” The scalar attention weights in $T$ remain context dependent.
5. FIT and SELECT outcomes for all fixed arms were computed in the same pass, although the source code selects the
   top two using FIT values only. This is mechanically non-adaptive but weaker than physical split separation. The
   next protocol must finish and freeze FIT selection before opening SELECT, then keep FINAL_TEST/OOD unopened.
6. The literal execution price was 457 batched model calls and 20,212 example evaluations, not simply “19.6k
   forwards.” Both units should be recorded by the shared battery.

This result improves the computational specification of the numbered-list circuit, but it is not a new high-quality
circuit by itself: the intervention remains nonselective on the active copy control, no reader subset passed the
concentration gate, and FINAL_TEST/OOD remain unopened. It should guide the reusable battery's reader-path machinery,
not trigger another bespoke reader sweep.
