# Subject-number downstream response census

v642 traces the frozen five-port pre-L11 intervention through the complete native
suffix on128 original and32 historical-fresh prompts. These panels are now opened.
Six prefix and six suffix calls, zero fits; native runtime6.76seconds.

The accounting instrument passes: residual-vector relative closure1.20e-6,
answer-margin effect closure3.48e-6, selected-logit maximum absolute replay
error6.31e-6. The decomposition keeps every downstream attention and MLP response,
residual scaling, final RMS and softcap. Its secant reader uses both endpoints,
so the census is not an extracted predictor.

| Contribution | Aligned fraction across12 direction/template cells |
| --- | --- |
| Propagated pre-L11 carry | .460 to .708 |
| Attention11 | .027 to .096 |
| MLP11 | -.062 to .100 |
| Each attention12–17 separately | -.006 to .016 |
| MLP17 | .063 to .236 |
| Final RMS correction | -.222 to .073 |

Aligned fractions are signed projections onto the total effect, not independent
causal effects or probabilities. Other MLP writes remain in the full receipt.
The direct-carry>=50%-everywhere hypothesis fails on the original singular cells;
the single-module>=25%-everywhere hypothesis fails for every module. MLP17 remains
positive in every cell but is not alone sufficient. Direction-dependent final
normalization is material and must not be silently frozen.

## Next causal discriminator: v643

The small attention12–17 attributions nominate a **joint causal test**, not an
omission justified by magnitude. Hold these six attention writes at their native
baseline values, while recomputing the edited residual, attention11, every MLP,
every RMS and final softcap. Compare predicted versus true five-port intervention
damage, requiring relativeL2<=.10 and cosine>=.99 in every cell. The same frozen
background must replay the baseline to maximum absolute error<=1e-4.

Opposing explanation: individually small attention responses jointly change MLP
feedback enough to invalidate the reduced response computation. Preserve this
negative if it occurs. The proposed program still depends on six exact native
attention-write backgrounds, the pre-L11 prefix and first-value state; these are
charged ports, not free inputs. Even success would establish a conditional response
reduction, not a complete simple circuit or fresh OOD reuse.

Primary receipt: [v642 JSON](../bilinear_quotient/circuits/followups/subject_suffix_census_v642_result.json).
Executable: [v642](../bilinear_quotient/ops/run_subject_suffix_census_v642.py).
Next registered executable: [v643](../bilinear_quotient/ops/run_subject_attention_freeze_v643.py).
