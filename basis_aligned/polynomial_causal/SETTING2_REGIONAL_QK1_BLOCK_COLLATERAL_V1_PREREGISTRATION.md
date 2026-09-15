# Split QK1 routing collateral by ordered interaction block

Frozen 15 September 2026 after the additive split assigned the failed work/jobs
reader to late-touching QK1 routing. Reuse the same opened rows. With late group
$D$ and remainder $R$, run native removals of $B(D,D)$, $B(D,R)$, $B(R,D)$,
and all three together. QK2 and values remain native.

- `pred_a_exact_instrument`: preserve exact audits within $2\times10^{-6}$ over
  30 physical batches.
- `pred_b_full_routing_live`: full routing target and work/jobs RMS each exceed
  `.002` in both families.
- `pred_c_work_jobs_block_localized`: one block reproduces the failed family-1
  full-routing work/jobs effect within `.35` relative error.
- `pred_d_selectivity_improving_block`: one block retains at least 20% of full
  target RMS while reducing work/jobs/target ratio by at least 30%.
- `pred_e_recursive_block_sum`: the three separately removed block effects sum
  to full-removal target and work/jobs effects within `.35` in both families.

This is opened-panel interaction attribution. It does not provide fresh transfer,
a fitted circuit, parameter updates, or quantization.
