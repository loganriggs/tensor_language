# Price and test attention7 omissions inside the extracted city-write generator

The single-residual6 generator has now passed fresh effect/selectivity gates, but
its full attention7 computation increases price. Test whether attention7 terms can
be omitted from this already supported operator while preserving its causal effect.
This is post-identification simplification of the declared generator; it is not a
claim that whole native heads are independent semantic circuits.

Use only the original opened CITY_FULL_PILE_V2 rows. Keep residual6, token tables,
MLP7 L/R, five folded readers, head8 maps and all normalizers fixed. In the generated
attention7 sum, evaluate all nine heads, no heads, and each of nine leave-one-head-out
sets. Recompute normalized MLP7 input, all five readers, query/key normalization and
mixed8 approximate RMS for every candidate. Do not just subtract its final write.
No fitting, no ranking by weight norm or activation variance.

CPU preflight: full-head version reproduces frozen single-input generator within
1e-4relative on all40sequences; finite supported outputs for all candidates; price
retained maps directly. Each retained attention7 head costs six128x1152 maps
(Q1,K1,Q2,K2,V,output); all other tables/weights charged. This preflight alone cannot
select a causal simplification.

Installed CPU screen:13padded batches of40sequences (520equivalent forwards,
234block calls),twoCPUthreads,300second cap: native,fullnative city removal,
fullgenerated removal,no-attention7 removal,nineleave-one-head-out removals.
All interventions at attention8, native MLP8/suffix recomputed. No native attention7
module is edited in the body; the omission is internal to the edit generator.

Predictions/gates frozen before execution:
a. Native/fullreference CPU scores match frozen original GPU reference <=1e-4abs
   AND<=1e-5relative; fullgenerated matches previous CPUvalues at same bars;
   finite supported outputs; exact call counts; CUDA uninitialized.
b. At least one strictly cheaper generator has target-effect relative L2 error
   <=.05 versus exact fullnative removal, positive attenuation>=.90 and mean>=.02
   among at least90/120capable pairs, each control RMS<=.5target RMS.
c. If b passes, select minimum retained-head count, then minimum effect error,
   then lexicographic head set. Record literal total weight count and input count;
   saving must be positive. If none passes, no simplification is selected.

All omitted configurations including failures remain in the receipt. These opened
selection results are a screen, not a new selective-removal or OOD certificate.
Next selected formula requires same-site norm-matched nulls and a further fresh
panel before promotion. Do not reuse the now-opened FRESH_V1 panel as fresh.
No independent composition or matched-effect random-component simplicity claim.
