# Block-3 Family-F v2 recovered fit result

**Outcome:** receipt-last reporting recovery succeeded; the registered refitted
Family-F programs fail the composable-port fit gate.  No validation or final role was
opened.

## What was recovered

Family-F v1 had completed all fitting and written the exact program bundle, but its
terminal report validator compared a CUDA reduction maximum directly with a CPU
maximum.  V2 pinned the spent v1 bytes, independently reconstructed every program from
sealed parents, reran only the frozen 480-row fit report, validated CPU and CUDA
polarization separately, and published a result plus receipt.

- v2 result SHA256:
  `18b03ccf3d6710813375bb7e09b1a3c313d5e7790e2ca3c9a9b683fbf91897c5`
- v2 receipt SHA256:
  `e81673095c7b6202fdec293c6ad34924fb9acb15213d02ba4b203d5ff8c65a5a`
- elapsed time: `75.26 s`
- maximum allocated CUDA memory: `4,719,026,176 bytes`
- rows: 480 fit, 0 validation, 0 final
- ground-truth target tokens used: 0
- report calls: 60 prefixes, 60 teachers, and 60 calls for each of 18 students
- model state before/after: byte-identical

Independent semantic replay of the saved result and receipt passes.

## Main numbers

The metric below is KL from the native teacher's next-token distribution on the fit
documents.  Smaller is better.  `Summed-write NRMSE` is the root-mean-square error of
the complete MLP3 residual write, divided by the RMS magnitude of the native write.

| Program | K | Document-balanced teacher KL | Summed-write NRMSE |
|---|---:|---:|---:|
| continuous downstream scores, native Down | 512 effective mass | 0.05302 | 0.85629 |
| binary downstream support, native Down | 256 | 0.08041 | 0.93505 |
| binary downstream support, local decoder refit | 256 | 0.14913 | 0.78860 |
| matched random support, local decoder refit | 256 | 0.17394 | 0.78394 |
| binary downstream support, native Down | 512 | 0.05772 | 0.86957 |
| binary downstream support, local decoder refit | 512 | 0.08476 | 0.70275 |
| matched random support, local decoder refit | 512 | 0.10077 | 0.70074 |
| prior activation-selected Family A | 512 | 0.08862 | 0.67916 |
| same-support permuted-cross control | 512 | 0.46105 | 0.93835 |

## What worked

Downstream consequence contains real support-selection signal:

- at K512, refitted Family F improves KL over matched random by `0.01600` nat
  (`15.9%` of the random KL);
- it improves over the prior activation-selected Family A by `0.00385` nat (`4.35%`);
- the same-support wrong-cross control is much worse (`0.46105`), so the result is not
  explained by arbitrary support plus a flexible decoder;
- row-reversed and document-deranged selector controls are also worse (`0.13075` and
  `0.13030`).

So the frozen nonlinear suffix does identify a somewhat better finite set of native
product gates than activation energy or random selection.

## Why Family F nevertheless stops

The registered useful-interface gate requires summed-write NRMSE at most `0.20`.
Family F obtains `0.78860` at K256 and `0.70275` at K512.  These are the actual summed
write errors, not the earlier incommensurate four-term stacked statistic.  Both fail
by a wide margin.  Under the v2 recovery contract, validation may open only if the
original fit gates pass; therefore neither candidate advances and final remains
sealed.

This means Family F has found a better *behavioral support selector*, not a faithful,
independently editable MLP3 port.

## The important unexpected result

Local decoder refitting improves write NRMSE but makes downstream behavior worse:

- K256: KL worsens from `0.08041` with native Down to `0.14913` after refit;
- K512: KL worsens from `0.05772` with native Down to `0.08476` after refit;
- at K512, NRMSE simultaneously improves from `0.86957` to `0.70275`.

This is direct evidence that local Euclidean reconstruction and downstream functional
faithfulness are misaligned at this interface.  The support was learned while the
native Down columns were fixed; replacing those columns with the locally optimal
decoder partially destroys the causal solution.  The zero-marginal-storage affine
diagnostic does not repair this: calibrated teacher-F KL is `0.07534`, still worse
than the uncalibrated binary/native-Down arm's `0.05772`.

The binary/native-Down arm was registered as diagnostic and cannot be promoted from
this outcome.  It does justify a fresh prospective experiment: freeze that grammar,
test it on new documents with finite positive/negative edits and matched controls, and
decide explicitly whether the goal is behavioral extraction or a locally faithful
port.  It must not be relabeled as a Family-F validation success.

### Post-outcome weight diagnostics

These exact tensor comparisons are descriptive and do not change the registered
decision.  At K512, the refitted decoder differs from the corresponding native Down
columns by `1.066` native-decoder Frobenius norms.  Across columns, median directional
cosine is `0.828` and median norm ratio is `1.680`; at K256 they are `0.747` and
`2.023`.  Thus the refit is not a small scalar correction—it substantially rotates
and amplifies the selected residual directions, consistent with its causal damage.

The K512 Family-F support overlaps Family A in 258 of 512 gates (Jaccard `0.337`) and
the fixed random support in 260 (Jaccard `0.340`).  Consequence selection is therefore
not merely a small refinement of activation selection.  By contrast, the row-reversal
and document-derangement null supports overlap in 505 of 512 gates, suggesting both
nonsignal objectives collapse toward nearly the same generic boundary solution.  No
atom-level semantic claim follows from these overlaps.

## Project consequence

Family F closes one hypothesis: downstream-aware gate selection followed by a local
least-squares decoder is not enough to produce a faithful MLP3 replacement at 256 or
512 products.  It opens a sharper one: downstream selection may work specifically
because it preserves the native decoder geometry.  Future joint optimization should
either keep that decoder fixed or optimize support and decoder together in downstream
KL; inserting a local-MSE refit between them is now empirically contraindicated.
