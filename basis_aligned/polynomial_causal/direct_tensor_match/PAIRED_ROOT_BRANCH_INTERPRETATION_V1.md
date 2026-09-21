**The paired graph preserves its larger parent's native behavior, but does not beat the cheap baseline.**

Frozen native branch screen completed 21 September 2026, 20:08 UTC in 2.15 seconds: 32 FineWeb and 16 stdlib documents, 239 scored tokens per document. Panels were already opened. No fitting occurred. Actual normalization, attention/residual background, cross terms and final softcap remain in the model; only the pure quartic numerator is replaced.

| Branch | Products | FineWeb CE added | Code CE added | FineWeb branch-effect error | Code branch-effect error |
|---|---:|---:|---:|---:|---:|
| Remove branch | 0 | .15265 | .56676 | 100% | 100% |
| Narrow baseline | 26 | .00413 | .01934 | 32.88% | 22.89% |
| Joint-fit parent | 656 | .00703 | .01597 | 33.71% | 27.21% |
| Paired graph | 384 | .00702 | .01764 | 33.64% | 27.20% |

Lower CE added is better. Branch-effect error is candidate logit-discrepancy norm divided by the native branch-ablation logit-change norm, pooled across the domain. It is not polynomial reconstruction error or a semantic removal score.

Registered outcomes: instrument PASS, mean CE/KL preservation PASS, baseline comparison FAIL. The paired graph meets <=1.1 parent error in both domains but fails <=.9 narrow26 in both. Maximum exact replay2.44e-7 and paired/spectral numerator replay5.48e-7 rule out a gross adapter or rewrite error. Passing average CE/KL<.02 does not mean every document passes: paired worst-document CE added .03292/.03729.

A post-screen CPU audit uses 10,000 paired document bootstrap resamples and leave-one-document-out ratios. On code, paired/narrow error ratio is1.189, descriptive95% interval[1.101,1.282]; every leave-one-out ratio remains>1.16, and paired wins only1/16documents. Thus that disadvantage is not one outlier. FineWeb ratio1.023 has interval[.932,1.111], so its small aggregate error disadvantage is less stable. FineWeb CE/KL favor narrow26; code CE does not clearly distinguish narrow26 from paired384. No multiple-testing or untouched-OOD claim follows from these descriptive intervals.

The two-stage compiler succeeds at making a given approximation smaller. The experiment does not show that the resulting larger feature dictionary is worth its cost relative to a26-product alternative. Do not extend this graph's rank sweep on the strength of aggregate CE. The next scientific question is changed-input fidelity: whether the narrow program's apparent advantage survives source interchanges, or whether extra features preserve responses hidden by ordinary examples. Reuse the existing pure-quartic replacement and source-interchange machinery; preserve both controls. This is a branch-level question, not yet semantic feature identification.

[Preregistration](PAIRED_ROOT_BRANCH_PLAN_V1.md) · [Native rows](PAIRED_ROOT_BRANCH_V1.json) · [Document audit](PAIRED_ROOT_BRANCH_DOCUMENT_AUDIT_V1.json) · [Audit code](audit_paired_branch_documents.py) · [Compiler evidence](PAIRED_ROOT_INTERPRETATION_V2.md).
