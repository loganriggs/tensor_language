# Rank640 behavioral agreement: accuracy passes, exact argmax agreement fails

Date: 2026-08-28

Status: prospective new-instrument gate failure on previously opened cross-task roles.
Result SHA-256:
`031e2631c3c1e2564e9e34f3b7a2e59cf1fb6e35b46c918187e429a3c49bfb37`.

Provenance status: bounded measured evidence, not strict fail-closed authority. An
independent audit reproduced every number, parent hash, row hash, and every declared
source hash at source commit `f0af4779`, but found that the declared source list omitted
three executed transitive modules:
`early_mlp_suffix_transport_v1_rows.py`, `tensor_bilin18_shared_qk_rank512.py`, and
`tensor_preserving_attention_identity.py`. The pinned commit still identifies their
contents, so this is an enumeration defect rather than a detected numerical defect;
the stronger “source-closed” label is withheld.

## Result

The admitted 516,707,766-value rank640 complete program exactly replayed its prior CE
receipt and passed ownership, KL, accuracy, and rare-target gates. It missed the frozen
98% top-1 agreement threshold on both roles.

| role | top-1 agreement | live accuracy | program accuracy | retained accuracy | KL(live‖program) |
|---|---:|---:|---:|---:|---:|
| skip31000 | 95.782% | 40.460% | 40.232% | 99.437% | 0.005114 |
| skip35000 | 96.077% | 41.536% | 41.401% | 99.673% | 0.004526 |

On targets seen at most four times in the fit rows, retained accuracy is 99.661% and
100.035%; the absolute changes are -0.092 and +0.010 percentage points. Thus rank640
does not inherit the context-free table's catastrophic rare-target failure.

## Interpretation

The prior ownership, CE, and intervention certificates remain true in their frozen
currencies. Under the expanded behavioral validation vector, rank640 is 5-for-6 and
must not be described as near-exact in argmax identity. It changes the selected token
on about 4% of positions while losing only 0.14--0.23 accuracy points. This arithmetic
suggests that most changed choices are either wrong under both models or exchange very
close candidates, but logit-margin stratification was not preregistered and is not
claimed here.

This result improves the definition of simplicity. Top-1 agreement, task accuracy,
full-distribution KL, rare-target behavior, and causal transport are distinct
consequences. A candidate can fail exact decision replay while preserving predictive
utility. Future frontiers should display all of them rather than silently replacing
one with another or retroactively relaxing a failed threshold.

## Claim boundary

The audit used the already-opened skip31000/35000 roles and rank. It is a prospective
instrument test, not fresh OOD promotion. It does not change the 5.3481% storage price,
the 10.923% named strict-causal denominator, or the 32.1% semantic estimate. It also
does not establish that the disagreements occur at small margins; that requires a
separately frozen analysis. Because of the transitive-source enumeration omission, use
this result as bounded evidence until a fresh source-closed replication is frozen.
