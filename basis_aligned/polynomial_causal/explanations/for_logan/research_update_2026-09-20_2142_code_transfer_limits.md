# Code transfer exposes errors that average loss concealed

2026-09-20 21:42 UTC

Both frozen intervention tests were repeated on16 local Python source files,
selected and hashed before model evaluation, with unchanged thresholds.
This is a limited domain-shift screen: the files are related, the corpus is
not representative, and pretraining overlap is unknown.

| Branch intervention on code | CE added above native | KL from native | Top-token agreement |
|---|---:|---:|---:|
| Remove branch | 0.37197 | 0.51239 | 66.31% |
| Ten-product replacement | 0.01061 | 0.05907 | 89.79% |
| 26-product replacement | 0.01080 | 0.04780 | 90.58% |

The small program passes the CE<.02 component but fails KL<.02. Its predictions
move substantially more than the small signed average loss suggests. Exact
replay passes, and disturbance remains less than half that caused by ablation.

Individual native-removal prediction also transfers unevenly:

| Feature | FineWeb effect error | Code effect error | Code effect cosine |
|---|---:|---:|---:|
| 0 | 15.8% | 18.2% | .988 |
| 1 | 31.3% | 48.1% | .892 |
| 2 | 48.3% | 46.9% | .884 |
| 3 | 59.0% | 54.8% | .851 |
| Joint | 19.1% | 22.9% | .975 |

The stricter modes0/1 transfer bar fails because of feature1. The broader
all-four bar passes; these are distinct registered criteria. There is no
justification for claiming all features are robustly extracted.

A follow-on CPU diagnostic asks whether an optimal scalar multiplier of the
predicted effect vector could repair the feature1 failure. It cannot reach the
40% bar: even the same-panel oracle has45.2% error. This is a limitation of
scalar rescaling of post-softcap effect vectors, not a bound on changing a
nonlinear model intervention's amplitude. No coefficients were refit or exported.

The next research decision should target directional or extrapolation error,
rather than optimize signed average CE. The original goal remains unmet:
semantic selectivity, broader OOD evidence, reusable components across tasks,
and a simpler complete executable path are not established.

[Branch results](../../direct_tensor_match/CODE_SHIFT_BRANCH_V1.json),
[feature results](../../direct_tensor_match/CODE_SHIFT_MODES_V1.json),
[gain diagnostic](../../direct_tensor_match/INTERVENTION_GAIN_ORACLE_V1.json).
