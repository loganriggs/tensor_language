# Invalid receipt: attention value weight pullback v1

The causal outcomes are preserved in `temporal_five_mlp_attention_value_weight_pullback_v1_result.json`, but its formal terminal is invalid.

The sole validity miss is numerical: the maximum local projected-value coordinate RSE is `1.1022704970997665e-09`, above the preregistered `1e-10` bar. The independently computed whole-program compiled-versus-captured centered-logit RSE is `2.4861236333495462e-11`, all target/control gates pass, and the minimum aligned input-weight-map cosine is `0.9415028691291809`.

No scientific outcome may be promoted directly from this receipt. A separately preregistered tolerance audit must bind the frozen result bytes and justify any revised numerical threshold without rerunning or changing the causal arms.
