#!/usr/bin/env python3
"""Float64 state-telescope repair of the frozen v23 direct-readout factorial."""

# BQGATE: EXPERIMENT pred_a_hash_hooks_state_closure_logit_replay_finiteness_and_exact_price pred_b_direct_identity_carry_is_dominant_selective_and_split_stable pred_c_downstream_response_is_a_real_sign_stable_correction pred_d_final_decoder_direct_response_interaction_is_small pred_e_analytic_carry_is_exact_finite_and_nonzero
import run_temporal_iswas_v23_direct_residual_readout_factorial_v1 as experiment


PREDICTION_KEYS = (
    "pred_a_hash_hooks_state_closure_logit_replay_finiteness_and_exact_price",
    "pred_b_direct_identity_carry_is_dominant_selective_and_split_stable",
    "pred_c_downstream_response_is_a_real_sign_stable_correction",
    "pred_d_final_decoder_direct_response_interaction_is_small",
    "pred_e_analytic_carry_is_exact_finite_and_nonzero",
)
experiment.AUTHORITY = experiment.ROOT.parent / "polynomial_causal/TEMPORAL_ISWAS_V23_DIRECT_RESIDUAL_READOUT_FACTORIAL_V2_PREREGISTRATION.md"
experiment.PRIOR = experiment.ROOT / "circuits/prior_art/temporal_iswas_v23_direct_residual_readout_factorial_v2.json"
experiment.OUT = experiment.ROOT / "circuits/followups/temporal_iswas_v23_direct_residual_readout_factorial_v2_result.json"
experiment.CANDIDATE_ID = "cross_task.temporal_iswas.v23_direct_residual_readout_factorial_v2"
experiment.SCHEMA = "temporal_iswas_v23_direct_residual_readout_factorial_result_v2"
FAILED_V1 = experiment.ROOT / "circuits/followups/temporal_iswas_v23_direct_residual_readout_factorial_v1_result.json"
experiment.EXTRA_PATHS = {"failed_v1": FAILED_V1}
experiment.EXPECTED["authority"] = "14050b658c3523d01252831fffc2e7be9fb03ce97c77441738097cda61653cba"
experiment.EXPECTED["prior"] = "38c1b88b143b69093db8aa0a01bd7bfe7951cc287dc8a8b081fb00a1ea236481"
experiment.EXPECTED["failed_v1"] = "71ecf650a593f4a19010c703df37a22b65eaa0e555dd9a85e1198f1ac514251d"


if __name__ == "__main__":
    experiment.main()
