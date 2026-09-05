#!/usr/bin/env python3
# BQGATE: frozen OOD transfer, removal, capability, and replay predictions are emitted here.
"""Open the pre-TEST-frozen Task14 OOD head-11.3 candidate once."""

from pathlib import Path

import circuit_fast_screen_candidate_task14_ood_cross_syntax as candidate
import run_circuit_fast_screen_task14_test_transfer_removal as shared


REGISTERED_PREDICTIONS = (
    "pred_a_native_capability",
    "pred_b_head11_3_cross_noun_transfer",
    "pred_c_head11_3_literal_removal",
    "pred_d_native_head_replay",
    "pred_e_head_hook_live",
)


PROTOCOL = shared.RunProtocol(
    candidate=candidate,
    request_id="task14-ood-cross-noun-head11-3-transfer-removal-v1",
    experiment_id="fast-screen-task14-ood-cross-noun-head11-3-transfer-removal-v1",
    result_relative=Path(
        "circuits/fast_screens/task14_ood_cross_noun_head11_3_transfer_removal_v1_result.json"
    ),
    prior_art_sha256="b62daaf46bbf4e77e960283ea27c33fbfffb866591d1f2bc58d1a94cfb537dde",
    result_schema="task14_ood_cross_noun_head11_3_transfer_removal_result_v1",
    novelty=(
        "First Task14 OOD opening: fronted/two-attractor cross-noun transfer and literal "
        "removal for preselected head11.3, with pairing and bars frozen before TEST outcomes."
    ),
    limits=(
        "OOD only, using the candidate frozen before TEST. This tests the current native head "
        "carrier; it does not prove the head is the smallest semantic unit."
    ),
)


if __name__ == "__main__":
    shared.cli(PROTOCOL)
