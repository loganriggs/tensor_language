import circuit_candidate_numbered_list_cached_term_head_source_factorial as candidate
import run_numbered_list_cached_term_head_source_factorial as runner


def _records(*, t3_target=1.0, t7_target=0.1, joint_target=1.2,
             t3_control=0.05, t7_control=0.4, joint_control=0.5):
    output = []
    for family in candidate.FAMILIES:
        for endpoint in candidate.ENDPOINTS:
            for index in range(16):
                target = family in candidate.TARGET_FAMILIES
                damages = {"zero_T3": t3_target if target else t3_control,
                           "zero_T7": t7_target if target else t7_control,
                           "zero_T3_T7": joint_target if target else joint_control}
                record = {"row_id": f"{family}/{endpoint}/{index}",
                          "group_id": f"g/{index}" if family != "sequence_word_copy_control" else f"w/{index}",
                          "family": family, "endpoint": endpoint, "native_margin": 4.0}
                for condition, damage in damages.items():
                    record[f"{condition}_margin_damage"] = damage
                    record[f"{condition}_ce_increase"] = 0.05 if target else 0.0
                    record[f"{condition}_logit_rms"] = 0.05 if condition == "zero_T3" else 1.0
                    record[f"{condition}_term_norm"] = 500.0
                    record[f"{condition}_answer_preserved"] = True
                record["interaction_margin_damage"] = joint_target - t3_target - t7_target if target \
                    else joint_control - t3_control - t7_control
                output.append(record)
    return output


def test_scoring_finds_selective_t3_and_positive_interaction():
    scored = runner.score_records(_records(), replay_rse=0.0, joint_term_rse=0.0)
    assert scored["instrument_passed"] is True
    assert scored["selectively_necessary"]["zero_T3"] is True
    assert scored["selectively_necessary"]["zero_T7"] is False
    assert scored["predictions"]["pred_b_individual_source_selective"] is True
    assert scored["interaction"]["ci95_low"] > 0


def test_exactness_failure_invalidates_instrument():
    scored = runner.score_records(_records(), replay_rse=1e-3, joint_term_rse=0.0)
    assert scored["instrument_passed"] is False
    assert scored["predictions"]["pred_a_instrument_exact"] is False


def test_joint_synergy_without_selective_singleton_is_characterized():
    scored = runner.score_records(_records(t3_target=.1, t7_target=.1, joint_target=1.0,
        t3_control=.8, t7_control=.8, joint_control=.5), replay_rse=0.0, joint_term_rse=0.0)
    assert scored["predictions"]["pred_b_individual_source_selective"] is False
    assert scored["predictions"]["pred_c_cooperative_service"] is True
