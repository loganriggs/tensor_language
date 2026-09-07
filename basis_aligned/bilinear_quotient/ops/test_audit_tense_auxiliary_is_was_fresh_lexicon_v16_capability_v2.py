import audit_tense_auxiliary_is_was_fresh_lexicon_v16_capability_v2 as audit


def test_corrected_bar_is_same_registered_ratio_on_actual_panel_size():
    assert audit.CORRECTED_JOINT_BAR == 12
    assert audit.CORRECTED_JOINT_BAR / 16 == 24 / 32


def test_audit_price_opens_no_model_or_causal_outcome():
    assert all(value == 0 for value in audit.PRICE.values())
