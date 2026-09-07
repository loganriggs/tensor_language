import audit_temporal_iswas_selected_projector_attn15_reader_path_v1 as audit


def test_audit_is_strictly_zero_model():
    assert all(value == 0 for value in audit.PRICE.values())


def test_audit_binds_queued_multi_environment_runner():
    assert audit.EXPECTED["multi"] == \
        "8c14405674b4773303e197be04668f29ca97d2662dbeccc0f9653c119716c556"
