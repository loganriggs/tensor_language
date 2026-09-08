import inspect

import run_temporal_iswas_v15_entry12_rank2_normalized_weight_reader_atlas_v1 as runner


def test_price_and_scope_are_frozen():
    assert runner.PRICE_MAX["differentiable_transformer_forwards"] == 6
    assert runner.PRICE_MAX["checkpoint_loads"] == 1
    assert runner.KNOWN_WRITERS == {"L8H1", "L9H1", "L9H4", "L11H3"}


def test_reader_atlas_uses_all_literal_factor_matrices_and_exact_rms_contract():
    source = inspect.getsource(runner.reader_records)
    for name in ("c_q", "c_k", "c_q2", "c_k2", "c_v", "Left", "Right"):
        assert name in source
    assert "reader.reader_response" in source
    assert "block.lambdas[0]" in source and "block.lambdas[1]" in source


def test_writer_atlas_splits_heads_and_mlps():
    source = inspect.getsource(runner.writer_records)
    assert "c_proj.weight" in source
    assert "block.mlp.Down.weight" in source
    assert "basis.T @ matrix" in source


def test_predictions_keep_weight_geometry_diagnostic():
    source = inspect.getsource(runner.main)
    assert '"scope": "normalization_aware_weight_geometry_not_causal_identification"' in source
    for suffix in ("rankings_differ", "tracks_exact_finite_response",
                   "nominations_are_fold_stable", "shared_exact_reader_is_nominated"):
        assert suffix in source
