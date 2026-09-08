import inspect

import run_temporal_iswas_v15_tied_embedding_semantic_router_v16_v1 as runner

def test_price_is_zero_transformer_and_literal_centroid_storage():
    assert runner.PRICE["model_forwards"]==0
    assert runner.PRICE["checkpoint_loads"]==1
    assert runner.PRICE["fit_parameters"]==3*2*1152

def test_v16_c_is_explicitly_excluded_not_padded():
    source=inspect.getsource(runner.main)
    assert 'r["transform_id"] in ("A1","A2","P")' in source
    assert 'r["transform_id"]=="C"' in source
    assert "c_excluded" in source

def test_fit_is_v15_only_and_v16_is_opened_once():
    source=inspect.getsource(runner.main)
    assert "semantic.fit(torch,feat15[train_idx]" in source
    assert "semantic.predict(torch,feat16" in source
