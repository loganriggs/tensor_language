import pytest

import attention_response_source_eval as response


def test_response_groups_reject_invalid_head_before_hook():
    batch = type("Batch", (), {
        "row_ids": ("r",), "semantic_positions": (4,),
        "token_rows": ((1, 2, 3, 4, 5),),
    })()
    donor = type("Batch", (), {
        "row_ids": ("r",), "semantic_positions": (4,),
        "token_rows": ((9, 2, 3, 4, 5),),
    })()
    backend = type("Backend", (), {"model": type("Model", (), {"config": type("Config", (), {"n_head": 2, "n_embd": 4})()})()})()
    with pytest.raises(response.AttentionResponseSourceError):
        response.intervene_response_groups(
            backend, batch, donor, {}, {}, ("cue",), layer=11, selected_heads=(2,)
        )
