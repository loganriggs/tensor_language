import pytest

import attention_path_mediation_eval as mediation


class Batch:
    row_ids = ("a", "b")
    semantic_positions = (5, 8)


def test_reader_positions_accept_exact_causal_sets():
    assert mediation.validate_reader_positions(Batch(), ((1, 2), (4, 5))) == (
        (1, 2), (4, 5)
    )


def test_reader_positions_reject_duplicates_and_post_query():
    with pytest.raises(mediation.AttentionPathMediationError):
        mediation.validate_reader_positions(Batch(), ((1, 1), (4, 5)))
    with pytest.raises(mediation.AttentionPathMediationError):
        mediation.validate_reader_positions(Batch(), ((1, 2), (4, 9)))


class HookHandle:
    def __init__(self, owner):
        self.owner = owner

    def remove(self):
        self.owner.removed = True


class HookPoint:
    def __init__(self):
        self.hook = None
        self.removed = False

    def register_forward_pre_hook(self, hook):
        self.hook = hook
        return HookHandle(self)


class FakeBackend:
    def __init__(self):
        projection = HookPoint()
        attention = type("Attention", (), {"c_proj": projection})()
        block = type("Block", (), {"attn": attention})()
        transformer = type("Transformer", (), {"h": [block]})()
        self.model = type("Model", (), {"transformer": transformer, "config": type("Config", (), {"n_head": 1, "n_embd": 1})()})()

    def forward_states(self, batch, *, maximum_boundary):
        return (batch, maximum_boundary)


def test_capture_source_written_states_removes_hook(monkeypatch):
    backend = FakeBackend()
    monkeypatch.setattr(mediation, "fixed_source_delta_hook", lambda *args, **kwargs: "hook")
    result = mediation.capture_source_written_states(
        backend, "base", "donor", {}, {}, ((0,),), maximum_boundary=1, writer_layer=0
    )
    projection = backend.model.transformer.h[0].attn.c_proj
    assert result == ("base", 1)
    assert projection.hook == "hook"
    assert projection.removed is True
