from __future__ import annotations

import torch
import torch.nn.functional as F

import tensor_bilin18_shared_qk_whole_program as whole


def test_score_partition_is_complete_and_matches_direct_ce() -> None:
    logits = torch.tensor([[[2.0, 0.0, -1.0], [0.0, 3.0, -2.0]]])
    rows = torch.tensor([[0, 1, 2]])
    seen = torch.tensor([True, False, True])
    accumulator = whole.empty_score()
    whole.add_score(accumulator, logits, rows, seen, score_start=0)
    measured = whole.finalize_score(accumulator)
    direct = float(F.cross_entropy(logits.reshape(-1, 3), rows[:, 1:].reshape(-1)))
    assert abs(measured["all"]["ce"] - direct) < 1e-7
    assert measured["all"]["positions"] == 2
    assert measured["seen_current"]["positions"] == 1
    assert measured["unseen_current"]["positions"] == 1


def test_context_metric_is_one_for_exact_transport_and_zero_for_no_transport() -> None:
    base = torch.zeros(1, 40, 2)
    changed = base.clone()
    changed[:, 35, 0] = 2.0
    exact = whole.context_metrics(base, changed, base, changed)
    assert exact["context_delta_recovery"] == 1.0
    assert exact["context_delta_cosine"] == 1.0
    zero = whole.context_metrics(base, changed, base, base)
    assert zero["context_delta_recovery"] == 0.0
    assert zero["program_max_abs"] == 0.0


def test_protocol_binds_whole_program_context_roles_and_price() -> None:
    source = whole.Path(whole.__file__).read_text()
    prereg = whole.PREREG.read_text()
    for fragment in (
        "490_165_686", "545_904_054", "score_program", "score_native",
        "model_reference() is not None", "context_delta_recovery",
        "seen_current", "unseen_current", "os.O_EXCL",
    ):
        assert fragment in source
    assert "already-opened roles" in prereg
    assert "all-position" in prereg
    assert whole.ATTENTION_PARENT.name == "tensor_attention_projection_frontier_results.json"
    assert whole.EXACT_PARENT.name == "tensor_bilin18_standalone_identity_results.json"
