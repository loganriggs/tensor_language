from __future__ import annotations

import torch

import tensor_bilin18_rank512_cross_task_validation as validation


def test_fresh_fixture_is_deterministic_supported_and_distinct() -> None:
    first = validation.fresh_deterministic_tokens(torch.device("cpu"))
    second = validation.fresh_deterministic_tokens(torch.device("cpu"))
    assert torch.equal(first, second)
    assert first.shape == (4, 256) and first.dtype == torch.long
    assert int(first.min()) >= 0 and int(first.max()) < 50_257


def test_cross_task_roles_are_authorized_and_hash_bound() -> None:
    for path in validation.ROLE_PATHS.values():
        receipt = validation.validate_role(path)
        assert receipt["shape"] == [192, 513]
        assert len(receipt["serialized_sha256"]) == 64
        assert len(receipt["tensor_raw_sha256"]) == 64


def test_protocol_binds_fresh_roles_fixture_and_parent() -> None:
    source = validation.Path(validation.__file__).read_text()
    prereg = validation.PREREG.read_text()
    for fragment in (
        "base.deterministic_tokens = fresh_deterministic_tokens",
        "frontier.EVAL_ROLES = ROLE_PATHS", "context_delta_recovery",
        "model_reference() is not None", "unseen_ce_harm", "os.O_EXCL",
    ):
        assert fragment in source
    assert "cross-task heldout" in prereg
    assert "skip31000" in prereg and "skip35000" in prereg
    assert validation.RANK512_PARENT.name == "tensor_bilin18_shared_qk_rank512_results.json"
