from causal_response_tensor_split import document_side


def test_document_split_is_deterministic() -> None:
    first = [document_side(document_id) for document_id in range(100)]
    second = [document_side(document_id) for document_id in range(100)]
    assert first == second
    assert set(first) == {"FIT", "EVAL"}


def test_document_split_changes_with_seed() -> None:
    default = [document_side(document_id) for document_id in range(100)]
    alternate = [document_side(document_id, seed=185) for document_id in range(100)]
    assert default != alternate
