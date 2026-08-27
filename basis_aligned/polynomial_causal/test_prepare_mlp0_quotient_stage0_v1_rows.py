import importlib.util
from pathlib import Path

import pytest
import torch


PATH = Path(__file__).with_name("prepare_mlp0_quotient_stage0_v1_rows.py")
SPEC = importlib.util.spec_from_file_location("prepare_mlp0_rows_v1", PATH)
ROWS = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(ROWS)


def _record(document, index):
    return {"document_id": document, "dataset_document_index": index}


def test_eval_disjointness_checks_documents_rows_and_prefixes():
    fit = torch.arange(2 * 513, dtype=torch.long).view(2, 513)
    evaluate = fit + 5000
    gates = ROWS.validate_new_eval(
        fit, evaluate, [_record("fit-a", 1), _record("fit-b", 2)],
        [_record("eval-a", 20), _record("eval-b", 21)],
        ({"prior"}, {10}, {tuple(range(513))}, {tuple(range(32))}),
    )
    assert all(gates.values())

    bad = evaluate.clone()
    bad[0, :32] = fit[0, :32]
    with pytest.raises(RuntimeError, match="prefix32"):
        ROWS.validate_new_eval(
            fit, bad, [_record("fit-a", 1), _record("fit-b", 2)],
            [_record("eval-a", 20), _record("eval-b", 21)],
            (set(), set(), set(), set()),
        )


def test_eval_cannot_reuse_prior_document_index_even_if_label_differs():
    fit = torch.arange(513, dtype=torch.long).view(1, 513)
    evaluate = fit + 5000
    with pytest.raises(RuntimeError, match="document_index"):
        ROWS.validate_new_eval(
            fit, evaluate, [_record("fit", 1)], [_record("new-label", 20)],
            (set(), {20}, set(), set()),
        )
