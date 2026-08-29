from __future__ import annotations

import torch

import run_mlp2_rank512_refit_v1 as assay
import prepare_mlp2_rank512_refit_v1_rows as rows


def test_rank512_price_and_gauge_function_replay() -> None:
    generator = torch.Generator().manual_seed(7)
    model = assay.RankBilinear(
        torch.randn(assay.RANK, assay.WIDTH, generator=generator) * 0.01,
        torch.randn(assay.RANK, assay.WIDTH, generator=generator) * 0.01,
        torch.randn(assay.WIDTH, assay.RANK, generator=generator) * 0.01,
        torch.randn(assay.WIDTH, generator=generator) * 0.01,
    )
    receipt = assay.canonicalize_minimum_norm(model)
    assert model.price()["stored_scalar_values"] == 1_770_624
    assert model.price()["products"] == 512
    assert receipt["canary_max_abs_error"] < 1e-5


def test_document_reduction_native_identity() -> None:
    generator = torch.Generator().manual_seed(8)
    logits = torch.randn(2, 256, 13, generator=generator)
    targets = torch.randint(0, 13, (2, 256), generator=generator)
    reduced = assay.reduce_document(logits, logits.clone(), targets)
    assert reduced.shape == (2, 9)
    torch.testing.assert_close(reduced[:, 1], reduced[:, 0])
    torch.testing.assert_close(reduced[:, 2], torch.zeros(2, dtype=torch.float64), atol=1e-6, rtol=0)
    torch.testing.assert_close(reduced[:, 3], torch.zeros(2, dtype=torch.float64))
    assert torch.equal(reduced[:, 5], torch.full((2,), 192.0, dtype=torch.float64))


def test_row_split_is_role_disjoint() -> None:
    tensor = torch.arange(rows.TOTAL_DOCUMENTS * rows.TOKEN_LENGTH).reshape(
        rows.TOTAL_DOCUMENTS, rows.TOKEN_LENGTH,
    )
    records = [{"document_id": str(i), "dataset_document_index": i}
               for i in range(rows.TOTAL_DOCUMENTS)]
    split = rows.split_rows(tensor, records)
    assert split["TRAIN"][0].shape == (192, 257)
    assert split["EVALUATION"][0].shape == (192, 257)
    assert set(split["TRAIN"][1][0].values()).isdisjoint(
        set(split["EVALUATION"][1][0].values())
    )
