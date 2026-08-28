from __future__ import annotations

from dataclasses import replace

import pytest
import torch

import compilation_mask_cut_rank_v1_bilin18_backend as backend_module
import compilation_mask_cut_rank_v1_gpu_adapter as adapter
import compilation_mask_cut_rank_v1_measurements as measurement


def _hash(label: str) -> str:
    return backend_module._logical_sha256(label)


class TinyAttention(torch.nn.Module):
    def __init__(self, width: int, scale: float):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.eye(width) * scale, requires_grad=False)

    def forward(self, value, first_value=None):
        output = value @ self.weight
        return output, value if first_value is None else first_value


class TinyMLP(torch.nn.Module):
    def __init__(self, width: int, scale: float):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.eye(width) * scale, requires_grad=False)

    def forward(self, value):
        return torch.tanh(value @ self.weight)


class TinyBlock(torch.nn.Module):
    def __init__(self, width: int, layer: int):
        super().__init__()
        self.attn = TinyAttention(width, 0.01 * (layer + 1))
        self.mlp = TinyMLP(width, 0.005 * (layer + 1))

    def forward(self, value, first_value, initial):
        attention, first_value = self.attn(F_norm(value), first_value)
        value = value + attention
        value = value + self.mlp(F_norm(value))
        return value, first_value


def F_norm(value):
    return torch.nn.functional.rms_norm(value, (value.shape[-1],))


class TinyTransformer(torch.nn.Module):
    def __init__(self, dimensions):
        super().__init__()
        self.wte = torch.nn.Embedding(dimensions.tokenizer_vocab, dimensions.model_width)
        self.h = torch.nn.ModuleList([
            TinyBlock(dimensions.model_width, layer)
            for layer in range(dimensions.layer_count)
        ])


class TinyModel(torch.nn.Module):
    def __init__(self, dimensions):
        super().__init__()
        self.transformer = TinyTransformer(dimensions)
        self.lm_head = torch.nn.Linear(
            dimensions.model_width, dimensions.logit_vocab, bias=False,
        )
        generator = torch.Generator().manual_seed(20260828)
        with torch.no_grad():
            self.transformer.wte.weight.copy_(torch.randn(
                self.transformer.wte.weight.shape, generator=generator,
            ))
            self.lm_head.weight.copy_(torch.randn(
                self.lm_head.weight.shape, generator=generator,
            ))
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)


def _dimensions():
    return backend_module.ProgramDimensions(
        model_width=4, tokenizer_vocab=11, logit_vocab=11, layer_count=18,
        table_rank=2, map_rank=2, ridge=1e-2,
        expected_covered_token_count=6, build_batch_size=3, eval_batch_size=2,
    )


def _wave(*, row_count=3):
    rows = torch.arange(row_count * 257, dtype=torch.long).reshape(row_count, 257) % 6
    provenance = [
        {"document_id": "doc-a" if row < 2 else "doc-b",
         "dataset_document_index": row // 2, "chunk_id": row,
         "token_start": row * 257}
        for row in range(row_count)
    ]
    return adapter.RowWave(
        rows=rows.contiguous(), provenance=provenance,
        source_receipt_sha256=_hash("synthetic row receipt"),
    )


def _binding(model):
    config = _hash("tiny config")
    weights = _hash("tiny weights")
    implementation = _hash("tiny implementation")
    return adapter.ModelBinding(
        config_sha256=config, weights_sha256=weights,
        implementation_sha256=implementation,
        model_realization_sha256=backend_module._logical_sha256({
            "config_sha256": config, "weights_sha256": weights,
            "implementation_sha256": implementation,
        }),
        component_tree_sha256=backend_module.model_tree_sha256(model),
    )


def _tiny_backend(*, model=None, fit_wave=None, program_builder=None):
    dimensions = _dimensions()
    model = model or TinyModel(dimensions)
    fit_wave = fit_wave or _wave(row_count=2)
    calls = {"model": 0, "fit": 0, "builder": 0}

    def load_model():
        calls["model"] += 1
        return model, _binding(model)

    def load_fit():
        calls["fit"] += 1
        return fit_wave

    if program_builder is not None:
        def count_builder(*arguments):
            calls["builder"] += 1
            return program_builder(*arguments)
        builder = count_builder
    else:
        builder = None
    backend = backend_module.Bilin18CutRankBackend(
        dimensions=dimensions, device="cpu", model_loader=load_model,
        fit_wave_loader=load_fit, program_builder=builder,
    )
    return backend, calls, model


def test_create_backend_is_lazy_and_production_policy_is_exact():
    backend = backend_module.create_backend()
    assert backend._model is None and backend._program is None
    assert backend.batch_size == 8
    assert backend.source_paths == backend_module.SOURCE_PATHS
    assert backend.dimensions == backend_module.PRODUCTION_DIMENSIONS
    assert adapter.GAIN_POLICY == "identity_gains_no_mask_specific_refitting"


def test_centered_rank_truncation_is_deterministic_and_has_registered_rank():
    generator = torch.Generator().manual_seed(7)
    left = torch.randn(9, 2, generator=generator)
    right = torch.randn(2, 5, generator=generator)
    mean = torch.randn(1, 5, generator=generator)
    table = (mean + left @ right).float().contiguous()
    first = backend_module.centered_rank_truncate(table, 2)
    second = backend_module.centered_rank_truncate(table, 2)
    assert torch.equal(first, second)
    assert torch.linalg.matrix_rank((first - first.mean(0)).double(), tol=1e-5) <= 2
    assert torch.allclose(first, table, atol=2e-6, rtol=2e-6)
    with pytest.raises(ValueError, match="malformed"):
        backend_module.centered_rank_truncate(table, 0)


def test_ridge_map_and_token_materialization_keep_covered_rows_exact():
    embeddings = torch.tensor([
        [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0], [0.0, 1.0, 1.0],
    ])
    coefficient_true = torch.tensor([
        [2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.5],
    ])
    covered_ids = torch.tensor([0, 2, 4], dtype=torch.long)
    covered_embeddings = embeddings[covered_ids]
    covered_rows = covered_embeddings @ coefficient_true
    coefficient = backend_module.rank_truncated_ridge_map(
        covered_embeddings, covered_rows, rank=3, ridge=1e-8,
    )
    dense = backend_module.materialize_token_rows(
        token_embeddings=embeddings, covered_token_ids=covered_ids,
        covered_rows=covered_rows, coefficient=coefficient,
    )
    assert torch.equal(dense[covered_ids], covered_rows)
    assert torch.allclose(
        dense[[1, 3]], (embeddings[[1, 3]].double() @ coefficient).float(), atol=1e-6,
    )


def test_output_nearest_control_is_cosine_and_covered_ids_self_map():
    covered = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    all_rows = torch.tensor([[0.9, 0.1], [0.1, 0.9], [1.0, 1.0], [2.0, 0.0]])
    covered_ids = torch.tensor([1, 3], dtype=torch.long)
    nearest = backend_module.output_nearest_indices(
        covered_probabilities=covered, all_probabilities=all_rows,
        covered_token_ids=covered_ids,
    )
    assert torch.equal(nearest[covered_ids], torch.tensor([0, 1]))
    assert nearest[0] == 0 and nearest[2] == 0


def test_real_tiny_builder_freezes_actual_tensors_and_unique_identity_gain_masks():
    backend, calls, _ = _tiny_backend()
    evaluation = _wave().clone_rows()
    assert calls == {"model": 0, "fit": 0, "builder": 0}
    bank = backend.prepare(evaluation, measurement.REQUESTS)
    assert calls == {"model": 1, "fit": 1, "builder": 0}
    assert len(bank.programs) == 64 and len(set(bank.program_realization_sha256s)) == 64
    assert all(program.live_attention_gain_sites == () for program in bank.programs)
    assert all(program.gain_policy == adapter.GAIN_POLICY for program in bank.programs)
    assert len({program.shared_program_state_sha256 for program in bank.programs}) == 1
    assert backend._program.manifest["fallback_control"] == backend_module.FALLBACK_CONTROL
    assert backend._program.manifest["executed_uncovered_path"] == (
        backend_module.EXECUTED_UNCOVERED_PATH
    )
    assert len(backend._program.table_sha256s) == 36
    assert all(_hash_value == backend_module.tensor_content_sha256(
        backend._program.rows_for(site)
    ) for site, _hash_value in backend._program.table_sha256s)
    escaped_manifest = backend._program.manifest
    escaped_manifest["gain_policy"] = "changed"
    assert backend._program.manifest["gain_policy"] == adapter.GAIN_POLICY
    with pytest.raises(AttributeError, match="sealed"):
        backend._program.table_sha256s = ()
    component = backend.close()
    assert len(component) == 64


def test_execute_cell_uses_exact_hooks_counts_support_and_float64_ce():
    backend, _, model = _tiny_backend()
    rows = _wave().clone_rows()
    bank = backend.prepare(rows, measurement.REQUESTS)
    request = measurement.REQUESTS[9]
    program = bank.programs[9]
    result = backend.execute_cell(request, rows.clone(), program)
    result.call_ledger.validate(program, row_count=3, batch_count=2)
    assert result.statistics.row_count == 3
    _, ce, token_count = result.statistics._clone_values()
    assert ce.dtype == torch.float64 and bool(torch.isfinite(ce).all())
    assert torch.equal(token_count, torch.full((3,), 192, dtype=torch.long))
    assert result.call_ledger.live_attention_gain_calls == ()
    assert result.call_ledger.fitter_calls == 0
    assert result.call_ledger.retained_logits == 0
    assert all(not module._forward_hooks for block in model.transformer.h for module in (
        block.attn, block.mlp,
    ))
    assert backend.close() == bank.model.component_tree_sha256


def test_wrong_rows_descriptor_and_program_mutation_fail_closed():
    backend, _, _ = _tiny_backend()
    rows = _wave().clone_rows()
    bank = backend.prepare(rows, measurement.REQUESTS)
    bad_rows = rows.clone()
    bad_rows[0, 0] += 1
    with pytest.raises(RuntimeError, match="differs from prepare"):
        backend.execute_cell(measurement.REQUESTS[0], bad_rows, bank.programs[0])
    with pytest.raises(RuntimeError, match="differs from prepare"):
        backend.execute_cell(measurement.REQUESTS[0], rows, bank.programs[1])
    backend._program.rows_for(("attn", 0))[0, 0] += 1.0
    with pytest.raises(RuntimeError, match="program tensors changed"):
        backend.execute_cell(measurement.REQUESTS[0], rows, bank.programs[0])
    with pytest.raises(RuntimeError, match="program tensors changed"):
        backend.close()


def test_model_mutation_and_prepare_reuse_are_detected():
    backend, _, model = _tiny_backend()
    rows = _wave().clone_rows()
    bank = backend.prepare(rows, measurement.REQUESTS)
    with pytest.raises(RuntimeError, match="non-pristine"):
        backend.prepare(rows, measurement.REQUESTS)
    with torch.no_grad():
        model.lm_head.weight[0, 0] += 1.0
    with pytest.raises(RuntimeError, match="model state changed"):
        backend.execute_cell(measurement.REQUESTS[0], rows, bank.programs[0])
    with pytest.raises(RuntimeError, match="model state changed"):
        backend.close()


def test_shared_program_hash_covers_output_nearest_tables_and_fit_currency():
    backend, _, _ = _tiny_backend()
    rows = _wave().clone_rows()
    bank = backend.prepare(rows, measurement.REQUESTS)
    program = backend._program
    original = program.sha256
    changed_manifest = dict(program.manifest)
    changed_manifest["output_nearest_covered_index_sha256"] = _hash("different nearest")
    assert backend_module._logical_sha256(changed_manifest) != original
    changed_manifest = dict(program.manifest)
    changed_manifest["fit_wave_receipt"] = {
        **changed_manifest["fit_wave_receipt"], "row_tensor_sha256": _hash("different fit")
    }
    assert backend_module._logical_sha256(changed_manifest) != original
    assert all(descriptor.shared_program_state_sha256 == original for descriptor in bank.programs)
    backend.close()


def test_tensor_hash_rejects_alias_unsafe_inputs_and_binds_dtype_shape():
    value = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    digest = backend_module.tensor_content_sha256(value)
    assert digest != backend_module.tensor_content_sha256(value.to(torch.float64))
    assert digest != backend_module.tensor_content_sha256(value.reshape(2, 6))
    with pytest.raises(ValueError, match="contiguous"):
        backend_module.tensor_content_sha256(value.T)
    grad = value.clone().requires_grad_()
    with pytest.raises(ValueError, match="detached"):
        backend_module.tensor_content_sha256(grad)
