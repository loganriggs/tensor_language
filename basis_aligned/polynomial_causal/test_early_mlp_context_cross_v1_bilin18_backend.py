from __future__ import annotations

import torch

import compilation_mask_cut_rank_v1_bilin18_backend as parent
import compilation_mask_cut_rank_v1_gpu_adapter as adapter
import early_mlp_context_cross_v1_measurements as measurement
import early_mlp_context_cross_v1_statistics as statistics
import early_mlp_context_cross_v1_bilin18_backend as backend_module


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
        attention, first_value = self.attn(
            torch.nn.functional.rms_norm(value, (value.shape[-1],)), first_value,
        )
        value = value + attention
        value = value + self.mlp(
            torch.nn.functional.rms_norm(value, (value.shape[-1],))
        )
        return value, first_value


class TinyTransformer(torch.nn.Module):
    def __init__(self, dimensions):
        super().__init__()
        self.wte = torch.nn.Embedding(dimensions.logit_vocab, dimensions.model_width)
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
    return parent.ProgramDimensions(
        model_width=4, tokenizer_vocab=11, logit_vocab=11, layer_count=18,
        table_rank=2, map_rank=2, ridge=1e-2,
        expected_covered_token_count=6, build_batch_size=3, eval_batch_size=2,
    )


def _wave(*, row_count=3, offset=0):
    rows = (
        torch.arange(row_count * adapter.TARGET_STOP, dtype=torch.long)
        .reshape(row_count, adapter.TARGET_STOP)
        .add_(offset)
        .remainder_(6)
        .contiguous()
    )
    provenance = [
        {
            "document_id": f"doc-{offset}-{row // 2}",
            "dataset_document_index": offset + row // 2,
            "chunk_id": row,
            "token_start": row * adapter.TARGET_STOP,
        }
        for row in range(row_count)
    ]
    return adapter.RowWave(
        rows=rows, provenance=provenance,
        source_receipt_sha256=_hash(f"synthetic row receipt {offset}"),
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
        component_tree_sha256=parent.model_tree_sha256(model),
    )


def _tiny_backend():
    dimensions = _dimensions()
    model = TinyModel(dimensions)
    fit_wave = _wave(row_count=2, offset=0)
    calls = {"model": 0, "fit": 0}

    def load_model():
        calls["model"] += 1
        return model, _binding(model)

    def load_fit():
        calls["fit"] += 1
        return fit_wave

    backend = backend_module.Bilin18ContextCrossBackend(
        dimensions=dimensions, device="cpu", model_loader=load_model,
        fit_wave_loader=load_fit, expected_shared_program_sha256=None,
    )
    return backend, calls, model


def _role_rows():
    return {
        "skip7000": _wave(row_count=3, offset=1).clone_rows(),
        "skip11000": _wave(row_count=4, offset=2).clone_rows(),
    }


def test_create_backend_is_lazy_and_origin_mlp0_descriptors_are_physical():
    backend = backend_module.create_backend()
    assert backend._model is None and backend._program is None
    assert backend.expected_shared_program_sha256 == measurement.SHARED_PROGRAM_SHA256
    assert measurement.REQUESTS[0].sites == ()
    assert measurement.REQUESTS[8].sites == (("mlp", 0),)


def test_prepare_builds_one_program_and_binds_both_role_rows():
    backend, calls, _ = _tiny_backend()
    rows = _role_rows()
    bank = backend.prepare(rows, measurement.REQUESTS)
    assert calls == {"model": 1, "fit": 1}
    assert tuple(role for role, _ in bank.evaluation_role_row_sha256s) == (
        statistics.ROLE_NAMES
    )
    assert dict(bank.evaluation_role_row_sha256s) == {
        role: statistics.tensor_sha256(value) for role, value in rows.items()
    }
    assert bank.programs[0].installed_compiled_sites == ()
    assert bank.programs[8].installed_compiled_sites == (("mlp", 0),)
    assert len({program.shared_program_sha256 for program in bank.programs}) == 1
    assert backend.close() == bank.model.component_tree_sha256


def test_execute_origin_and_mlp0_have_exact_call_census_and_clear_hooks():
    backend, _, model = _tiny_backend()
    rows = _role_rows()
    bank = backend.prepare(rows, measurement.REQUESTS)
    origin = backend.execute_cell(
        "skip7000", measurement.REQUESTS[0], rows["skip7000"].clone(),
        bank.programs[0],
    )
    assert origin.call_ledger.native_module_calls == tuple(
        (site, 2) for site in adapter.ALL_NATIVE_SITES
    )
    assert origin.call_ledger.substitution_calls == ()
    mlp0 = backend.execute_cell(
        "skip11000", measurement.REQUESTS[8], rows["skip11000"].clone(),
        bank.programs[8],
    )
    assert mlp0.call_ledger.native_module_calls == tuple(
        (site, 2) for site in adapter.ALL_NATIVE_SITES
    )
    assert mlp0.call_ledger.substitution_calls == ((("mlp", 0), 2),)
    assert mlp0.statistics._values()[1].dtype == torch.float64
    assert all(
        not module._forward_hooks
        for block in model.transformer.h for module in (block.attn, block.mlp)
    )
    assert backend.close() == bank.model.component_tree_sha256


def test_wrong_role_rows_and_descriptor_fail_before_forward():
    backend, _, _ = _tiny_backend()
    rows = _role_rows()
    bank = backend.prepare(rows, measurement.REQUESTS)
    bad = rows["skip7000"].clone()
    bad[0, 0] = (bad[0, 0] + 1) % 6
    for arguments in (
        ("skip7000", measurement.REQUESTS[0], bad, bank.programs[0]),
        ("skip11000", measurement.REQUESTS[0], rows["skip7000"], bank.programs[0]),
        ("skip7000", measurement.REQUESTS[0], rows["skip7000"], bank.programs[1]),
    ):
        try:
            backend.execute_cell(*arguments)
        except RuntimeError as error:
            assert "differs from prepare" in str(error)
        else:
            raise AssertionError("malformed role cell was accepted")
    backend.close()
