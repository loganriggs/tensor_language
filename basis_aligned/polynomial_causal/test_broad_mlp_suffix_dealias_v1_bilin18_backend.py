from __future__ import annotations

import pytest
import torch

import broad_mlp_suffix_dealias_v1 as assay
import broad_mlp_suffix_dealias_v1_bilin18_backend as backend_module
import broad_mlp_suffix_dealias_v1_measurements as measurement
import compilation_mask_cut_rank_v1_bilin18_backend as parent
import compilation_mask_cut_rank_v1_gpu_adapter as adapter
import early_mlp_context_cross_v1_statistics as statistics


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
        generator = torch.Generator().manual_seed(20260829)
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

    backend = backend_module.Bilin18BroadMLPSuffixBackend(
        dimensions=dimensions, device="cpu", model_loader=load_model,
        fit_wave_loader=load_fit, expected_shared_program_sha256=None,
    )
    return backend, calls, model


def _role_rows():
    return {
        "skip7000": _wave(row_count=3, offset=1).clone_rows(),
        "skip11000": _wave(row_count=4, offset=2).clone_rows(),
    }


def _all_hooks_are_clear(model) -> bool:
    return all(
        not module._forward_hooks
        for block in model.transformer.h for module in (block.attn, block.mlp)
    )


def test_create_backend_is_lazy_and_registry_is_exactly_eight_mlp_only_cells():
    backend = backend_module.create_backend()
    assert backend._model is None and backend._program is None
    assert backend.expected_shared_program_sha256 == measurement.SHARED_PROGRAM_SHA256
    assert len(measurement.REQUESTS) == assay.CELL_COUNT == 8
    assert measurement.REQUESTS[0].sites == assay.MLP_SUFFIX
    assert measurement.REQUESTS[1].sites == (("mlp", 0), *assay.MLP_SUFFIX)
    assert all(kind == "mlp" for request in measurement.REQUESTS for kind, _ in request.sites)


def test_prepare_reconstructs_one_parent_program_and_binds_both_role_rows():
    backend, calls, _ = _tiny_backend()
    rows = _role_rows()
    bank = backend.prepare(rows, measurement.REQUESTS)
    assert calls == {"model": 1, "fit": 1}
    assert len(bank.programs) == 8
    assert tuple(role for role, _ in bank.evaluation_role_row_sha256s) == assay.ROLE_NAMES
    assert dict(bank.evaluation_role_row_sha256s) == {
        role: statistics.tensor_sha256(value) for role, value in rows.items()
    }
    assert bank.programs[0].installed_compiled_sites == assay.MLP_SUFFIX
    assert bank.programs[1].installed_compiled_sites == (("mlp", 0), *assay.MLP_SUFFIX)
    assert len({program.shared_program_sha256 for program in bank.programs}) == 1
    assert backend._program.manifest["gain_policy"] == backend_module.GAIN_POLICY
    assert backend.verify_pre_outcome(bank) == (
        bank.model.component_tree_sha256, bank.shared_program_sha256,
    )
    assert backend.close() == bank.model.component_tree_sha256


def test_suffix_and_prefix_suffix_cells_have_exact_physical_call_ledgers():
    backend, _, model = _tiny_backend()
    rows = _role_rows()
    bank = backend.prepare(rows, measurement.REQUESTS)
    suffix = backend.execute_cell(
        "skip7000", measurement.REQUESTS[0], rows["skip7000"].clone(),
        bank.programs[0],
    )
    assert suffix.call_ledger.native_module_calls == tuple(
        (site, 2) for site in adapter.ALL_NATIVE_SITES
    )
    assert suffix.call_ledger.substitution_calls == tuple(
        (site, 2) for site in assay.MLP_SUFFIX
    )
    prefix_suffix = backend.execute_cell(
        "skip11000", measurement.REQUESTS[1], rows["skip11000"].clone(),
        bank.programs[1],
    )
    assert prefix_suffix.call_ledger.native_module_calls == tuple(
        (site, 2) for site in adapter.ALL_NATIVE_SITES
    )
    assert prefix_suffix.call_ledger.substitution_calls == tuple(
        (site, 2) for site in (("mlp", 0), *assay.MLP_SUFFIX)
    )
    assert prefix_suffix.statistics._values()[1].dtype == torch.float64
    assert suffix.call_ledger.fitter_calls == prefix_suffix.call_ledger.fitter_calls == 0
    assert suffix.call_ledger.retained_logits == prefix_suffix.call_ledger.retained_logits == 0
    assert _all_hooks_are_clear(model)
    assert backend.close() == bank.model.component_tree_sha256


def test_wrong_request_role_rows_and_descriptor_fail_before_forward():
    backend, _, model = _tiny_backend()
    rows = _role_rows()
    bank = backend.prepare(rows, measurement.REQUESTS)
    bad = rows["skip7000"].clone()
    bad[0, 0] = (bad[0, 0] + 1) % 6
    for arguments in (
        ("skip7000", measurement.REQUESTS[0], bad, bank.programs[0]),
        ("skip11000", measurement.REQUESTS[0], rows["skip7000"], bank.programs[0]),
        ("skip7000", measurement.REQUESTS[0], rows["skip7000"], bank.programs[1]),
    ):
        with pytest.raises(RuntimeError, match="differs from prepare"):
            backend.execute_cell(*arguments)
    assert _all_hooks_are_clear(model)
    backend.close()


def test_forward_failure_removes_every_native_and_substitution_hook():
    backend, _, model = _tiny_backend()
    rows = _role_rows()
    bank = backend.prepare(rows, measurement.REQUESTS)

    def fail(_tokens):
        raise RuntimeError("synthetic outer failure")

    backend._forward_logits = fail
    with pytest.raises(RuntimeError, match="synthetic outer failure"):
        backend.execute_cell(
            "skip7000", measurement.REQUESTS[7], rows["skip7000"], bank.programs[7],
        )
    assert backend._active_handles == []
    assert _all_hooks_are_clear(model)
    backend.close()


def test_prepare_rejects_reordered_or_incomplete_request_registry_before_loading():
    for requests in (measurement.REQUESTS[::-1], measurement.REQUESTS[:-1]):
        backend, calls, _ = _tiny_backend()
        with pytest.raises(RuntimeError, match="prepare is non-pristine or malformed"):
            backend.prepare(_role_rows(), requests)
        assert calls == {"model": 0, "fit": 0}
        backend.close()
