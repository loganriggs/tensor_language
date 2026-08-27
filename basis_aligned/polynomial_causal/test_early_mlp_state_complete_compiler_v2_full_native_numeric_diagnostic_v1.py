from __future__ import annotations

import json

import torch

import early_mlp_state_complete_compiler_v2_full_native_numeric_diagnostic_v1 as diagnostic


def test_diagnostic_pins_preserved_failure_and_is_nonauthorizing() -> None:
    assert diagnostic.file_sha256(diagnostic.FAILURE) == diagnostic.FAILURE_SHA256
    failure = json.loads(diagnostic.FAILURE.read_text())
    assert failure["status"] == "failed_compiler_v2_site0"
    assert all("diagnostic" in path.name for path in diagnostic.OUTPUTS)
    protocol = json.loads(diagnostic.PROTOCOL.read_text())
    assert protocol["frozen_tolerance"]["max_absolute_validation_row_ce_drift"] == 2e-6
    assert protocol["noncanonical_variants"].endswith("new preregistration.")


def test_projected_output_variants_have_registered_shapes() -> None:
    generator = torch.Generator().manual_seed(41)
    n, d, k, c = 3, diagnostic.compiler.D_MODEL, 5, 2
    z = torch.randn(n, d, generator=generator)
    left = torch.randn(k, d, generator=generator)
    right = torch.randn(k, d, generator=generator)
    q = torch.randn(k, c, generator=generator)
    beta = torch.randn(c, generator=generator)
    native = {
        "left": left, "right": right, "projected_decoder": q,
        "beta": beta, "mode": "native32",
    }
    expected = ((z @ left.T) * (z @ right.T)) @ q + beta
    assert torch.equal(diagnostic.projected_output(z, native), expected)


def test_source_closure_contains_diagnostic_test_and_failed_site0_closure() -> None:
    names = {path.name for path in diagnostic.SOURCE_CLOSURE}
    assert "test_early_mlp_state_complete_compiler_v2_full_native_numeric_diagnostic_v1.py" in names
    assert "early_mlp_state_complete_compiler_v2_site0.py" in names
