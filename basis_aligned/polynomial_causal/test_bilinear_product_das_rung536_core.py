import torch

import bilinear_product_das_rung536_core as core


def fixture():
    torch.manual_seed(536043)
    dtype = torch.float64
    token = torch.randn(19, 7, dtype=dtype)
    context = torch.randn(19, 7, dtype=dtype)
    donor_token = torch.randn(19, 7, dtype=dtype)
    donor_context = torch.randn(19, 7, dtype=dtype)
    left = torch.randn(17, 7, dtype=dtype)
    right = torch.randn(17, 7, dtype=dtype)
    down = torch.randn(9, 17, dtype=dtype)
    basis = torch.linalg.qr(torch.randn(17, 4, dtype=dtype), mode="reduced").Q
    return token, context, donor_token, donor_context, left, right, down, basis


def test_branches_close_full_product():
    token, context, _, _, left, right, _, _ = fixture()
    parts = core.product_branches(token, context, left, right)
    full = core.product_features(token, context, left, right)
    torch.testing.assert_close(parts["T"] + parts["I"] + parts["C"], full)


def test_token_hybrid_contains_exact_T_plus_I_changes():
    token, context, donor, _, left, right, _, _ = fixture()
    full_delta, target = core.token_hybrid_pair(token, context, donor, left, right)
    base = core.product_branches(token, context, left, right)
    changed = core.product_branches(donor, context, left, right)
    torch.testing.assert_close(full_delta, changed["T"] - base["T"] + changed["I"] - base["I"])
    torch.testing.assert_close(target, changed["T"] - base["T"])


def test_context_hybrid_contains_exact_I_plus_C_changes():
    token, context, _, donor, left, right, _, _ = fixture()
    full_delta, target = core.interaction_hybrid_pair(token, context, donor, left, right)
    base = core.product_branches(token, context, left, right)
    changed = core.product_branches(token, donor, left, right)
    torch.testing.assert_close(full_delta, changed["I"] - base["I"] + changed["C"] - base["C"])
    torch.testing.assert_close(target, changed["I"] - base["I"])


def test_compiled_basis_matches_direct_projected_output_and_gauge():
    token, context, _, _, left, right, down, basis = fixture()
    state = token + context
    product = core.product_features(token, context, left, right)
    direct = ((product @ basis) @ basis.T) @ down.T
    forms, directions = core.compile_basis(left, right, down, basis)
    compiled = core.compiled_output(state, forms, directions)
    torch.testing.assert_close(compiled, direct)

    rotation = torch.linalg.qr(torch.randn(4, 4, dtype=state.dtype), mode="reduced").Q
    forms_rotated, directions_rotated = core.compile_basis(left, right, down, basis @ rotation)
    compiled_rotated = core.compiled_output(state, forms_rotated, directions_rotated)
    torch.testing.assert_close(compiled_rotated, direct)
