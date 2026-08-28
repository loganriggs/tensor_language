import pytest
import torch

import early_mlp_suffix_transport_v1_runtime as runtime


def basis():
    return torch.eye(runtime.D_MODEL, dtype=torch.float32)[:, :runtime.CODE_DIM]


def state(seed: int):
    generator = torch.Generator().manual_seed(seed)
    return {
        "grammar": "affine", "interface": "state_complete_p",
        "mean": torch.randn(runtime.D_MODEL, generator=generator),
        "scale": torch.rand(runtime.D_MODEL, generator=generator) + 0.5,
        "left": torch.randn(runtime.D_MODEL, runtime.CODE_DIM, generator=generator) / 20,
        "right": torch.randn(runtime.CODE_DIM, runtime.CODE_DIM, generator=generator) / 20,
        "bias": torch.randn(runtime.CODE_DIM, generator=generator) / 20,
    }


def joint(route="L"):
    return runtime.JointAffineProgram.from_v21_states(
        {0: state(1), 1: state(2)}, route=route,
    )


def student_hook():
    return runtime.StudentCorrectionHook(
        {0: basis(), 1: basis()}, issuer_id="a" * 64,
        coordinator=runtime.ScopeCoordinator(),
    )


def call_hook(hook, site, z, mo, nonce):
    deployed_n = runtime.mint_deployed_n_write(
        site=site, state=z, value=mo, forward_nonce=nonce,
        issuer_id=hook.issuer_id,
    )
    return hook(site, z, deployed_n, forward_nonce=nonce)


def trace_identity(hook, ordinal: int, *, route: str | None = None):
    requested = route or (hook.program.route if hook.program is not None else "L")
    teacher = "coordinate_labels" if requested in {"Q", "L"} else "oon_logits"
    inputs = torch.arange(4 * 256, dtype=torch.long).view(4, 256) + ordinal
    return runtime.TraceIdentity.from_inputs(
        inputs=inputs, ordered_batch_indices=range(4), source_commit="b" * 40,
        inherited_snapshot_sha256="c" * 64, rows_receipt_sha256="d" * 64,
        fit_role_tensor_sha256="e" * 64,
        program_snapshot_sha256=runtime.program_snapshot_sha256(hook.program),
        teacher_mapping_sha256="f" * 64, phase="fit", route=requested,
        control="true", teacher_kind=teacher, trial=0, epoch=0,
        optimizer_step=ordinal, batch_ordinal=ordinal,
        student_states=tuple((site, hook.states.get(site, "N")) for site in (0, 1))
        + ((2, "N"),),
    )


def test_v21_initialization_is_exact_full_product_affine():
    source = state(3)
    program = runtime.AffineCodeProgram.from_v21_state(source)
    z = torch.randn(2, 5, runtime.D_MODEL)
    expected = ((z - source["mean"]) / source["scale"]) @ (
        source["left"] @ source["right"]
    ) + source["bias"]
    assert torch.allclose(program(z), expected, atol=2e-5, rtol=2e-6)
    assert not program.mean.requires_grad and not program.scale.requires_grad
    assert program.weight.requires_grad and program.bias.requires_grad


def test_projected_replacement_sets_code_and_preserves_complement():
    deployed = torch.randn(2, 3, runtime.D_MODEL)
    code = torch.randn(2, 3, runtime.CODE_DIM)
    b = basis()
    replaced = runtime.JointAffineProgram.projected_replacement(deployed, code, b)
    assert torch.allclose(replaced.float() @ b, code, atol=2e-5, rtol=0)
    complement = deployed.clone(); complement[..., :runtime.CODE_DIM] = 0
    observed = replaced.clone(); observed[..., :runtime.CODE_DIM] = 0
    assert torch.equal(observed, complement)


def test_student_rejects_raw_and_native_o_writes() -> None:
    with pytest.raises(TypeError, match="cannot be subclassed"):
        class Forged(runtime.DeployedNWrite):
            pass

    hook = student_hook()
    hook.configure(program=joint("L"), states={0: "P", 1: "P"})
    identity = trace_identity(hook, 0)
    z = torch.randn(4, 256, runtime.D_MODEL)
    with pytest.raises(TypeError, match=r"typed deployed N"):
        with hook.forward_scope(identity):
            hook(0, z, z, forward_nonce=identity.nonce)

    hook.clear_configuration()
    hook.configure(program=joint("L"), states={0: "P", 1: "P"})
    identity = trace_identity(hook, 1)
    with pytest.raises(TypeError, match=r"typed deployed N"):
        with hook.forward_scope(identity):
            hook(0, z, runtime.NativeOWrite(z), forward_nonce=identity.nonce)


def test_deployed_n_write_rejects_cross_site_state_nonce_and_replay() -> None:
    hook = student_hook()
    hook.configure(program=joint("L"), states={0: "P", 1: "P"})
    identity = trace_identity(hook, 2)
    z0 = torch.randn(4, 256, runtime.D_MODEL)
    z1 = torch.randn(4, 256, runtime.D_MODEL)
    write0 = runtime.mint_deployed_n_write(
        site=0, state=z0, value=z0, forward_nonce=identity.nonce,
        issuer_id=hook.issuer_id,
    )
    with pytest.raises(AttributeError, match="sealed"):
        write0._DeployedNWrite__site = 1
    with pytest.raises(RuntimeError, match="site/state/forward"):
        with hook.forward_scope(identity):
            hook(1, z1, write0, forward_nonce=identity.nonce)

    hook.clear_configuration()
    hook.configure(program=joint("L"), states={0: "P", 1: "P"})
    identity = trace_identity(hook, 3)
    write0 = runtime.mint_deployed_n_write(
        site=0, state=z0, value=z0, forward_nonce=identity.nonce,
        issuer_id=hook.issuer_id,
    )
    with pytest.raises(RuntimeError, match="already consumed"):
        with hook.forward_scope(identity):
            hook(0, z0, write0, forward_nonce=identity.nonce)
            hook(1, z1, write0, forward_nonce=identity.nonce)


def test_deployed_n_write_rejects_tensor_alias_mutation() -> None:
    hook = student_hook()
    hook.configure(program=joint("L"), states={0: "P", 1: "P"})
    identity = trace_identity(hook, 4)
    z = torch.randn(4, 256, runtime.D_MODEL)
    deployed = z.clone()
    write = runtime.mint_deployed_n_write(
        site=0, state=z, value=deployed, forward_nonce=identity.nonce,
        issuer_id=hook.issuer_id,
    )
    deployed.add_(1.0)
    with hook.forward_scope(identity):
        observed = hook(0, z, write, forward_nonce=identity.nonce)
    assert torch.isfinite(observed).all()

    hook.clear_configuration()
    hook.configure(program=joint("L"), states={0: "P", 1: "P"})
    identity = trace_identity(hook, 5)
    deployed = z.clone()
    original = deployed.clone()
    write = runtime.mint_deployed_n_write(
        site=0, state=z, value=deployed, forward_nonce=identity.nonce,
        issuer_id=hook.issuer_id,
    )
    deployed.data.fill_(float("nan"))
    with hook.forward_scope(identity):
        observed = hook(0, z, write, forward_nonce=identity.nonce)
    assert torch.isfinite(observed).all()
    assert torch.equal(observed[..., runtime.CODE_DIM:], original[..., runtime.CODE_DIM:])

    hook.clear_configuration()
    hook.configure(program=joint("L"), states={0: "P", 1: "P"})
    identity = trace_identity(hook, 6)
    z2 = z.clone()
    write = runtime.mint_deployed_n_write(
        site=0, state=z2, value=z2.clone(), forward_nonce=identity.nonce,
        issuer_id=hook.issuer_id,
    )
    z2.data.add_(3.0)
    with pytest.raises(RuntimeError, match="state content mutated after mint"):
        with hook.forward_scope(identity):
            hook(0, z2, write, forward_nonce=identity.nonce)


def test_transport_uses_executable_parent_but_only_cross_is_trainable():
    program = joint("T")
    parameters = program.set_route_trainability()
    assert parameters == (program.cross,)
    z0 = torch.randn(1, 2, runtime.D_MODEL)
    z1 = torch.randn(1, 2, runtime.D_MODEL)
    parent = program.site0_code(z0)
    zero = program.site1_code(z1, parent)
    assert torch.equal(zero, program.site1(z1))
    with torch.no_grad():
        program.cross.copy_(torch.eye(runtime.CODE_DIM))
    transported = program.site1_code(z1, parent)
    assert torch.equal(transported, program.site1(z1) + parent)
    transported.square().mean().backward()
    assert program.cross.grad is not None and float(program.cross.grad.abs().sum()) > 0
    assert program.site0.weight.grad is None and program.site1.weight.grad is None


def test_mapped_transport_writes_source_code_but_cross_reads_false_parent() -> None:
    program = joint("T")
    with torch.no_grad():
        program.cross.copy_(torch.eye(runtime.CODE_DIM))
    hook = student_hook()
    inputs = torch.arange(4 * 256, dtype=torch.long).view(4, 256)
    identity = runtime.TraceIdentity.from_inputs(
        inputs=inputs, ordered_batch_indices=range(4), source_commit="b" * 40,
        inherited_snapshot_sha256="c" * 64, rows_receipt_sha256="d" * 64,
        fit_role_tensor_sha256="e" * 64,
        program_snapshot_sha256=runtime.program_snapshot_sha256(program),
        teacher_mapping_sha256="f" * 64, phase="fit", route="T",
        control="A_null_00", teacher_kind="oon_logits", trial=0, epoch=0,
        optimizer_step=0, batch_ordinal=0,
        student_states=((0, "P"), (1, "P"), (2, "N")),
    )
    mapped = torch.full(
        (runtime.BATCH_SIZE, runtime.SEQUENCE_LENGTH, runtime.CODE_DIM), 3.0,
    )
    released = []
    handle = runtime.MappedParentCode(
        value=mapped, identity_sha256=identity.sha256, issuer_id=hook.issuer_id,
        program_sha256=runtime.program_snapshot_sha256(program),
        release=released.append,
    )
    hook.configure(
        program=program, states={0: "P", 1: "P"}, mapped_parent=handle,
    )
    assert released == [identity.sha256] and hook.has_mapped_parent
    state0 = torch.randn(4, 256, runtime.D_MODEL)
    state1 = torch.randn(4, 256, runtime.D_MODEL)
    deployed0, deployed1 = torch.randn_like(state0), torch.randn_like(state1)
    with hook.forward_scope(identity):
        out0 = call_hook(hook, 0, state0, deployed0, identity.nonce)
        out1 = call_hook(hook, 1, state1, deployed1, identity.nonce)
    torch.testing.assert_close(
        out0.float() @ basis(), program.site0_code(state0), atol=2e-5, rtol=0,
    )
    torch.testing.assert_close(
        out1.float() @ basis(), program.site1(state1) + mapped,
        atol=2e-5, rtol=0,
    )
    assert not hook.has_mapped_parent and hook.parent_code is None
    with pytest.raises(RuntimeError, match="already consumed"):
        handle._consume(
            issuer_id=hook.issuer_id,
            program_sha256=runtime.program_snapshot_sha256(program),
        )


def test_local_and_zero_transport_have_disjoint_equal_initializations():
    local = joint("L")
    transported = local.independent_clone(route="T")
    z0 = torch.randn(1, 2, runtime.D_MODEL)
    z1 = torch.randn(1, 2, runtime.D_MODEL)
    assert torch.equal(local.site1_code(z1), transported.site1_code(z1, transported.site0_code(z0)))
    assert local.site0.weight.data_ptr() != transported.site0.weight.data_ptr()
    assert transported.cross is not None and int(torch.count_nonzero(transported.cross)) == 0


@pytest.mark.parametrize("route,expected", [
    ("L", ("site0.weight", "site0.bias", "site1.weight", "site1.bias")),
    ("R", ("site0.weight", "site0.bias", "site1.weight", "site1.bias")),
    ("S0", ("site0.weight", "site0.bias")),
    ("S1", ("site1.weight", "site1.bias")),
    ("T", ("cross",)),
])
def test_route_identity_binds_execution_and_trainability(route, expected):
    program = joint(route)
    selected = program.set_route_trainability()
    by_id = {id(value): name for name, value in program.named_parameters()}
    assert tuple(by_id[id(value)] for value in selected) == expected
    assert tuple(name for name, value in program.named_parameters() if value.requires_grad) == expected
    if route == "T":
        assert program.cross is not None
    else:
        assert program.cross is None
        with pytest.raises(ValueError, match="cannot consume"):
            program.site1_code(
                torch.randn(1, 2, runtime.D_MODEL),
                torch.randn(1, 2, runtime.CODE_DIM),
            )


def test_route_and_cross_topology_reject_public_or_internal_mutation():
    program = joint("T")
    with pytest.raises(AttributeError, match="immutable"):
        program.route = "L"
    with pytest.raises(AttributeError, match="immutable"):
        program._route = "L"
    with pytest.raises(AttributeError, match="immutable"):
        program.cross = None
    saved = program._parameters["cross"]
    program._parameters["cross"] = None
    with pytest.raises(RuntimeError, match="topology"):
        program.site0_code(torch.randn(1, 2, runtime.D_MODEL))
    program._parameters["cross"] = saved


def test_student_hook_has_no_original_capability_and_consumes_one_parent():
    program = joint("T")
    hook = student_hook()
    z0 = torch.randn(4, 256, runtime.D_MODEL)
    z1 = torch.randn(4, 256, runtime.D_MODEL)
    mo0, mo1 = torch.randn_like(z0), torch.randn_like(z1)
    hook.configure(program=program, states={0: "P", 1: "P"})
    identity1 = trace_identity(hook, 1)
    with hook.forward_scope(identity1, capture_sites={0, 1}):
        out0 = call_hook(hook, 0, z0, mo0, identity1.nonce)
        out1 = call_hook(hook, 1, z1, mo1, identity1.nonce)
        with pytest.raises(RuntimeError, match="more than once"):
            call_hook(hook, 1, z1, mo1, identity1.nonce)
    assert torch.allclose(out0.float() @ basis(), program.site0_code(z0), atol=2e-5, rtol=0)
    assert torch.allclose(
        out1.float() @ basis(), program.site1_code(z1, program.site0_code(z0)),
        atol=2e-5, rtol=0,
    )
    trace = hook.pop_trace(identity1)
    captured = trace._consume(issuer_id="a" * 64, identity=identity1)
    assert captured[0].grad_fn is None and captured[1].grad_fn is None
    assert hook.parent_code is None
    identity2 = trace_identity(hook, 2)
    with hook.forward_scope(identity2):
        call_hook(hook, 0, z0, mo0, identity2.nonce)
        call_hook(hook, 1, z1, mo1, identity2.nonce)
    assert hook.calls == {0: 2, 1: 2}


def test_student_hook_rejects_missing_stale_native_or_overwritten_parent():
    z = torch.randn(1, 256, runtime.D_MODEL)
    mo = torch.randn_like(z)
    program = joint("T")
    hook = student_hook()
    hook.configure(program=program, states={0: "P", 1: "P"})
    one = trace_identity(hook, 3)
    with hook.forward_scope(one), pytest.raises(RuntimeError, match="lacks one unused"):
        call_hook(hook, 1, z, mo, one.nonce)
    hook.configure(program=program, states={0: "N", 1: "P"})
    with pytest.raises(ValueError, match="P/P/N"):
        trace_identity(hook, 4)
    hook.configure(program=program, states={0: "P", 1: "P"})
    three = trace_identity(hook, 5)
    with hook.forward_scope(three):
        call_hook(hook, 0, z, mo, three.nonce)
        with pytest.raises(RuntimeError, match="outside a forward scope"):
            call_hook(hook, 1, z, mo, "f" * 64)
        with pytest.raises(RuntimeError, match="more than once"):
            call_hook(hook, 0, z, mo, three.nonce)


def test_student_hook_applies_edit_to_executable_parent_only():
    program = joint("T")
    hook = student_hook()
    z = torch.randn(4, 256, runtime.D_MODEL)
    mo = torch.randn_like(z)
    edit = torch.randn(1, 256, runtime.CODE_DIM)
    edit = edit.expand(4, -1, -1).clone()
    hook.configure(program=program, states={0: "P", 1: "P"}, site0_edit=edit)
    identity = trace_identity(hook, 6, route="T")
    with hook.forward_scope(identity):
        returned = call_hook(hook, 0, z, mo, identity.nonce)
        parent = hook.parent_code.detach().clone()
    expected = program.site0_code(z) + edit
    assert torch.allclose(parent, expected, atol=1e-6, rtol=0)
    assert torch.allclose(returned.float() @ basis(), expected, atol=2e-5, rtol=0)


def test_student_hook_rejects_silently_ignored_edit():
    hook = student_hook()
    edit = torch.zeros(1, 256, runtime.CODE_DIM)
    with pytest.raises(ValueError, match="executable predicted"):
        hook.configure(program=joint("L"), states={0: "N"}, site0_edit=edit)
    with pytest.raises(ValueError, match="executable predicted"):
        hook.configure(program=None, states={0: "N"}, site0_edit=edit)


def test_forward_local_trace_rejects_missing_exception_replay_and_outstanding():
    hook = student_hook()
    prog = joint("L")
    hook.configure(program=prog, states={0: "P", 1: "P"})
    z = torch.zeros(4, 256, runtime.D_MODEL)
    identity1 = trace_identity(hook, 20)
    with pytest.raises(RuntimeError, match="did not call/capture"):
        with hook.forward_scope(identity1, capture_sites={0, 1}):
            call_hook(hook, 0, z, z, identity1.nonce)
    with pytest.raises(RuntimeError, match="no completed"):
        hook.pop_trace(identity1)
    with pytest.raises(RuntimeError, match="already spent"):
        with hook.forward_scope(identity1, capture_sites={0, 1}):
            pass

    identity2 = trace_identity(hook, 21)
    with pytest.raises(ValueError, match="synthetic"):
        with hook.forward_scope(identity2, capture_sites={0, 1}):
            call_hook(hook, 0, z, z, identity2.nonce)
            raise ValueError("synthetic")
    with pytest.raises(RuntimeError, match="no completed"):
        hook.pop_trace(identity2)

    identity3 = trace_identity(hook, 22)
    with hook.forward_scope(identity3, capture_sites={0, 1}):
        call_hook(hook, 0, z, z, identity3.nonce)
        call_hook(hook, 1, z, z, identity3.nonce)
    trace = hook.pop_trace(identity3)
    with pytest.raises(RuntimeError, match="outstanding"):
        with hook.forward_scope(trace_identity(hook, 23), capture_sites={0, 1}):
            pass
    with pytest.raises(RuntimeError, match="active trace"):
        hook.configure(program=prog, states={0: "P", 1: "P"})
    trace._consume(issuer_id="a" * 64, identity=identity3)
    hook.configure(program=prog, states={0: "P", 1: "P"})


def test_trace_clones_state_and_rejects_program_or_basis_mutation():
    hook = student_hook()
    prog = joint("L")
    hook.configure(program=prog, states={0: "P", 1: "P"})
    z0 = torch.randn(4, 256, runtime.D_MODEL)
    z1 = torch.randn(4, 256, runtime.D_MODEL)
    original0 = z0.clone()
    identity1 = trace_identity(hook, 24)
    with hook.forward_scope(identity1, capture_sites={0, 1}):
        call_hook(hook, 0, z0, z0, identity1.nonce)
        z0.add_(100)
        call_hook(hook, 1, z1, z1, identity1.nonce)
    trace = hook.pop_trace(identity1)
    captured = trace._consume(issuer_id="a" * 64, identity=identity1)
    assert torch.equal(captured[0], original0)

    identity2 = trace_identity(hook, 25)
    with pytest.raises(RuntimeError, match="program mutated"):
        with hook.forward_scope(identity2, capture_sites={0, 1}):
            call_hook(hook, 0, z1, z1, identity2.nonce)
            call_hook(hook, 1, z1, z1, identity2.nonce)
            with torch.no_grad():
                prog.site0.bias.add_(1)

    hook.configure(program=prog, states={0: "P", 1: "P"})
    identity3 = trace_identity(hook, 26)
    with pytest.raises(RuntimeError, match="basis mutated"):
        with hook.forward_scope(identity3, capture_sites={0, 1}):
            call_hook(hook, 0, z1, z1, identity3.nonce)
            call_hook(hook, 1, z1, z1, identity3.nonce)
            hook._bases[0][0, 0] += 1


def fit_labels(seed=4):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(384, 192, runtime.CODE_DIM, dtype=torch.float64, generator=generator)


def test_local_loss_detaches_labels_and_enforces_scored_support():
    y0, y1 = fit_labels(5)[:4].requires_grad_(), fit_labels(6)[:4].requires_grad_()
    p0 = torch.randn_like(y0, dtype=torch.float32, requires_grad=True)
    p1 = torch.randn_like(y1, dtype=torch.float32, requires_grad=True)
    d0 = runtime.centered_second_moment(fit_labels(5), ordered_support_sha256="a" * 64)
    d1 = runtime.centered_second_moment(fit_labels(6), ordered_support_sha256="b" * 64)
    loss = runtime.normalized_local_loss((p0, p1), (y0, y1), (d0, d1))
    expected = torch.mean((p0 - y0.detach().float()).square()) / d0.float() + \
        torch.mean((p1 - y1.detach().float()).square()) / d1.float()
    assert torch.equal(loss, expected)
    loss.backward()
    assert p0.grad is not None and p1.grad is not None
    assert y0.grad is None and y1.grad is None
    with pytest.raises(ValueError, match="support"):
        runtime.normalized_local_loss(
            (p0[:, :191], p1[:, :191]), (y0[:, :191], y1[:, :191]), (d0, d1),
        )


def test_moment_statistics_merge_stably_and_bind_fit_support():
    values = fit_labels(7)
    direct = runtime.MomentSufficientStatistics.from_labels(values).finalize(
        expected_count=384 * 192, ordered_support_sha256="c" * 64,
    )
    left = runtime.MomentSufficientStatistics.from_labels(values[:192])
    right = runtime.MomentSufficientStatistics.from_labels(values[192:])
    merged = left.merge(right).finalize(
        expected_count=384 * 192, ordered_support_sha256="c" * 64,
    )
    assert merged["count"] == 384 * 192
    for key in ("coordinate_sum", "coordinate_square_sum", "mean",
                "centered_sum_of_squares", "denominator"):
        torch.testing.assert_close(merged[key], direct[key], atol=2e-11, rtol=2e-14)
    with pytest.raises(ValueError, match="count"):
        left.finalize(expected_count=384 * 192, ordered_support_sha256="c" * 64)


def test_moment_chan_accumulation_is_stable_for_high_offset_labels():
    generator = torch.Generator().manual_seed(8)
    values = 1e6 + torch.randn(384, 192, runtime.CODE_DIM, dtype=torch.float64,
                              generator=generator)
    result = runtime.MomentSufficientStatistics.from_labels(values).finalize(
        expected_count=384 * 192, ordered_support_sha256="d" * 64,
    )
    flat = values.reshape(-1, runtime.CODE_DIM)
    expected = torch.sum((flat - flat.mean(0)).square()) / flat.numel()
    torch.testing.assert_close(result["denominator"], expected, atol=2e-11, rtol=2e-13)


def test_suffix_kl_orientation_detach_weighting_and_support():
    teacher = torch.randn(3, 256, 7, requires_grad=True)
    student = torch.randn(3, 256, 7, requires_grad=True)
    loss = runtime.teacher_student_kl(teacher, student)
    t = torch.log_softmax(teacher.detach()[:, 64:256], dim=-1)
    s = torch.log_softmax(student[:, 64:256], dim=-1)
    assert torch.equal(loss, torch.mean(torch.sum(t.exp() * (t - s), dim=-1)))
    loss.backward()
    assert teacher.grad is None and student.grad is not None
    first = runtime.teacher_student_kl(teacher[:1].detach(), student[:1].detach())
    second = runtime.teacher_student_kl(teacher[1:].detach(), student[1:].detach())
    assert torch.allclose(loss.detach(), (first + 2 * second) / 3)
    with pytest.raises(ValueError, match="support"):
        runtime.teacher_student_kl(teacher[:, :191], student[:, :191])


def test_scored_positions_are_exactly_64_through_255():
    value = torch.arange(2 * 256 * 3).view(2, 256, 3)
    scored = runtime.scored_positions(value)
    assert tuple(scored.shape) == (2, 192, 3)
    assert torch.equal(scored[:, 0], value[:, 64])
    assert torch.equal(scored[:, -1], value[:, 255])
    with pytest.raises(ValueError, match="256"):
        runtime.scored_positions(torch.zeros(1, 257, 3))


def test_fit_schedule_and_optimizer_are_frozen_and_reproducible():
    assert all(torch.equal(a, b) for a, b in zip(
        runtime.fit_permutations(384, 2), runtime.fit_permutations(384, 2), strict=True,
    ))
    assert len(runtime.batch_indices(384, 2)) == runtime.EPOCHS * 96
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = runtime.make_optimizer([parameter], 3e-5)
    group = optimizer.param_groups[0]
    assert group["lr"] == 3e-5 and group["betas"] == (0.9, 0.999)
    assert group["eps"] == 1e-8 and group["weight_decay"] == 0
    assert runtime.optimizer_step(parameter.square(), optimizer) == pytest.approx(2.0)
