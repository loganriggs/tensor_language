from __future__ import annotations

import pytest
import torch

import early_mlp_suffix_transport_v1_capabilities as capabilities
import early_mlp_suffix_transport_v1_mapped as mapped
import early_mlp_suffix_transport_v1_runtime as runtime


def _records(spec=(2, 2, 1, 1, 1)):
    records = []
    for document_index, count in enumerate(spec):
        for chunk in range(count):
            records.append({
                "document_id": f"doc-{document_index}",
                "dataset_document_index": 100 + document_index,
                "chunk_id": chunk,
                "token_start": chunk * 256,
            })
    return records


def test_document_plan_is_deterministic_block_bijection_without_fixed_documents() -> None:
    records = _records()
    plan = mapped.build_document_block_plan(records, control="document_shuffle")
    replay = mapped.build_document_block_plan(records, control="document_shuffle")
    assert plan == replay and plan.seed == mapped.DOCUMENT_SHUFFLE_SEED
    assert sorted(plan.row_targets) == list(range(len(records)))
    assert all(
        source != target
        for source, target in zip(plan.source_documents, plan.target_documents, strict=True)
    )
    # Both rows from each two-row document move together and preserve within-doc order.
    assert plan.row_targets[1] == plan.row_targets[0] + 1
    assert plan.row_targets[3] == plan.row_targets[2] + 1
    assert {(item.rows_per_document, len(item.documents)) for item in plan.strata} == {
        (1, 3), (2, 2),
    }


def test_all_registered_control_seeds_and_degenerate_strata_fail_closed() -> None:
    records = _records()
    nulls = [
        mapped.build_document_block_plan(records, control=f"A_null_{index:02d}")
        for index in range(20)
    ]
    assert [plan.seed for plan in nulls] == list(range(2026083100, 2026083120))
    assert all(plan.control == f"A_null_{index:02d}" for index, plan in enumerate(nulls))
    with pytest.raises(ValueError, match="A_null"):
        mapped.control_seed("A_null_20")
    with pytest.raises(RuntimeError, match="at least two"):
        mapped.build_document_block_plan(_records((2, 1, 1)), control="document_shuffle")


def test_interleaved_documents_and_bad_provenance_are_rejected() -> None:
    records = _records((2, 2))
    interleaved = [records[0], records[2], records[1], records[3]]
    with pytest.raises(RuntimeError, match="not contiguous"):
        mapped.build_document_block_plan(interleaved, control="document_shuffle")
    malformed = [dict(record) for record in records]
    malformed[0]["extra"] = 1
    with pytest.raises(ValueError, match="schema"):
        mapped.build_document_block_plan(malformed, control="document_shuffle")


def _program(route: str) -> runtime.JointAffineProgram:
    state = {
        site: {
            "grammar": "affine", "interface": "state_complete_p",
            "mean": torch.zeros(runtime.D_MODEL), "scale": torch.ones(runtime.D_MODEL),
            "left": torch.zeros(runtime.D_MODEL, runtime.CODE_DIM),
            "right": torch.zeros(runtime.CODE_DIM, runtime.CODE_DIM),
            "bias": torch.zeros(runtime.CODE_DIM),
        }
        for site in (0, 1)
    }
    return runtime.JointAffineProgram.from_v21_states(state, route=route)


def test_mapped_context_binds_source_schedule_plan_and_target_tokens(monkeypatch) -> None:
    monkeypatch.setattr(capabilities, "FIT_ROW_COUNT", runtime.BATCH_SIZE)
    monkeypatch.setattr(capabilities, "FIT_BATCHES_PER_EPOCH", 1)
    rows = torch.arange(
        runtime.BATCH_SIZE * 513, dtype=torch.long,
    ).view(runtime.BATCH_SIZE, 513)
    records = _records((1, 1, 1, 1))
    plan = mapped.build_document_block_plan(records, control="document_shuffle")
    base = capabilities.RunContext(
        source_commit="1" * 40, inherited_snapshot_sha256="2" * 64,
        rows_receipt_sha256="3" * 64,
        fit_role_tensor_sha256=runtime.tensor_identity_sha256(rows),
        identity_teacher_mapping_sha256="4" * 64,
        fit_row_count=runtime.BATCH_SIZE,
    )
    context = mapped.MappedRunContext(base=base, plan=plan)
    source_indices = tuple(int(value) for value in runtime.fit_permutations(4, 0)[0])
    source = rows[torch.tensor(source_indices), :runtime.SEQUENCE_LENGTH]
    targets = plan.target_indices(source_indices)
    teacher = rows[torch.tensor(targets), :runtime.SEQUENCE_LENGTH]
    program = _program("L")
    identity = runtime.TraceIdentity.from_inputs(
        inputs=source, ordered_batch_indices=source_indices,
        source_commit=base.source_commit,
        inherited_snapshot_sha256=base.inherited_snapshot_sha256,
        rows_receipt_sha256=base.rows_receipt_sha256,
        fit_role_tensor_sha256=base.fit_role_tensor_sha256,
        program_snapshot_sha256=runtime.program_snapshot_sha256(program),
        teacher_mapping_sha256=plan.sha256, phase="fit", route="L",
        control="document_shuffle", teacher_kind="coordinate_labels", trial=0,
        epoch=0, optimizer_step=0, batch_ordinal=0,
        student_states=((0, "P"), (1, "P"), (2, "N")),
    )
    context.require_identity(
        identity, fit_rows=rows, student_inputs=source,
        student_indices=source_indices, teacher_inputs=teacher,
        teacher_indices=targets,
    )
    with pytest.raises(RuntimeError, match="teacher indices"):
        context.require_identity(
            identity, fit_rows=rows, student_inputs=source,
            student_indices=source_indices, teacher_inputs=teacher,
            teacher_indices=tuple(reversed(targets)),
        )
    changed = teacher.clone(); changed[0, 0] += 1
    with pytest.raises(RuntimeError, match="teacher tokens"):
        context.require_identity(
            identity, fit_rows=rows, student_inputs=source,
            student_indices=source_indices, teacher_inputs=changed,
            teacher_indices=targets,
        )
