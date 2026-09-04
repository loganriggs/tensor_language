#!/usr/bin/env python3
"""Focused CPU tests for the battery-to-framework integration boundary."""

from dataclasses import replace

import pytest

import circuit_battery_integration_contract as battery
import circuit_experiment_spec as framework
import circuit_managed_entry as managed


ZERO_SHA = "0" * 64


def task() -> battery.BatteryTaskSpec:
    return battery.BatteryTaskSpec(
        task_id="synthetic.successor", generator_role="generate_successor",
        answer_role="score_next_token",
        transforms=(
            battery.TransformSpec("A1", "change_start", True, "toward_donor"),
            battery.TransformSpec("A2", "change_step", True, "toward_donor"),
            battery.TransformSpec("P", "change_surface", False, "invariant"),
            battery.TransformSpec("C", "copy_control", True, "registered_active"),
        ),
        joint_arms=(battery.JointArmSpec("reader8_reader10", ("reader8", "reader10")),),
    )


def rows():
    output = []
    for split in battery.PHASES:
        for transform in task().transforms:
            output.append({
                "task_id": task().task_id, "split": split,
                "group_id": f"{split}:group0", "row_id": f"{split}:{transform.transform_id}",
                "transform_id": transform.transform_id,
                "answer_changes": transform.answer_changes, "ids": [1, 2],
            })
    return output


def compiled_fit():
    fit_rows = [row for row in rows() if row["split"] == "FIT"]
    arm_names = ("native", "reader8", "reader10", "reader8_reader10")
    arm_specs = (framework.ArmSpec("native", "native", "undirected"),) + tuple(
        framework.ArmSpec(name, "counterfactual", "undirected") for name in arm_names[1:]
    )
    family = framework.CallFamilySpec(
        name="fit_reader_arms", split="FIT", arms=arm_names, batch_size=4,
        call_kind="suffix", guard="fit_only", call_id_template="{split}:{arm}:{batch}",
        arm_specs=arm_specs,
    )
    table = framework.AuthorityTableSpec(
        "rows", ("row_id",), framework.canonical_sha256(fit_rows),
        group_fields=("group_id",), expected_counts={"FIT": 4}, expected_total=4,
    )
    spec = framework.CircuitExperimentSpec(
        experiment_id="battery-fit", rung=1, artifacts=(),
        phases=(framework.PhaseSpec("FIT"),), authority_tables=(table,), calls=(family,),
    )
    return fit_rows, framework.compile_experiment(
        spec, authority_tables={"rows": fit_rows}, call_source_records=fit_rows,
    )


def receipt(phase, selection=ZERO_SHA, decision="pass"):
    price = battery.ExactPhasePrice(phase, 1, 4, 0, 0, 100)
    return battery.PhaseReceipt(task().task_id, phase, decision, selection, "1" * 64, price)


def test_task_rows_require_split_disjoint_complete_group_panels() -> None:
    assert battery.validate_rows(task(), rows()) == framework.canonical_sha256(rows())
    missing = [row for row in rows() if row["row_id"] != "SELECT:A2"]
    with pytest.raises(battery.BatteryContractError, match="exactly one"):
        battery.validate_rows(task(), missing)
    crossed = rows()
    crossed[-1] = dict(crossed[-1], group_id="FIT:group0")
    with pytest.raises(battery.BatteryContractError, match="crosses split"):
        battery.validate_rows(task(), crossed)
    reused_generator = replace(
        task(), transforms=task().transforms[:1] +
        (replace(task().transforms[1], generator_role="change_start"),) + task().transforms[2:]
    )
    with pytest.raises(battery.BatteryContractError, match="independent generator"):
        battery.validate_task(reused_generator)


def test_phase_opening_requires_exact_passing_prefix_and_frozen_selection() -> None:
    battery.authorize_phase(task(), "FIT", ())
    fit = receipt("FIT")
    battery.authorize_phase(task(), "SELECT", (fit,))
    select = receipt("SELECT")
    battery.authorize_phase(task(), "TEST", (fit, select))
    test = receipt("TEST")
    battery.authorize_phase(task(), "OOD", (fit, select, test))
    with pytest.raises(battery.BatteryContractError, match="exact required prefix"):
        battery.authorize_phase(task(), "TEST", (fit,))
    with pytest.raises(battery.BatteryContractError, match="fail"):
        battery.authorize_phase(task(), "SELECT", (replace(fit, decision="fail"),))
    with pytest.raises(battery.BatteryContractError, match="selection changed"):
        battery.authorize_phase(task(), "TEST", (fit, replace(select, selection_sha256="2" * 64)))
    with pytest.raises(battery.BatteryContractError, match="exact required prefix"):
        battery.authorize_phase(task(), "OOD", (fit, select))
    with pytest.raises(battery.BatteryContractError, match="price is for another phase"):
        battery.authorize_phase(
            task(), "SELECT", (replace(fit, price=replace(fit.price, phase="SELECT")),)
        )


def test_phase_artifacts_exclude_future_split_and_use_framework_types() -> None:
    source = battery.PhasedArtifact(
        framework.ArtifactRef("producer", "producer.py", ZERO_SHA, "source", True, True),
        "PROTOCOL",
    )
    fit_rows = battery.PhasedArtifact(
        framework.ArtifactRef("fit_rows", "fit.json", ZERO_SHA, "authority"), "FIT",
    )
    test_rows = battery.PhasedArtifact(
        framework.ArtifactRef("test_rows", "test.json", ZERO_SHA, "authority"), "TEST",
    )
    references = battery.phase_artifacts("SELECT", (source, fit_rows))
    assert references == (source.artifact, fit_rows.artifact)
    managed.validate_dryrun_closure(framework.CircuitExperimentSpec(
        "battery-select", 1, references, (), (), (),
    ))
    with pytest.raises(battery.BatteryContractError, match="future split"):
        battery.phase_artifacts("SELECT", (source, test_rows))
    disguised = battery.PhasedArtifact(
        framework.ArtifactRef("old_result", "result.json", ZERO_SHA, "outcome"), "PROTOCOL"
    )
    with pytest.raises(battery.BatteryContractError, match="relabeled"):
        battery.phase_artifacts("FIT", (source, disguised))


def test_literal_price_is_bound_to_compiled_calls_and_receipt() -> None:
    _, compiled = compiled_fit()
    price = battery.ExactPhasePrice("FIT", 4, 16, 2, 0, 4096)
    battery.validate_price(compiled, price)
    battery.validate_price_receipt(price, price)
    with pytest.raises(battery.BatteryContractError, match="forward-call"):
        battery.validate_price(compiled, replace(price, forward_calls=3))
    with pytest.raises(battery.BatteryContractError, match="example-evaluation"):
        battery.validate_price(compiled, replace(price, example_evaluations=15))
    with pytest.raises(battery.BatteryContractError, match="measured phase price"):
        battery.validate_price_receipt(price, replace(price, evidence_bytes=4095))


def joint_evidence(fit_rows):
    return [{
        "joint_arm_id": "reader8_reader10", "row_id": row["row_id"],
        "group_id": row["group_id"], "joint_call_id": "FIT:reader8_reader10:0",
        "member_call_ids": {"reader8": "FIT:reader8:0", "reader10": "FIT:reader10:0"},
        "singleton_effects": {"reader8": 0.2, "reader10": 0.3},
        "joint_effect": 0.7, "interaction": 0.2,
    } for row in fit_rows]


def test_joint_arm_requires_physical_call_exact_coverage_and_interaction() -> None:
    fit_rows, compiled = compiled_fit()
    evidence = joint_evidence(fit_rows)
    battery.validate_joint_arm_evidence(task(), fit_rows, compiled, evidence)
    invented = [dict(record) for record in evidence]
    invented[0] = dict(invented[0], joint_call_id="FIT:reader8:0")
    with pytest.raises(battery.BatteryContractError, match="physical model call"):
        battery.validate_joint_arm_evidence(task(), fit_rows, compiled, invented)
    wrong_interaction = [dict(record) for record in evidence]
    wrong_interaction[0] = dict(wrong_interaction[0], interaction=0.0)
    with pytest.raises(battery.BatteryContractError, match="interaction"):
        battery.validate_joint_arm_evidence(task(), fit_rows, compiled, wrong_interaction)
    with pytest.raises(battery.BatteryContractError, match="incomplete"):
        battery.validate_joint_arm_evidence(task(), fit_rows, compiled, evidence[:-1])
    duplicate = dict(compiled)
    duplicate_call = dict(next(
        call for call in compiled["call_manifest"] if call["arm"] == "reader8_reader10"
    ))
    duplicate_call["call_id"] = "FIT:reader8_reader10:duplicate"
    duplicate["call_manifest"] = [*compiled["call_manifest"], duplicate_call]
    with pytest.raises(battery.BatteryContractError, match="more than one physical"):
        battery.validate_joint_arm_evidence(task(), fit_rows, duplicate, evidence)
