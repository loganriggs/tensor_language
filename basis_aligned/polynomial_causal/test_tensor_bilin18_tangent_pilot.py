from __future__ import annotations

from pathlib import Path

import tensor_bilin18_tangent_pilot as pilot


def test_pilot_binds_admitted_program_plan_parents_and_create_only_result() -> None:
    source = Path(pilot.__file__).read_text()
    assert pilot.RANK == 640
    assert pilot.CUTS == (1, 2, 3)
    assert pilot.EXPECTED_PLAN_FINGERPRINT in source
    for fragment in (
        "rank640_parent", "causal_parent", "516_707_766", "os.O_EXCL",
        "collect_write_geometry_bank", "TensorBilin18TangentTransaction",
        "target hash ledger", "compare_split_cuts", "consequence_stage_authorized",
    ):
        assert fragment in source


def test_stage1_requires_every_cut_not_a_favorable_subset() -> None:
    passing = {str(cut): {"passes": True} for cut in pilot.CUTS}
    assert pilot.stage1_passes(passing)
    passing["2"]["passes"] = False
    assert not pilot.stage1_passes(passing)
    assert not pilot.stage1_passes({"1": {"passes": True}})


def test_source_manifest_covers_every_new_scientific_boundary() -> None:
    names = {path.name for path in pilot.SOURCES}
    assert {
        "FINITE_HORIZON_TANGENT_REALIZATION_PREREGISTRATION.md",
        "finite_horizon_tangent_plan.json",
        "tensor_bilin18_tangent_collector.py",
        "finite_horizon_tangent_response_bank.py",
        "finite_horizon_tangent_realization.py",
        "tensor_bilin18_program.py",
        "test_tensor_bilin18_tangent_pilot.py",
        "test_tensor_bilin18_tangent_collector.py",
    } <= names
