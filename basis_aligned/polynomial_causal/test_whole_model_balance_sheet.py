import importlib.util
from pathlib import Path


PATH = Path(__file__).with_name("whole_model_balance_sheet.py")
SPEC = importlib.util.spec_from_file_location("whole_model_balance_sheet", PATH)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MOD)


def test_balance_sheet_keeps_distinct_currencies():
    sheet = MOD.build_balance_sheet(MOD.DEFAULT_SOURCES, Path("/workspace/theseus-bench"))
    ledgers = sheet["ledgers"]
    assert sheet["metrics_not_averagable"] is True
    assert ledgers["analytic_interface_substitutability"]["value"] > 0.99
    assert 0.25 < ledgers["named_variable_understanding"]["value"] < 0.5
    assert 0.05 < ledgers["causal_path_coverage"]["value"] < 0.2
    assert ledgers["legacy_composition_stress_test"]["value"] < 0.2
    assert len({row["currency"] for row in ledgers.values()}) == len(ledgers)


def test_balance_sheet_closes_primary_denominators():
    sheet = MOD.build_balance_sheet(MOD.DEFAULT_SOURCES, None)
    causal = sheet["ledgers"]["causal_path_coverage"]
    composed = sheet["ledgers"]["legacy_composition_stress_test"]
    assert abs(causal["value"] + causal["unnamed_fraction"] - 1.0) < 1e-6
    assert abs(composed["value"] + composed["residual_fraction"] - 1.0) < 1e-6
    assert sheet["registry_inventory"]["available"] is False


def test_current_composite_uses_frozen_registry():
    sheet = MOD.build_balance_sheet(MOD.DEFAULT_SOURCES, Path("/workspace/theseus-bench"))
    ship = sheet["registry_inventory"]["current_composite"]
    assert ship["top_level_targets_replaced"] == "36/36"
    assert abs(ship["delta_ce"] - (ship["composite_ce"] - ship["clean_ce"])) < 1e-6
    assert sheet["current_bottleneck"]["observed_global_delta_ce"] == ship["delta_ce"]


def test_output_basis_is_locator_not_controller():
    sheet = MOD.build_balance_sheet(MOD.DEFAULT_SOURCES, None)
    instrument = sheet["ledgers"]["causal_instrument_validation"]
    assert instrument["output_basis_recall"] > 2 * instrument["output_basis_random_recall"]
    assert instrument["output_basis_damage_fraction_of_oracle"] < 0.5
