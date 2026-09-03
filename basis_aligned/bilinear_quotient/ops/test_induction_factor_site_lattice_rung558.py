import importlib.util
import os
import subprocess
from pathlib import Path


SCRIPT = Path(__file__).with_name("induction_factor_site_lattice_rung558.py")
TEXT = SCRIPT.read_text()


def load_module():
    os.environ["BQLIB_NO_MODEL"] = "1"
    spec = importlib.util.spec_from_file_location("rung558", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frozen_factor_lattice_and_no_rank_search():
    assert "FIT_MASKS = tuple(range(1, 16))" in TEXT
    assert '"L5H5", 5, 5' not in TEXT  # term identity comes from the frozen factor implementation
    assert "rank" not in TEXT.lower().replace("rung", "")
    assert "FINAL_TEST" not in TEXT and '"OOD"' not in TEXT


def test_mobius_inversion_reconstructs_values():
    module = load_module()
    values = {mask: float(mask * mask - 3 * mask) for mask in range(16)}
    coefficients = {int(key, 2): value for key, value in module.mobius(values).items()}
    for mask, expected in values.items():
        reconstructed = sum(value for subset, value in coefficients.items() if (subset & mask) == subset)
        assert abs(reconstructed - expected) < 1e-9


def test_dryrun_plants_unique_minimal_subset_without_model():
    env = dict(os.environ, BQLIB_DRYRUN="1", BQLIB_NO_MODEL="1")
    result = subprocess.run(["python", str(SCRIPT)], env=env, check=True, capture_output=True, text=True)
    assert '"status": "dryrun_passed"' in result.stdout
    assert '"selected_mask": "0011"' in result.stdout
    assert '"model_loaded": false' in result.stdout
