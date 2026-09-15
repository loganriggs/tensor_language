import importlib.util
from pathlib import Path

import numpy as np


RUNNER = Path(__file__).with_name("run_numeric_downstream_oriented_bilinear_v1.py")


def load():
    spec = importlib.util.spec_from_file_location("oriented_v1", RUNNER)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_bound_dry_plan():
    module = load(); value = module.plan()
    assert value["conditionally_opened_split"] == "SELECT"
    assert value["forbidden_splits"] == ["FINAL_TEST", "OOD"]
    assert value["price"]["fits"] == 0


def test_oriented_scalar_identity():
    rng = np.random.default_rng(44)
    left, right, down = (rng.normal(size=(5, 3)), rng.normal(size=(5, 3)), rng.normal(size=(3, 5)))
    x0, d = rng.normal(size=3), rng.normal(size=3)
    l0, r0, ld, rd = left @ x0, right @ x0, left @ d, right @ d
    split = down @ (ld * r0) + down @ (l0 * rd) + down @ (ld * rd)
    direct = down @ ((left @ (x0 + d)) * (right @ (x0 + d)) - l0 * r0)
    np.testing.assert_allclose(split, direct, rtol=1e-12, atol=1e-12)
