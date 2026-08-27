import importlib.util
from pathlib import Path

import pytest


PATH = Path(__file__).with_name("source_global_preflight.py")
SPEC = importlib.util.spec_from_file_location("source_global_preflight", PATH)
PREFLIGHT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(PREFLIGHT)


def test_missing_nested_global_is_caught_before_expensive_setup(tmp_path):
    broken = tmp_path / "broken.py"
    broken.write_text("def later(x):\n    return math.sqrt(x)\n")
    assert PREFLIGHT.undefined_global_names(broken) == ["math"]
    with pytest.raises(RuntimeError, match="math"):
        PREFLIGHT.require_defined_globals([broken])


def test_authoritative_oracle_sources_have_no_undefined_globals():
    root = PATH.parents[1]
    files = [
        root / "bilinear_quotient/ship_error_attrib.py",
        PATH.with_name("code_ood_oracle.py"),
        PATH.with_name("prepare_fineweb_oracle_rows.py"),
        PATH.with_name("frozen_ship_oracle_v2.py"),
        PATH.with_name("local_fineweb_harvest.py"),
    ]
    assert {str(path): PREFLIGHT.undefined_global_names(path) for path in files} == {
        str(path): [] for path in files
    }
