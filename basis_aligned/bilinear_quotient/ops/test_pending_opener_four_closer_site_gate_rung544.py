import ast
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "ops" / "pending_opener_four_closer_site_gate_rung544.py"
spec = importlib.util.spec_from_file_location("r544", SCRIPT)
r544 = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(r544)


def test_price_and_scope_are_exact():
    rows = r544.load_rows()
    assert len(rows) == 720
    assert r544.EXPECTED_FORWARDS == 450
    assert set(row["split"] for row in rows) == {"FIT", "SELECT"}


def test_every_ordered_closer_pair_has_fit_and_select_support():
    rows = r544.load_rows()
    for split, expected in {"FIT": 8, "SELECT": 4}.items():
        for family in r544.TARGETS:
            cells = {}
            for row in rows:
                if row["split"] == split and row["family_id"] == family:
                    pair = (row["base_answer_id"], row["donor_answer_id"])
                    cells[pair] = cells.get(pair, 0) + 1
            assert len(cells) == 12
            assert set(cells.values()) == {expected}


def test_static_result_contract_preserves_unopened_splits():
    tree = ast.parse(SCRIPT.read_text())
    text = ast.unparse(tree)
    assert '"forbidden_splits_opened": []' in text or "'forbidden_splits_opened': []" in text
    assert "model_weights_updated" in text
