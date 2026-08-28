from __future__ import annotations

import ast
from pathlib import Path


BASIS_ALIGNED = Path(__file__).resolve().parents[1]
SCRIPT = BASIS_ALIGNED / "bilinear_quotient" / "ops" / "hybrid_tensor_class_oracle.py"
PREREG = Path(__file__).with_name("HYBRID_TENSOR_CLASS_ORACLE_PREREGISTRATION.md")


def literal_assignments(tree: ast.Module) -> dict[str, object]:
    values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(
            node.targets[0], ast.Name
        ):
            try:
                values[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    return values


def test_frozen_numeric_protocol_and_roles_are_literal() -> None:
    source = SCRIPT.read_text()
    assignments = literal_assignments(ast.parse(source))
    assert assignments["RANK"] == 8
    assert assignments["STEPS"] == 180
    assert assignments["EVERY"] == 30
    assert assignments["BATCH"] == 4
    assert assignments["LR"] == 1e-3
    assert "fineweb_n96_skip80.pt" in source
    assert "fineweb_n192_skip7000.pt" in source
    assert "fineweb_n192_skip11000.pt" in source
    assert "DISCOVERY ONLY" in source


def test_exact_four_arms_and_separate_costs_are_frozen_in_source_and_prereg() -> None:
    source = SCRIPT.read_text()
    prereg = PREREG.read_text()
    for arm in (
        "both_compiled", "attention_native", "mlp_native", "both_native",
    ):
        assert arm in source and f"`{arm}`" in prereg
    assert "factor_reals_M" in source
    assert "active_table_reals_M" in source
    assert "native parameters" in prereg


def test_script_has_no_queue_or_subprocess_side_effect() -> None:
    source = SCRIPT.read_text()
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "subprocess" not in imported
    assert "queue.txt" not in source and "queue2.txt" not in source
