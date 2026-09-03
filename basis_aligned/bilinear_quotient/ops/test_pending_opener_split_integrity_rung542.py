import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "ops" / "pending_opener_split_integrity_rung542.py"
spec = importlib.util.spec_from_file_location("r542", SCRIPT)
r542 = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(r542)


def test_exact_prompt_key_groups_duplicate_content_not_nominal_ids():
    rows = json.loads(r542.ROWS.read_text())["rows"]
    cell = [row for row in rows if row["split"] == "SELECT"
            and row["family_id"] == "opener_type_substitution"]
    assert len(cell) == 16
    assert len({row["group_id"] for row in cell}) == 16
    assert len({r542.prompt_key(row) for row in cell}) == 8


def test_split_audit_detects_pseudoreplication_but_no_exact_cross_split_leakage():
    rows = json.loads(r542.ROWS.read_text())["rows"]
    report = r542.dataset_report(rows)
    assert report["any_within_split_pseudoreplication"] is True
    assert report["any_exact_pair_cross_split"] is False
    assert report["any_exact_sequence_cross_split"] is False


def test_saved_r538_to_r540_decisions_survive_unique_prompt_rescore():
    result = json.loads(r542.OUT.read_text())
    assert result["pred_c_r538_site_decision_survives_unique_prompt_rescore"] is True
    assert result["pred_d_r539_control_liveness_survives_unique_prompt_rescore"] is True
    assert result["pred_e_r540_null_survives_unique_prompt_rescore"] is True
    assert result["r540_unique_prompt_rescore"]["summary"]["selected_rank"] is None
