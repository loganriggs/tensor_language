import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
POSITIONS = ROOT / "numbered_list_semantic_positions_rung573.json"


def test_all_r573_semantic_positions_are_complete_and_outcome_free():
    data = json.loads(POSITIONS.read_text())
    assert data["row_count"] == 288 and data["fit_rows"] == 192 and data["select_rows"] == 96
    assert data["all_queries_are_final_newlines"] and data["all_labels_are_single_semantic_tokens"]
    assert data["all_base_donor_line_counts_match"]
    assert data["model_loaded"] is False and data["model_forwards"] == 0 and data["outcomes_opened"] == []
    for row in data["mappings"]:
        for endpoint in row["endpoints"].values():
            assert endpoint["final_label_position"] == endpoint["label_positions"][-1]["token_position"]
            assert endpoint["final_query_position"] > endpoint["final_label_position"]
