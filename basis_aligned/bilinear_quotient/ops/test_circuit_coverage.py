import importlib.util
import json
from pathlib import Path


BQ = Path(__file__).resolve().parents[1]
SCRIPT = BQ / "make_circuit_coverage.py"
spec = importlib.util.spec_from_file_location("coverage", SCRIPT)
coverage = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(coverage)


def test_generated_coverage_contains_only_canonical_task_records():
    coverage.main()
    result = json.loads(coverage.OUT_JSON.read_text())
    assert set(result["records"]) == {
        "task.bracket.pending_opener", "task.increment.state", "task.induction.selector_payload",
        "task.successor.pointer", "subroutine.induction.equality_score",
    }
    assert len(result["categories"]) == 8


def test_known_positive_and_negative_evidence_remain_visible():
    result = json.loads(coverage.OUT_JSON.read_text())["records"]
    bracket = result["task.bracket.pending_opener"]["categories"]
    assert bracket["capability"]["status"] == "mixed"
    assert bracket["causal_site"]["status"] in {"held", "mixed"}
    assert bracket["cross_family"]["status"] == "blocked"
    equality = result["subroutine.induction.equality_score"]["categories"]
    assert equality["composition"]["status"] == "mixed"
    assert equality["ood"]["status"] == "blocked"


def test_no_rank_or_reconstruction_category_can_masquerade_as_circuit_evidence():
    assert "rank" not in coverage.CATEGORIES
    assert "compression" not in coverage.CATEGORIES
    assert "reconstruction" not in coverage.CATEGORIES
