import importlib.util
from pathlib import Path
import sys

import torch


OPS = Path(__file__).resolve().parent
for path in (OPS, OPS.parent, OPS.parent.parent / "polynomial_causal"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
spec = importlib.util.spec_from_file_location(
    "rung512", OPS / "mlp10_branch_first_consumer_quotient_rung512.py")
rung = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(rung)


def test_fixed_consumers_relations_and_price():
    assert rung.PRIMARY_SITES == ("a11", "m11", "q11")
    assert rung.N_NODES == 28
    assert 7 * 6 == 42
    assert 42 * len(rung.PRIMARY_SITES) == 126
    assert 4216 + 2046 + 124 * 126 == 21886


def test_question_scalar_is_sign_gauge_invariant():
    state = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    vectors = torch.eye(2)
    basis = {"vectors": vectors, "values": torch.tensor([2.0, -1.0])}
    original = rung.question_scalar(state, basis)
    flipped = rung.question_scalar(
        state, {"vectors": vectors * torch.tensor([-1.0, 1.0]),
                "values": basis["values"]})
    torch.testing.assert_close(original, flipped)
    torch.testing.assert_close(original, torch.tensor([[[-2.0, 2.0]]]).squeeze(0))


def test_planted_relation_is_consumer_convergence_not_source_similarity():
    collection = rung._toy_collection()
    candidates, summary = rung.discover_relations(collection)
    planted = [candidate for candidate in candidates
               if candidate["left_node"] == 0
               and candidate["right_node"] == rung.N_SUBSETS
               and candidate["site"] == "a11"]
    assert summary["consumer_tests"] == 126
    assert len(planted) == 1
    assert planted[0]["type"] == "consumer_convergence"
    assert planted[0]["beta_left_from_right"] == -0.5


def test_relations_never_cross_branch_subsets():
    candidates, _ = rung.discover_relations(rung._toy_collection())
    assert all(rung.r511.node_parts(candidate["left_node"])[1]
               == rung.r511.node_parts(candidate["right_node"])[1]
               for candidate in candidates)


def test_confirmation_never_refits_discovery_beta():
    collection = rung._toy_collection()
    candidates, _ = rung.discover_relations(collection)
    candidate = next(candidate for candidate in candidates
                     if candidate["site"] == "a11"
                     and candidate["left_node"] == 0
                     and candidate["right_node"] == rung.N_SUBSETS)
    beta = candidate["beta_left_from_right"]
    confirmed, checks = rung.confirm_relations(collection, [candidate])
    assert len(confirmed) == 1
    assert confirmed[0]["beta_left_from_right"] == beta
    key = f"{candidate['left_name']} <-> {candidate['right_name']} @ a11"
    assert checks[key]["consumer"]["beta_left_from_right"] == beta


def test_consumer_replacement_algebra_for_writes_and_question_direction():
    current = {
        "a11": torch.tensor([[[3.0, 4.0]]]),
        "m11": torch.tensor([[[5.0, 6.0]]]),
    }
    basis = {"direction": torch.tensor([1.0, 0.0])}
    key, write = rung.replacement_write(
        "a11", current, torch.tensor([[[1.0, 2.0]]]),
        torch.tensor([[[2.0, 1.0]]]), .5, basis)
    assert key == "a11"
    torch.testing.assert_close(write, torch.tensor([[[3.0, 2.5]]]))
    key, write = rung.replacement_write(
        "q11", current, torch.tensor([[2.0]]), torch.tensor([[6.0]]), .5, basis)
    assert key == "m11"
    torch.testing.assert_close(write, torch.tensor([[[6.0, 6.0]]]))


def test_all_frozen_hashes_and_parent_route_are_pinned():
    for path, expected in rung.HASHES.items():
        assert rung.sha256(path) == expected
    result = __import__("json").loads(rung.R511_RESULT.read_text())
    assert result["analysis"]["discovery_summary"]["candidate_count"] == 0
    assert result["next_step"] == (
        "localize_exact_branches_at_first_downstream_consumer_including_mlp11_question_interface")


def test_dry_run_opens_no_model_outcome(capsys):
    rung.dry_run()
    output = capsys.readouterr().out
    assert '"model_loaded": false' in output
    assert '"outcomes_opened": false' in output
    assert '"planted_consumer_convergence_recovered": true' in output
