import importlib.util
from pathlib import Path
import sys

import torch


OPS = Path(__file__).resolve().parent
for path in (OPS, OPS.parent, OPS.parent.parent / "polynomial_causal"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
spec = importlib.util.spec_from_file_location(
    "rung511", OPS / "mlp10_score_change_three_branch_factorial_rung511.py")
rung = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(rung)


class TinyMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.Left = torch.nn.Linear(5, 7, bias=False)
        self.Right = torch.nn.Linear(5, 7, bias=False)
        self.Down = torch.nn.Linear(7, 5, bias=False)
        self.Down_bias = torch.nn.Parameter(torch.randn(5))

    def forward(self, value):
        return self.Down(self.Left(value) * self.Right(value)) + self.Down_bias


def test_fixed_branch_subset_vocabulary_and_price():
    assert rung.BRANCH_NAMES == ("L", "R", "LR")
    assert rung.SUBSET_MASKS == (1, 2, 4, 3, 5, 6, 7)
    assert rung.SUBSET_NAMES == ("L", "R", "LR", "L+R", "L+LR", "R+LR", "L+R+LR")
    assert rung.N_ACTIONS == 4
    assert rung.N_NODES == 28
    assert 7 * 6 == 42
    assert 2 * 2108 + 372 + 124 * 42 == 9796


def test_deployed_branches_close_exactly_and_match_float32_identity():
    torch.manual_seed(511)
    mlp = TinyMLP().float()
    z0 = torch.randn(2, 3, 5)
    za = torch.randn(2, 3, 5)
    absent = {"z": z0, "deployed_write": mlp(z0).detach()}
    current = {"z": za, "deployed_write": mlp(za).detach()}
    branches, diagnostics = rung.deployed_branches(mlp, absent, current)
    total = current["deployed_write"] - absent["deployed_write"]
    torch.testing.assert_close(sum(branches), total, atol=1e-7, rtol=1e-6)
    assert diagnostics["absent_corner_replay_max_abs"] == 0
    assert diagnostics["current_corner_replay_max_abs"] == 0
    assert diagnostics["deployed_branch_sum_relative_squared"] <= 1e-12
    assert max(diagnostics["float32_ideal_branch_relative_squared"]) < 1e-10


def test_each_subset_is_literal_sum_of_predeclared_branches():
    values = (torch.tensor([1.0]), torch.tensor([10.0]), torch.tensor([100.0]))
    expected = (1, 10, 100, 11, 101, 110, 111)
    for index, target in enumerate(expected):
        assert rung.subset_output(values, index).item() == target


def test_discovery_tests_only_same_subset_cross_action_relations():
    generator = torch.Generator(device="cpu").manual_seed(1)
    circuit = .002 * torch.randn(rung.N_NODES, 32, generator=generator, dtype=torch.float64)
    task = .002 * torch.randn(rung.N_NODES, 4, generator=generator, dtype=torch.float64)
    circuit[rung.N_SUBSETS] = -2 * circuit[0]
    task[rung.N_SUBSETS] = -2 * task[0]
    matrices = {window: {"circuit": circuit.clone(), "task": task.clone()}
                for window in ("half0", "half1", "pooled")}
    candidates, summary = rung.discover_relations(matrices)
    assert summary["relations_tested"] == 42
    planted = [row for row in candidates
               if row["left_node"] == 0 and row["right_node"] == rung.N_SUBSETS]
    assert len(planted) == 1
    assert planted[0]["subset_name"] == "L"
    assert planted[0]["beta_left_from_right"] == -0.5
    assert all(rung.node_parts(row["left_node"])[1]
               == rung.node_parts(row["right_node"])[1] for row in candidates)


def test_preregistration_hash_and_rung510_zero_pair_route_are_pinned():
    assert rung.sha256(rung.PREREG) == rung.HASHES[rung.PREREG]
    assert rung.sha256(rung.R510_RESULT) == rung.HASHES[rung.R510_RESULT]
    assert rung.sha256(rung.R510_BUNDLE) == rung.HASHES[rung.R510_BUNDLE]
    assert rung.sha256(rung.R510_SOURCE) == rung.HASHES[rung.R510_SOURCE]
    result = __import__("json").loads(rung.R510_RESULT.read_text())
    assert result["analysis"]["discovery_summary"]["candidate_count"] == 0
    assert result["next_step"] == "registered_multi_term_signed_combinations_without_pair_ranking"


def test_dry_run_opens_no_model_outcome(capsys):
    rung.dry_run()
    output = capsys.readouterr().out
    assert '"model_loaded": false' in output
    assert '"outcomes_opened": false' in output
    assert '"relations_tested": 42' in output
