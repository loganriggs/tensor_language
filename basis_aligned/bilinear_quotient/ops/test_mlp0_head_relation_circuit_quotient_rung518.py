import importlib.util
from pathlib import Path

import torch


PATH = Path(__file__).with_name("mlp0_head_relation_circuit_quotient_rung518.py")
SPEC = importlib.util.spec_from_file_location("r518", PATH)
R = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R)


def test_atom_vocabulary_round_trip():
    assert len(R.ATOM_NAMES) == 45
    assert len(set(R.ATOM_NAMES)) == 45
    for head in range(9):
        for group in range(5):
            assert R.atom_parts(R.atom_index(head, group)) == (head, group)


def test_head_relation_atoms_close_and_contexts_use_one_piece():
    class Projection:
        weight = torch.eye(R.D)

    class Attention:
        c_proj = Projection()

    class Block:
        attn = Attention()

    batch, length = 1, 3
    generator = torch.Generator().manual_seed(518)
    pattern = torch.randn(batch, R.N_HEADS, length, length, generator=generator)
    causal = torch.tril(torch.ones(length, length, dtype=torch.bool))
    masks = torch.zeros(len(R.GROUPS), batch, length, length, dtype=torch.bool)
    masks[0] = torch.eye(length, dtype=torch.bool)
    masks[1, :, 1:, :-1] = torch.eye(length - 1, dtype=torch.bool)
    masks[-1] = causal & ~masks.sum(0).bool()
    value = torch.randn(batch, length, R.N_HEADS, R.HEAD_DIM, generator=generator)
    joined = torch.einsum("bhqk,bkhd->bqhd", pattern * causal, value).reshape(
        batch, length, R.D)
    split = {"pattern": pattern, "value": value, "partition_masks": masks,
             "native_write": joined + 0.125}
    decomposition = R.head_relation_atoms(Block(), split)
    assert decomposition["atoms"].shape == (45, batch, length, R.D)
    assert decomposition["relative_squared_closure"] <= 1e-12
    assert torch.equal(
        R.atom_context(split["native_write"], decomposition, 0, "SINGLE"),
        decomposition["remainder"] + decomposition["atoms"][0])
    assert torch.equal(
        R.atom_context(split["native_write"], decomposition, 0, "DROP"),
        split["native_write"] - decomposition["atoms"][0])


def test_response_matrices_keep_singleton_and_removal_backgrounds_separate():
    documents, task_cells, tags = 4, 5, 3
    task_sums = torch.zeros(len(R.ARMS), documents, task_cells, dtype=torch.float64)
    task_counts = torch.ones(documents, task_cells, dtype=torch.float64)
    task_sums[0] = 10
    task_sums[1] = 2
    task_sums[2] = 6
    task_sums[2 + R.N_ATOMS] = 5
    circuit_sums = torch.zeros(len(R.ARMS), 2, 2, tags, dtype=torch.float64)
    circuit_counts = torch.ones(2, 2, tags, dtype=torch.float64)
    circuit_sums[0, :, 0] = 10
    circuit_sums[0, :, 1] = 2
    circuit_sums[1, :, 0] = 2
    circuit_sums[2, :, 0] = 7
    circuit_sums[2, :, 1] = 1
    circuit_sums[2 + R.N_ATOMS, :, 0] = 5
    circuit_sums[2 + R.N_ATOMS, :, 1] = 1
    collection = {
        "bounds": (0, 4, 2), "arms": R.ARMS,
        "task_sums": task_sums, "task_counts": task_counts,
        "circuit_sums": circuit_sums, "circuit_counts": circuit_counts,
    }
    matrices = R.response_matrices(collection, (0, 1, 2, 3))
    for half in ("half0", "half1"):
        assert torch.equal(matrices[half]["task"][0, 0], torch.full((4,), 4.0))
        assert torch.equal(matrices[half]["task"][0, 1], torch.full((4,), 3.0))
        assert torch.equal(matrices[half]["circuit"][0, 0], torch.full((3,), 2.0))
        assert torch.equal(matrices[half]["circuit"][0, 1], torch.full((3,), 2.0))


def test_physical_scoring_requires_recovery_direction_and_off_target_preservation():
    documents, task_cells, tags = 4, 5, 3
    confirmation = {
        "bounds": (0, 4, 2), "arms": R.ARMS,
        "task_sums": torch.zeros(len(R.ARMS), documents, task_cells, dtype=torch.float64),
        "task_counts": torch.ones(documents, task_cells, dtype=torch.float64),
        "circuit_sums": torch.zeros(len(R.ARMS), 2, 2, tags, dtype=torch.float64),
        "circuit_counts": torch.ones(2, 2, tags, dtype=torch.float64),
    }
    confirmation["task_sums"][1] = 2
    for target in (0, 1):
        drop = 2 + R.N_ATOMS + target
        confirmation["task_sums"][drop] = 4
        confirmation["circuit_sums"][drop, :, 0] = 2
    substitutions = {
        "bounds": (0, 4, 2),
        "directions": [
            {"pair": 0, "target": 0, "donor": 1, "scale": 1.0, "name": "a<-b"},
            {"pair": 0, "target": 1, "donor": 0, "scale": 1.0, "name": "b<-a"},
        ],
        "task_sums": torch.full((2, documents, task_cells), 2.4, dtype=torch.float64),
        "task_counts": torch.ones(documents, task_cells, dtype=torch.float64),
        "circuit_sums": torch.zeros(2, 2, 2, tags, dtype=torch.float64),
        "circuit_counts": torch.ones(2, 2, tags, dtype=torch.float64),
    }
    substitutions["task_sums"][:, :, 4] = 2.001
    substitutions["circuit_sums"][:, :, 0] = .4
    passing, checks = R.score_substitutions(substitutions, confirmation, (0, 1, 2, 3), 4)
    assert len(passing) == 1
    assert all(row["holds"] for row in checks.values())


def test_native_boundary_evidence_merges_heads_and_splits_one_head():
    discovery = {
        half: {
            "circuit": torch.zeros(R.N_ATOMS, 2, 32, dtype=torch.float64),
            "task": torch.full((R.N_ATOMS, 2, 4), .001, dtype=torch.float64),
        }
        for half in ("half0", "half1")
    }
    for half in discovery.values():
        half["circuit"][0, :, 0] = .01
        half["circuit"][5, :, 0] = .01
        half["circuit"][1, :, 1] = .01
    candidates = [{"left": 0, "right": 5, "left_name": R.ATOM_NAMES[0],
                   "right_name": R.ATOM_NAMES[5]}]
    evidence = R.native_boundary_evidence(
        discovery, candidates, [{"pair": 0, "directions": ["x", "y"]}])
    assert evidence
    assert evidence[0]["crosses_heads"]
    assert any(row["same_head_split_atom"] == R.ATOM_NAMES[1]
               for row in evidence[0]["split_witnesses"])


def test_exact_proportional_pair_passes_both_backgrounds():
    responses, _expected = R.planted_problem(51800)
    left, right = R.PLANTED_PAIRS[0]
    metrics = R.pair_metrics(responses, left, right)
    assert metrics["holds"]
    assert abs(metrics["beta_left_from_right"] - 0.5) < 1e-12
    for half in metrics["halves"].values():
        for background in half.values():
            for kind in background.values():
                assert abs(kind["signed_cosine"] - 1) < 1e-12
                assert kind["left_from_right_relative_residual"] < 1e-12
                assert kind["right_from_left_relative_residual"] < 1e-12


def test_all_eight_planted_relations_recover_without_false_pairs():
    result = R.planted_suite()
    assert len(result["cases"]) == 8
    assert result["all_eight_exact"]


def test_circuit_permutations_destroy_planted_relations_and_confirmation_keeps_them():
    responses, _expected = R.planted_problem(51800)
    candidates = R.discover_pairs(responses)
    assert R.permutation_control_counts(responses) == [0] * 16
    confirmed, checks = R.confirmation_pairs(responses, candidates)
    assert len(confirmed) == 4
    assert all(row["holds"] for row in checks.values())


def test_random_unrelated_pair_fails():
    responses, _expected = R.planted_problem(51801)
    metrics = R.pair_metrics(responses, 1, 2)
    assert not metrics["holds"]


def test_dry_run_opens_no_model_or_outcome():
    result = R.dry_run()
    assert result["model_loaded"] is False
    assert result["model_outcomes_opened"] is False
    assert result["atoms"] == 45
    assert result["unordered_pairs"] == 990
