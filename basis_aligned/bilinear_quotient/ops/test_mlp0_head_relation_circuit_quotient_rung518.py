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
