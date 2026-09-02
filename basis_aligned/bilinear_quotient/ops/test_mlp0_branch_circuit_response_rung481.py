import torch

import mlp0_branch_circuit_response_rung481 as subject


def test_shapley_and_pair_interaction_recover_known_game():
    weights = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    performance = torch.tensor([
        sum(float(weights[index]) for index in range(4) if mask & (1 << index))
        for mask in range(16)
    ], dtype=torch.float64)
    assert torch.allclose(subject._shapley(performance), weights)
    assert torch.allclose(
        subject._pair_interactions(performance), torch.zeros(6, dtype=torch.float64),
        atol=1e-12)

    for mask in range(16):
        if mask & 1 and mask & 4:
            performance[mask] += 5.0
    interactions = subject._pair_interactions(performance)
    assert abs(float(interactions[subject.PAIR_NAMES.index("TxI")]) - 5.0) < 1e-12


def test_batch_crossing_document_split_allocates_rows_exactly():
    masks = {
        tag: {
            mask_type: torch.ones(1000 * subject.TOKENS, dtype=torch.bool)
            for mask_type in subject.MASK_TYPES
        }
        for tag in ("a", "b")
    }
    selections = subject._batch_selections(masks, ["a", "b"], 248, 252, 250)
    for half in (0, 1):
        assert all(int(row[3].sum()) == 2 * subject.TOKENS
                   for row in selections if row[0] == half)


def test_analysis_distinguishes_split_from_shared_profiles():
    circuits = 32
    counts = torch.full((2, 2, circuits), 10.0, dtype=torch.float64)
    difficulty = torch.linspace(-0.2, 0.2, circuits)

    def collection(shared):
        performance = torch.zeros(2, 2, circuits, 16, dtype=torch.float64)
        base = torch.linspace(-1.0, 1.0, circuits)
        other = base if shared else torch.sin(torch.linspace(0, 4 * torch.pi, circuits))
        for half in range(2):
            for circuit in range(circuits):
                effects = [base[circuit], .2 * base[circuit], other[circuit], .05]
                for mask in range(16):
                    performance[half, 0, circuit, mask] = sum(
                        effects[index] for index in range(4) if mask & (1 << index))
                    performance[half, 1, circuit, mask] = 0.0
        means = -performance
        means[:, 0, :, subject.FULL_ARM] += difficulty
        return {
            "ce_sums": means * counts[..., None], "counts": counts,
            "pooled_ce_sums": torch.zeros(2, 16),
            "pooled_counts": torch.ones(2), "instrument": {},
        }

    shared_report = subject.analyze_phase(collection(True))
    assert shared_report["pred_c_shared"] is True
    assert shared_report["pred_c_split"] is False
    split_report = subject.analyze_phase(collection(False))
    assert not (split_report["pred_c_shared"] and split_report["pred_c_split"])
