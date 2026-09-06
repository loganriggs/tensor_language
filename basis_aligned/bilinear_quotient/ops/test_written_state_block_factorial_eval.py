import written_state_block_factorial_eval as evaluator
import torch


class Output:
    def __init__(self, values):
        self.answer_foil = values


def test_pair_error_checks_both_endpoint_logits():
    first = Output(((1.0, 2.0), (3.0, 4.0)))
    second = Output(((1.5, 1.0), (3.25, 4.0)))
    assert evaluator.pair_error(first, second) == 1.0


def test_consumed_entry_uses_patched_state_not_prepatch_capture():
    block = type("Block", (), {"lambdas": (torch.tensor(2.0), torch.tensor(3.0))})()
    backend = type("Backend", (), {"model": type("Model", (), {"transformer": type("Transformer", (), {"h": [block]})()})()})()
    base_batch = type("Batch", (), {"semantic_positions": (1,)})()
    donor_batch = type("Batch", (), {"semantic_positions": (1,)})()
    base_states = (torch.tensor([[[1.0], [1.0]]]),)
    writer_states = (torch.tensor([[[4.0], [5.0]]]),)
    actual = evaluator.consumed_entry(
        backend, base_batch, donor_batch, base_states, writer_states,
        block_index=0, input_group="all_positions",
    )
    assert torch.equal(actual, torch.tensor([[[11.0], [13.0]]]))
