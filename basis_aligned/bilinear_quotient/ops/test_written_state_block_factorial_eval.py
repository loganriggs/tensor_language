import written_state_block_factorial_eval as evaluator


class Output:
    def __init__(self, values):
        self.answer_foil = values


def test_pair_error_checks_both_endpoint_logits():
    first = Output(((1.0, 2.0), (3.0, 4.0)))
    second = Output(((1.5, 1.0), (3.25, 4.0)))
    assert evaluator.pair_error(first, second) == 1.0
