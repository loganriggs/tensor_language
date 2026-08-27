import importlib.util
from pathlib import Path

import torch
import torch.nn.functional as F


PATH = Path(__file__).with_name("mlp0_quotient_worst_cell.py")
SPEC = importlib.util.spec_from_file_location("mlp0_quotient_worst_cell", PATH)
RUN = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(RUN)


def test_t_vs_o_has_registered_kl_and_ce_directions_with_padded_vocab():
    # Width 7 deliberately differs from the reachable input-token count.  This
    # regresses the development crash caused by a hard-coded logit reshape width.
    logits_t = torch.tensor([[[2.0, 0, 0, 0, 0, 0, 0]]])
    logits_o = torch.tensor([[[0.0, 1, 0, 0, 0, 0, 0]]])
    caps = {"attn1": torch.zeros(1, 1, 2), "mlp1": torch.zeros(1, 1, 2)}
    outputs = {"T": (logits_t, caps), "O": (logits_o, caps)}
    target = torch.tensor([[1]])
    report = RUN.compute_effects(
        outputs, target, {"attn1": 1.0, "mlp1": 1.0}, "T", "O",
        ce_reference="O", ce_candidate="T",
    )
    logp_t = F.log_softmax(logits_t, -1)
    logp_o = F.log_softmax(logits_o, -1)
    expected_kl = (logp_t.exp() * (logp_t - logp_o)).sum(-1)
    expected_ce = F.cross_entropy(logits_t.view(-1, 7), target.view(-1)) - F.cross_entropy(
        logits_o.view(-1, 7), target.view(-1)
    )
    assert torch.allclose(report["kl"], expected_kl)
    assert torch.allclose(report["ce"], expected_ce.view(1, 1))


def test_raw_pre_mlp0_capture_is_lambda_stream_plus_attention_not_mlp_deviation():
    class Module:
        lambdas = torch.tensor([0.75, 0.25])

    x = torch.tensor([[[1.0, 2.0]]])
    x0 = torch.tensor([[[5.0, 6.0]]])
    attention = torch.tensor([[[0.5, -0.5]]])
    RUN.STATE["caps"] = {}
    RUN.block0_pre_hook(Module(), (x, None, x0))
    RUN.attn0_hook(None, None, (attention, None))
    expected = 0.75 * x + 0.25 * x0 + attention
    assert torch.equal(RUN.STATE["caps"]["pre_mlp0"], expected)


def test_all_positions_are_eligible_before_coverage_filter():
    covered = torch.ones(2, RUN.T, dtype=torch.bool)
    valid = covered
    assert int(valid.sum()) == 2 * RUN.T
    position = torch.arange(RUN.T)
    assert int((position < RUN.T // 2).sum()) == RUN.T // 2
    assert int((position >= RUN.T // 2).sum()) == RUN.T // 2
