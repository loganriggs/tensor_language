import pytest
import torch

from whole_model_heldout_stats import (
    arm_gains,
    ceiling,
    document_cluster_bootstrap,
    gain_structure_holds,
    pooled_ce,
)


def test_ce_and_ceiling_known_answer():
    assert pooled_ce([2.0, 6.0], [1, 3]) == pytest.approx(2.0)
    assert ceiling(torch.tensor(8.0), torch.tensor(3.0), torch.tensor(5.0)) == pytest.approx(0.6)
    with pytest.raises(ValueError):
        ceiling(torch.tensor(3.0), torch.tensor(3.0), torch.tensor(2.0))


def test_gain_gate_rejects_vacuous_or_harmful_additivity():
    good = {"attn": 0.027, "mlp": 0.013, "both": 0.041}
    assert gain_structure_holds(good)
    assert not gain_structure_holds({"attn": -0.01, "mlp": -0.02, "both": -0.03})
    assert not gain_structure_holds({"attn": 0.001, "mlp": 0.0005, "both": 0.0015})
    assert not gain_structure_holds({"attn": 0.030, "mlp": 0.001, "both": 0.021})


def test_document_bootstrap_keeps_rows_from_a_document_together():
    # Every row has the same within-arm CE, so every resample has the known answer.
    counts = [2, 3, 4]
    ces = {"live": 3.0, "constant": 8.0, "simple": 5.5,
           "attn_upgraded": 5.3, "mlp_upgraded": 5.4, "both": 5.2}
    records = {
        name: {"loss_sums": [ce * n for n in counts], "counts": counts}
        for name, ce in ces.items()
    }
    result = document_cluster_bootstrap(records, ["doc-a", "doc-a", "doc-b"],
                                        draws=32, seed=7)
    assert result["n_documents"] == 2
    assert result["ceilings"]["simple"]["ci95"] == pytest.approx([0.5, 0.5])
    expected = arm_gains({name: (8.0 - ce) / 5.0
                          for name, ce in ces.items() if name not in ("live", "constant")})
    assert result["gains"]["both"]["mean"] == pytest.approx(expected["both"])
    assert result["gains"]["both_minus_attn"]["ci95"] == pytest.approx([0.02, 0.02])
    assert result["gains"]["both_minus_mlp"]["ci95"] == pytest.approx([0.04, 0.04])
