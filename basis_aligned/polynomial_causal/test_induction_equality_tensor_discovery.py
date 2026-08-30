import torch

import induction_equality_tensor_discovery as subject


def test_bootstrap_recovers_planted_effects():
    ledger = subject.empty_ledger()
    for _document in range(40):
        for arm in subject.ARMS:
            for cell in subject.CELLS:
                count = 4
                if arm in ("native", "full_replay"):
                    ce = 2.0
                elif arm == "heads_deleted":
                    ce = 3.0
                elif arm == "extract_equality":
                    ce = 2.2
                elif arm == "remove_equality":
                    ce = 2.5 if cell == "positive" else 2.01
                else:
                    ce = 2.9
                ledger[arm][cell]["count"].append(count)
                ledger[arm][cell]["loss_sum"].append(count * ce)
                ledger[arm][cell]["kl_sum"].append(0.0)
                ledger[arm][cell]["top1_changes"].append(0.0)
    effects = subject.bootstrap_effects(ledger)
    assert abs(effects["target_damage"]["mean"] - 0.5) < 1e-12
    assert abs(effects["specificity"]["mean"] - 0.49) < 1e-12
    assert abs(effects["extraction_recovery"]["mean"] - 0.8) < 1e-12


def test_pooled_reports_count_and_average_exactly():
    ledger = subject.empty_ledger()
    for arm in subject.ARMS:
        for cell in subject.CELLS:
            ledger[arm][cell]["count"] = [2, 3]
            ledger[arm][cell]["loss_sum"] = [4.0, 9.0]
            ledger[arm][cell]["kl_sum"] = [0.2, 0.3]
            ledger[arm][cell]["top1_changes"] = [1.0, 2.0]
    report = subject.pooled_reports(ledger)["native"]["positive"]
    assert report == {
        "tokens": 5,
        "ce": 2.6,
        "native_to_arm_kl": 0.1,
        "top1_change_fraction": 0.6,
    }
