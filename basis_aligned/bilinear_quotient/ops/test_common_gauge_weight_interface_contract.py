#!/usr/bin/env python3
from types import SimpleNamespace
import unittest

import torch

import common_gauge_weight_interface_contract as contract


def linear(rows, columns):
    return SimpleNamespace(weight=torch.arange(1, rows * columns + 1, dtype=torch.float32)
                           .reshape(rows, columns))


def model():
    blocks = []
    for _layer in range(18):
        attention = SimpleNamespace(c_proj=linear(18, 18), **{
            name: linear(18, 18) for name in contract.ATTENTION_READERS})
        mlp = SimpleNamespace(Down=linear(18, 36), Left=linear(36, 18), Right=linear(36, 18))
        blocks.append(SimpleNamespace(attn=attention, mlp=mlp))
    return SimpleNamespace(config=SimpleNamespace(n_embd=18, n_head=9),
                           transformer=SimpleNamespace(h=blocks))


class CommonGaugeWeightContractTests(unittest.TestCase):
    def test_projection_fractions_are_exact(self):
        basis = torch.eye(4)[:, :2]
        writer = torch.eye(4)
        reader = torch.eye(4)
        self.assertAlmostEqual(contract.projected_fraction(torch, writer, basis, "writer")["fraction"], .5)
        self.assertAlmostEqual(contract.projected_fraction(torch, reader, basis, "reader")["enrichment"], 1.0)

    def test_complete_inventory(self):
        interfaces = contract.weight_interfaces(model())
        self.assertEqual(sum(role == "writer" for role, _label, _matrix in interfaces), 180)
        self.assertEqual(sum(role == "reader" for role, _label, _matrix in interfaces), 846)
        self.assertEqual(len({(role, label) for role, label, _matrix in interfaces}), 1026)

    def test_score_and_top_label_are_deterministic(self):
        basis = torch.linalg.qr(torch.arange(1, 37, dtype=torch.float32).reshape(18, 2)).Q
        records = contract.score_interfaces(torch, model(), {"task": basis})
        self.assertEqual(len(records), 1026)
        self.assertEqual(len(contract.top_labels(records, "reader", "task_enrichment")), 20)
        self.assertTrue(all(0 <= row["task_fraction"] <= 1 for row in records))

    def test_jaccard_and_guards(self):
        self.assertAlmostEqual(contract.jaccard(["a", "b"], ["b", "c"]), 1 / 3)
        with self.assertRaises(ValueError):
            contract.projected_fraction(torch, torch.eye(4), torch.ones(4, 2), "reader")
        with self.assertRaises(ValueError):
            contract.projected_fraction(torch, torch.zeros(4, 4), torch.eye(4)[:, :2], "writer")


if __name__ == "__main__":
    unittest.main()
