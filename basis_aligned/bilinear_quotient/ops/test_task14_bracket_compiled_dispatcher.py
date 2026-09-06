#!/usr/bin/env python3
from __future__ import annotations

import unittest

import task14_bracket_compiled_dispatcher as dispatcher


class DispatcherTest(unittest.TestCase):
    def setUp(self):
        vector = [1.0] * dispatcher.WIDTH
        self.package = {"programs": {
            "task14": {"vectors": {"singular_to_plural.cardinality_2": vector}},
            "bracket": {"vectors": {"1->8": vector}},
        }}

    def test_task14(self):
        self.assertIs(dispatcher.dispatch_task14(self.package, recipient_number="singular", donor_number="plural", cardinality=2), self.package["programs"]["task14"]["vectors"]["singular_to_plural.cardinality_2"])

    def test_bracket_and_zero(self):
        self.assertIs(dispatcher.dispatch_bracket(self.package, recipient_closer_id=1, donor_closer_id=8), self.package["programs"]["bracket"]["vectors"]["1->8"])
        self.assertEqual(dispatcher.dispatch_bracket(self.package, recipient_closer_id=8, donor_closer_id=8), [0.0] * dispatcher.WIDTH)

    def test_rejections(self):
        with self.assertRaises(dispatcher.DispatchError):
            dispatcher.dispatch_task14(self.package, recipient_number="singular", donor_number="singular", cardinality=2)
        with self.assertRaises(dispatcher.DispatchError):
            dispatcher.dispatch_bracket(self.package, recipient_closer_id=2, donor_closer_id=8)


if __name__ == "__main__":
    unittest.main()
