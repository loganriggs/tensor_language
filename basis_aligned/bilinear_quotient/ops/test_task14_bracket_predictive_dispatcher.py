#!/usr/bin/env python3
import unittest
import task14_bracket_predictive_dispatcher as d
class TestDispatcher(unittest.TestCase):
 def setUp(self):self.p={"programs":{"task14":{"singular_to_plural.cardinality_0":{"displacement":[1.]*d.WIDTH,"predicted_donorward_effect":.1}},"bracket":{"absolute_terms":{"8":[2.]*d.WIDTH},"predicted_effects":{"1->8":3.}}}}
 def test_task(self):self.assertEqual(d.dispatch_task14(self.p,recipient_number="singular",donor_number="plural",cardinality=0)["operation"],"add_displacement")
 def test_bracket(self):self.assertEqual(d.dispatch_bracket(self.p,recipient_closer_id=1,donor_closer_id=8)["operation"],"replace_absolute");self.assertEqual(d.dispatch_bracket(self.p,recipient_closer_id=1,donor_closer_id=1)["operation"],"no_edit")
 def test_reject(self):
  with self.assertRaises(d.DispatchError):d.dispatch_bracket(self.p,recipient_closer_id=2,donor_closer_id=8)
if __name__=="__main__":unittest.main()
