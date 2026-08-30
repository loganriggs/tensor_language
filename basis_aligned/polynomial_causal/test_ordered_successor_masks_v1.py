import unittest

import torch

from ordered_successor_masks_v1 import (
    OrderedLexicon,
    build_ordered_successor_masks,
    support_by_cell,
)


LEXICON = OrderedLexicon("letters", ((10,), (11,), (12,)))


def mask_for(row, *, window=8):
    return build_ordered_successor_masks(
        torch.tensor([row], dtype=torch.long),
        LEXICON,
        window=window,
        first_prediction=0,
    )


class OrderedSuccessorMaskTests(unittest.TestCase):
    def test_remote_positive_excludes_query_token(self):
        # p=3 predicts token 11.  Token 10 is remote; the query token is neutral.
        result = mask_for([10, 90, 91, 92, 11])
        self.assertTrue(result.positive_clean[0, 3])
        self.assertEqual(int(result.pair_index[0, 3]), 0)

        # An otherwise identical immediate 10->11 transition is deliberately
        # separated from the remote clean cell.
        local = mask_for([90, 91, 92, 10, 11])
        self.assertFalse(local.positive_clean[0, 3])
        self.assertTrue(local.excluded_local_or_ambiguous[0, 3])

    def test_copy_overlap_and_copy_only_are_separate(self):
        both = mask_for([10, 11, 90, 91, 11])
        self.assertTrue(both.successor_copy_overlap[0, 3])
        self.assertFalse(both.copy_only[0, 3])

        copy = mask_for([11, 90, 91, 92, 11])
        self.assertTrue(copy.copy_only[0, 3])
        self.assertFalse(copy.successor_copy_overlap[0, 3])

    def test_wrong_source_and_no_source_are_exact_controls(self):
        # Predicting 11: 12 is a same-family but incorrect remote source.
        wrong = mask_for([12, 90, 91, 92, 11])
        self.assertTrue(wrong.wrong_source_clean[0, 3])
        self.assertFalse(wrong.no_source_clean[0, 3])

        absent = mask_for([80, 90, 91, 92, 11])
        self.assertTrue(absent.no_source_clean[0, 3])
        self.assertFalse(absent.wrong_source_clean[0, 3])

    def test_window_and_first_prediction_are_enforced(self):
        rows = torch.tensor([[10, 80, 81, 82, 83, 84, 11]], dtype=torch.long)
        result = build_ordered_successor_masks(
            rows, LEXICON, window=3, first_prediction=2
        )
        # Token 10 lies outside [p-3,p), so this is a no-source control.
        self.assertTrue(result.no_source_clean[0, 5])
        self.assertFalse(result.positive_clean[0, 5])
        self.assertFalse(result.eligible_target[0, :2].any())

    def test_cells_partition_targets_and_support_counts_documents(self):
        rows = torch.tensor(
            [
                [10, 90, 91, 92, 11],
                [10, 80, 81, 82, 11],
                [12, 70, 71, 72, 11],
            ],
            dtype=torch.long,
        )
        result = build_ordered_successor_masks(
            rows, LEXICON, window=8, first_prediction=0
        )
        result.validate_partition()
        support = support_by_cell(result)
        self.assertEqual(support["positive_clean"], {"positions": 2, "documents": 2})
        self.assertEqual(support["wrong_source_clean"], {"positions": 1, "documents": 1})

    def test_lexicon_and_rows_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "two ordered items"):
            OrderedLexicon("bad", ((10,), (10, 11)))
        with self.assertRaisesRegex(TypeError, "integer token IDs"):
            build_ordered_successor_masks(
                torch.zeros((1, 4), dtype=torch.float32), LEXICON, first_prediction=0
            )


if __name__ == "__main__":
    unittest.main()
