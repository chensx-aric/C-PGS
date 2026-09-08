import unittest

import numpy as np

from cpgs.style_prior import Hyperedge, infer_style_prior, inject_style_prior_numpy


class StylePriorTests(unittest.TestCase):
    def setUp(self):
        self.edges = (
            Hyperedge(frozenset({1, 2, 3}), "a", "large"),
            Hyperedge(frozenset({1, 2}), "a", "small"),
            Hyperedge(frozenset({2, 4}), "a", "other"),
            Hyperedge(frozenset({0, 4}), "b", "wrong-group"),
        )

    def test_max_overlap_then_minimum_cardinality(self):
        result = infer_style_prior([0, 1, 1, 0, 0], "a", self.edges, 5)
        self.assertEqual(result.overlap_score, 2)
        self.assertEqual(result.prototype.sample_id, "small")
        np.testing.assert_array_equal(result.prior, [0, 1, 1, 1, 0])

    def test_inclusive_support_bounds(self):
        result = infer_style_prior(
            [0, 1, 1, 0, 0],
            "a",
            self.edges,
            5,
            support_lower=2,
            support_upper=2,
        )
        np.testing.assert_array_equal(result.prior, [0, 1, 1, 0, 0])

    def test_author_confirmed_default_upper_bound_is_three(self):
        edges = (
            Hyperedge(frozenset({1}), "a", "prototype"),
            Hyperedge(frozenset({1, 4}), "a", "support-1"),
            Hyperedge(frozenset({1, 2, 4}), "a", "support-2"),
            Hyperedge(frozenset({1, 3, 4}), "a", "support-3"),
            Hyperedge(frozenset({0, 1, 4}), "a", "support-4"),
        )
        result = infer_style_prior([0, 1, 0, 0, 0], "a", edges, 5)
        self.assertEqual(result.support[4], 4)
        np.testing.assert_array_equal(result.prior, [1, 1, 1, 1, 0])

    def test_missing_group_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "no reference hyperedges"):
            infer_style_prior([1, 0, 0, 0, 0], "missing", self.edges, 5)

    def test_logit_injection_broadcasts_over_pixels(self):
        logits = np.zeros((2, 5, 3, 4), dtype=np.float32)
        prior = np.asarray([0, 1, 0, 1, 0], dtype=np.float32)
        adjusted = inject_style_prior_numpy(logits, prior, alpha=0.6)
        self.assertTrue(np.allclose(adjusted[:, 1], 0.6))
        self.assertTrue(np.allclose(adjusted[:, 3], 0.6))
        self.assertTrue(np.allclose(adjusted[:, (0, 2, 4)], 0.0))


if __name__ == "__main__":
    unittest.main()
