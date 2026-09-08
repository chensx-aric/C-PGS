import unittest

import numpy as np

from cpgs.instance_prior import refine_with_instance_prior


class InstancePriorTests(unittest.TestCase):
    def test_center_fallback_below_dominance_threshold(self):
        prediction = np.zeros((7, 7), dtype=np.int64)
        prediction[2:5, 2:5] = np.asarray(
            [[2, 2, 3], [2, 3, 2], [3, 3, 2]], dtype=np.int64
        )
        mask = np.zeros_like(prediction, dtype=bool)
        mask[2:5, 2:5] = True

        refined, audit = refine_with_instance_prior(
            prediction,
            [mask],
            num_classes=4,
            dominance_threshold=0.6,
            normalized_expansion=0,
        )
        self.assertEqual(audit[0].dominant_label, 2)
        self.assertTrue(audit[0].used_center_fallback)
        self.assertEqual(audit[0].label, 3)
        self.assertTrue(np.all(refined[mask] == 3))

    def test_dominant_class_and_radius_conversion(self):
        prediction = np.zeros((9, 9), dtype=np.int64)
        prediction[3:6, 3:6] = 2
        mask = np.zeros_like(prediction, dtype=bool)
        mask[3:6, 3:6] = True

        refined, audit = refine_with_instance_prior(
            prediction,
            [{"segmentation": mask}],
            num_classes=3,
            dominance_threshold=0.6,
            normalized_expansion=0.6,
            radius_scale=2.0,
        )
        self.assertEqual(audit[0].radius_pixels, 1)
        self.assertEqual(audit[0].label, 2)
        self.assertFalse(audit[0].used_center_fallback)
        self.assertGreater(audit[0].refined_pixels, audit[0].source_pixels)
        self.assertEqual(refined[2, 4], 2)

    def test_histogram_uses_initial_prediction_for_every_mask(self):
        prediction = np.zeros((5, 7), dtype=np.int64)
        prediction[1:4, 1:3] = 1
        prediction[1:4, 4:6] = 2
        left = np.zeros_like(prediction, dtype=bool)
        right = np.zeros_like(prediction, dtype=bool)
        left[1:4, 1:3] = True
        right[1:4, 4:6] = True

        _, audit = refine_with_instance_prior(
            prediction,
            [left, right],
            num_classes=3,
            normalized_expansion=1,
            radius_scale=2,
        )
        self.assertEqual([step.dominant_label for step in audit], [1, 2])


if __name__ == "__main__":
    unittest.main()

