import importlib.util
import unittest

import numpy as np

from cpgs.experiment_ip import (
    classify_instance_id_map,
    fuse_experiment_prediction,
)


class ExperimentInstancePriorTests(unittest.TestCase):
    def test_center_fallback_matches_verified_evaluator(self):
        prediction = np.asarray(
            [
                [2, 2, 3],
                [2, 3, 2],
                [3, 3, 2],
            ],
            dtype=np.int64,
        )
        instances = np.ones_like(prediction)
        classified = classify_instance_id_map(
            prediction,
            instances,
            num_classes=5,
            dominance_threshold=0.6,
        )
        self.assertTrue(np.all(classified == 3))

    @unittest.skipUnless(importlib.util.find_spec("cv2"), "OpenCV is not installed")
    def test_prediction_only_policy_preserves_prediction(self):
        prediction = np.zeros((7, 7), dtype=np.int64)
        classified = np.full((7, 7), 6, dtype=np.int64)
        fused = fuse_experiment_prediction(
            prediction,
            classified,
            normalized_expansion=0.6,
            radius_scale=100,
        )
        np.testing.assert_array_equal(fused, prediction)


if __name__ == "__main__":
    unittest.main()
