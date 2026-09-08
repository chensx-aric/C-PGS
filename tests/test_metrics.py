import unittest

import numpy as np

from cpgs.metrics import confusion_matrix, mean_iou


class MetricTests(unittest.TestCase):
    def test_confusion_and_mean_iou(self):
        prediction = np.asarray([[0, 1], [1, 1]])
        target = np.asarray([[0, 0], [1, 255]])
        matrix = confusion_matrix([prediction], [target], num_classes=2)
        np.testing.assert_array_equal(matrix, [[1, 1], [0, 1]])
        score, per_class = mean_iou(matrix)
        np.testing.assert_allclose(per_class, [0.5, 0.5])
        self.assertAlmostEqual(score, 0.5)


if __name__ == "__main__":
    unittest.main()

