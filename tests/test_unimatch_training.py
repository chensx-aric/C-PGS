from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys
import unittest

import torch


TRAIN_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "integrations"
    / "unimatch_v2"
    / "train.py"
)
SPEC = importlib.util.spec_from_file_location("cpgs_unimatch_training", TRAIN_PATH)
assert SPEC is not None and SPEC.loader is not None
TRAIN = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TRAIN
SPEC.loader.exec_module(TRAIN)


class UniMatchTrainingConfigurationTests(unittest.TestCase):
    def test_split_aliases_share_one_canonical_mapping(self) -> None:
        expected = {
            "1/16": "1/16",
            "1_16": "1/16",
            "0.0625": "1/16",
            "1/8": "1/8",
            "0.125": "1/8",
            "1/4": "1/4",
            "0.25": "1/4",
            "1/2": "1/2",
            "0.5": "1/2",
        }
        for value, canonical in expected.items():
            with self.subTest(value=value):
                self.assertEqual(TRAIN.canonical_split(value), canonical)

    def test_invalid_split_fails_before_training(self) -> None:
        with self.assertRaises(argparse.ArgumentTypeError):
            TRAIN.canonical_split("1/3")

    def test_style_prior_cache_shape_and_alignment(self) -> None:
        entries = [("image-a.jpg", "label-a.npy"), ("image-b.jpg", "label-b.npy")]
        cache = [torch.zeros(18), torch.ones(18)]
        TRAIN._validate_prior_cache(cache, entries, 18)

        with self.assertRaisesRegex(ValueError, "1 rows; manifest has 2"):
            TRAIN._validate_prior_cache(cache[:1], entries, 18)
        with self.assertRaisesRegex(ValueError, r"expected \(18,\)"):
            TRAIN._validate_prior_cache([torch.zeros(17), torch.ones(18)], entries, 18)


if __name__ == "__main__":
    unittest.main()
