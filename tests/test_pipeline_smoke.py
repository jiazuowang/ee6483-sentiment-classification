"""Small checks on artificial text; no instructor-provided data are required."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

spec = importlib.util.spec_from_file_location(
    "sentiment_experiments",
    Path(__file__).resolve().parents[1] / "code" / "run_experiments.py",
)
experiments = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiments)


class PipelineSmokeTest(unittest.TestCase):
    def test_selected_pipeline_accepts_raw_text(self):
        configs = dict(experiments.candidates())
        self.assertEqual(len(configs), 19)
        texts = [
            "wonderful useful product", "excellent useful product",
            "wonderful reliable product", "excellent reliable product",
            "terrible broken product", "awful broken product",
            "terrible useless product", "awful useless product",
        ]
        labels = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        pipeline = configs["SVM_word_char_balanced_C0.25"]
        pipeline.fit(texts, labels)
        result = pipeline.predict(["excellent useful product", "terrible broken product"])
        self.assertEqual(result.tolist(), [1, 0])
        self.assertGreater(pipeline[:-1].transform(texts).shape[1], 0)

    def test_duplicate_groups_stay_together(self):
        groups = np.repeat(np.arange(12), 2)
        labels = np.repeat(np.arange(12) % 2, 2)
        splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=2026)
        for training, validation in splitter.split(np.zeros(len(groups)), labels, groups):
            self.assertTrue(set(groups[training]).isdisjoint(groups[validation]))


if __name__ == "__main__":
    unittest.main()
