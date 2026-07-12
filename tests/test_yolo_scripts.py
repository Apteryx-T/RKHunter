from __future__ import annotations

import unittest

from scripts.predict_yolo import repo_path as prediction_repo_path
from scripts.predict_yolo import safe_run_name as prediction_run_name
from scripts.predict_yolo import validate_source
from scripts.train_yolo import repo_path as training_repo_path
from scripts.train_yolo import safe_run_name as training_run_name


class YoloScriptBoundaryTest(unittest.TestCase):
    def test_local_paths_and_run_names_are_accepted(self) -> None:
        self.assertTrue(prediction_repo_path("pyproject.toml", must_be_file=True).is_file())
        self.assertTrue(training_repo_path("pyproject.toml", must_be_file=True).is_file())
        self.assertEqual(prediction_run_name("reviewed-v1.0"), "reviewed-v1.0")
        self.assertEqual(training_run_name("reviewed-v1.0"), "reviewed-v1.0")

    def test_paths_cannot_escape_the_repository(self) -> None:
        with self.assertRaises(SystemExit):
            prediction_repo_path("../outside")
        with self.assertRaises(SystemExit):
            training_repo_path("../outside")

    def test_run_names_cannot_contain_paths(self) -> None:
        for value in (
            "",
            ".",
            "..",
            "../escape",
            "nested/run",
            "nested\\run",
            "CON",
            "NUL.txt",
            "COM1",
            "LPT9.log",
            "run.",
        ):
            with self.subTest(value=value):
                with self.assertRaises(SystemExit):
                    prediction_run_name(value)
                with self.assertRaises(SystemExit):
                    training_run_name(value)

    def test_indirect_prediction_sources_are_rejected(self) -> None:
        for value in ("images.streams", "images.txt", "images.csv"):
            with self.subTest(value=value), self.assertRaises(SystemExit):
                validate_source(prediction_repo_path(value))


if __name__ == "__main__":
    unittest.main()
