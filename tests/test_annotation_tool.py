from __future__ import annotations

import json
import os
import sqlite3
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from rkhunter.annotator.adapters import default_registry
from rkhunter.annotator import DATABASE_SCHEMA_VERSION
from rkhunter.annotator.api import JobManager, create_app
from rkhunter.annotator.database import SCHEMA_V1
from rkhunter.annotator.models import BoxProposal
from rkhunter.annotator.service import AnnotationService, RevisionConflict
from fastapi.testclient import TestClient
from train_yolo import verify_export_manifest


class FakeAdapter:
    adapter_name = "test_fake"

    def __init__(self, context):
        self.context = context

    def propose(self, image_path, *, image_width, image_height, params):
        return [
            BoxProposal(
                class_id=0,
                x1=10,
                y1=5,
                x2=min(60, image_width),
                y2=min(35, image_height),
                confidence=0.7,
                source="auto",
                model_id=self.context.model.id,
            ),
            BoxProposal(
                class_id=4,
                x1=1,
                y1=1,
                x2=5,
                y2=5,
                confidence=0.9,
                source="auto",
                model_id=self.context.model.id,
            ),
        ]


class ConcurrentEditAdapter:
    adapter_name = "test_concurrent_edit"
    adapter_version = "1"
    callback = None

    def __init__(self, context):
        self.context = context

    def propose(self, image_path, *, image_width, image_height, params):
        if type(self).callback:
            type(self).callback()
        return [
            BoxProposal(
                class_id=0,
                x1=5,
                y1=5,
                x2=min(30, image_width),
                y2=min(30, image_height),
                confidence=0.8,
                source="auto",
            )
        ]


if "test_fake" not in default_registry.names():
    default_registry.register("test_fake", FakeAdapter)
if "test_concurrent_edit" not in default_registry.names():
    default_registry.register("test_concurrent_edit", ConcurrentEditAdapter)


class AnnotationToolTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.dataset = self.root / "dataset"
        for split in ("train", "val", "test"):
            (self.dataset / "images" / split).mkdir(parents=True)
            (self.dataset / "labels" / split).mkdir(parents=True)
        Image.new("RGB", (100, 50), "#d8bd87").save(
            self.dataset / "images" / "train" / "rock.jpg"
        )
        Image.new("RGB", (80, 80), "#c8b98f").save(
            self.dataset / "images" / "val" / "empty.jpg"
        )
        (self.dataset / "labels" / "train" / "rock.txt").write_text(
            "0 0.500000 0.500000 0.400000 0.400000\n", encoding="utf-8"
        )
        (self.dataset / "labels" / "val" / "empty.txt").write_text("", encoding="utf-8")
        self.service = AnnotationService(self.root, self.root / "runtime" / "annotations.db")
        self.classes = {
            0: "suspected_meteorite",
            1: "dark_rock",
            2: "metal_debris",
            3: "shadow",
            4: "background",
        }
        self.service.register_project("test-project", "Test", self.dataset, self.classes)
        self.import_result = self.service.import_yolo("test-project")

    def tearDown(self):
        self.temp.cleanup()

    def test_import_pixel_coordinates_and_empty_background(self):
        self.assertEqual(self.import_result["images"], 2)
        self.assertEqual(self.import_result["labels"], 2)
        self.assertEqual(self.import_result["boxes"], 1)
        payload = self.service.list_images("test-project", limit=10)
        rock = next(item for item in payload["items"] if item["rel_path"].endswith("rock.jpg"))
        detail = self.service.get_image("test-project", rock["id"])
        box = detail["annotations"][0]
        self.assertAlmostEqual(box["x1"], 30)
        self.assertAlmostEqual(box["y1"], 15)
        self.assertAlmostEqual(box["x2"], 70)
        self.assertAlmostEqual(box["y2"], 35)
        empty = next(item for item in payload["items"] if item["rel_path"].endswith("empty.jpg"))
        self.assertEqual(self.service.get_image("test-project", empty["id"])["annotations"], [])

    def test_revision_conflict_and_audit_transaction(self):
        item = self.service.list_images("test-project", limit=10)["items"][0]
        detail = self.service.get_image("test-project", item["id"])
        saved = self.service.save_annotations(
            "test-project",
            item["id"],
            [{"class_id": 1, "x1": 2, "y1": 3, "x2": 20, "y2": 25}],
            expected_revision=detail["revision"],
            status="reviewed",
        )
        self.assertEqual(saved["revision"], detail["revision"] + 1)
        self.assertEqual(saved["annotations"][0]["status"], "approved")
        with self.assertRaises(RevisionConflict):
            self.service.save_annotations(
                "test-project", item["id"], [], expected_revision=detail["revision"]
            )
        events = self.service.audit_events("test-project")
        self.assertTrue(any(event["action"] == "annotations_saved" for event in events))

        second_service = AnnotationService(self.root, self.service.database.path)
        with self.assertRaises(RevisionConflict):
            second_service.save_annotations(
                "test-project", item["id"], [], expected_revision=detail["revision"]
            )

    def test_unknown_and_background_class_handling(self):
        item = self.service.list_images("test-project", limit=10)["items"][0]
        detail = self.service.get_image("test-project", item["id"])
        with self.assertRaises(ValueError):
            self.service.save_annotations(
                "test-project",
                item["id"],
                [{"class_id": 99, "x1": 1, "y1": 1, "x2": 4, "y2": 4}],
                expected_revision=detail["revision"],
            )
        with self.assertRaises(ValueError):
            self.service.save_annotations(
                "test-project",
                item["id"],
                [{"class_id": 4, "x1": 1, "y1": 1, "x2": 4, "y2": 4}],
                expected_revision=detail["revision"],
            )
        model_file = self.root / "fake.weights"
        model_file.write_text("fake", encoding="utf-8")
        self.service.register_model("fake", "Fake", "test_fake", model_file)
        auto = self.service.auto_label_image("test-project", item["id"], "fake", {})
        self.assertEqual(len(auto["annotations"]), 1)
        self.assertEqual(auto["annotations"][0]["class_id"], 0)
        self.assertEqual(auto["annotations"][0]["status"], "draft")

    def test_human_empty_and_rejected_images_cannot_be_auto_overwritten(self):
        model_file = self.root / "fake.weights"
        model_file.write_text("fake", encoding="utf-8")
        self.service.register_model("fake", "Fake", "test_fake", model_file)
        items = self.service.list_images("test-project", limit=10)["items"]
        for status, item in zip(("reviewed", "rejected"), items, strict=True):
            detail = self.service.get_image("test-project", item["id"])
            self.service.save_annotations(
                "test-project",
                item["id"],
                [],
                expected_revision=detail["revision"],
                status=status,
            )
            with self.assertRaises(ValueError):
                self.service.auto_label_image("test-project", item["id"], "fake", {})

    def test_stale_auto_inference_cannot_overwrite_concurrent_draft_edit(self):
        model_file = self.root / "race.weights"
        model_file.write_text("race", encoding="utf-8")
        self.service.register_model(
            "race-model", "Race", "test_concurrent_edit", model_file
        )
        item = self.service.list_images("test-project", limit=10)["items"][0]
        before = self.service.get_image("test-project", item["id"])
        second_service = AnnotationService(self.root, self.service.database.path)

        def concurrent_edit():
            second_service.save_annotations(
                "test-project",
                item["id"],
                [{"class_id": 1, "x1": 2, "y1": 2, "x2": 12, "y2": 12}],
                expected_revision=before["revision"],
                status="auto_labeled",
                actor="concurrent-editor",
            )

        ConcurrentEditAdapter.callback = concurrent_edit
        try:
            with self.assertRaises(RevisionConflict):
                self.service.auto_label_image(
                    "test-project", item["id"], "race-model", {}
                )
        finally:
            ConcurrentEditAdapter.callback = None
        after = self.service.get_image("test-project", item["id"])
        self.assertEqual(after["revision"], before["revision"] + 1)
        self.assertEqual(after["annotations"][0]["class_id"], 1)

    def test_warm_adapter_cache_still_detects_weight_content_drift(self):
        model_file = self.root / "drift.weights"
        model_file.write_text("aaaa", encoding="utf-8")
        self.service.register_model("drift", "Drift", "test_fake", model_file)
        item = self.service.list_images("test-project", limit=10)["items"][0]
        self.service.auto_label_image("test-project", item["id"], "drift", {})
        stat = model_file.stat()
        model_file.write_text("bbbb", encoding="utf-8")
        os.utime(model_file, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        with self.assertRaises(RuntimeError):
            self.service.auto_label_image("test-project", item["id"], "drift", {})

    def test_model_alias_upgrades_keep_immutable_annotation_provenance(self):
        model_file = self.root / "versioned.weights"
        model_file.write_text("model-v1", encoding="utf-8")
        first = self.service.register_model(
            "versioned", "Versioned", "test_fake", model_file, {"threshold": 0.1}
        )
        item = self.service.list_images("test-project", limit=10)["items"][0]
        run_id = self.service.create_run(
            "test-project", "versioned", [item["id"]], {"conf": 0.1}
        )
        second = self.service.register_model(
            "versioned", "Versioned", "test_fake", model_file, {"threshold": 0.2}
        )
        self.assertNotEqual(first["revision_id"], second["revision_id"])
        run = self.service.get_run(run_id)
        self.assertEqual(run["model_revision_id"], first["revision_id"])
        labeled = self.service.auto_label_image(
            "test-project",
            item["id"],
            "versioned",
            {},
            run_id=run_id,
            model_revision_id=run["model_revision_id"],
        )
        self.assertEqual(
            labeled["annotations"][0]["model_revision_id"], first["revision_id"]
        )
        revisions = self.service.list_model_revisions("versioned")
        self.assertEqual({value["id"] for value in revisions}, {first["revision_id"], second["revision_id"]})

        model_file.write_text("model-v2", encoding="utf-8")
        third = self.service.register_model(
            "versioned", "Versioned", "test_fake", model_file, {"threshold": 0.2}
        )
        self.assertNotEqual(second["revision_id"], third["revision_id"])
        historical = self.service.get_image("test-project", item["id"])
        self.assertEqual(
            historical["annotations"][0]["model_revision_id"], first["revision_id"]
        )

    def test_client_cannot_pair_model_alias_with_another_model_revision(self):
        model_file = self.root / "lineage.weights"
        model_file.write_text("lineage", encoding="utf-8")
        first = self.service.register_model("lineage-a", "A", "test_fake", model_file)
        second = self.service.register_model("lineage-b", "B", "test_fake", model_file)
        item = self.service.list_images("test-project", limit=10)["items"][0]
        detail = self.service.get_image("test-project", item["id"])
        with self.assertRaises(ValueError):
            self.service.save_annotations(
                "test-project",
                item["id"],
                [
                    {
                        "class_id": 0,
                        "x1": 2,
                        "y1": 2,
                        "x2": 12,
                        "y2": 12,
                        "model_id": first["id"],
                        "model_revision_id": second["revision_id"],
                    }
                ],
                expected_revision=detail["revision"],
            )

    def test_interrupted_batch_run_is_recovered_and_claimed_once(self):
        model_file = self.root / "recover.weights"
        model_file.write_text("recover", encoding="utf-8")
        self.service.register_model("recover", "Recover", "test_fake", model_file)
        item = self.service.list_images("test-project", limit=10)["items"][0]
        run_id = self.service.create_run(
            "test-project", "recover", [item["id"]], {}
        )
        self.service.update_run(run_id, status="running")
        self.assertIn(run_id, self.service.recover_interrupted_runs())
        self.assertTrue(self.service.claim_run(run_id))
        self.assertFalse(self.service.claim_run(run_id))

    def test_resumed_batch_preserves_errors_from_before_interruption(self):
        model_file = self.root / "resume.weights"
        model_file.write_text("resume", encoding="utf-8")
        self.service.register_model("resume", "Resume", "test_fake", model_file)
        items = self.service.list_images("test-project", limit=10)["items"]
        run_id = self.service.create_run(
            "test-project", "resume", [item["id"] for item in items], {}
        )
        prior_error = {"image_id": items[0]["id"], "error": "before-crash"}
        self.service.update_run(
            run_id,
            completed=1,
            result_json=json.dumps({"errors": [prior_error]}),
        )
        jobs = JobManager(self.service)
        jobs._execute(run_id)
        jobs.shutdown()
        run = self.service.get_run(run_id)
        self.assertEqual(run["status"], "completed_with_errors")
        self.assertEqual(run["result"]["errors"], [prior_error])

    def test_import_detects_source_label_changes(self):
        label = self.dataset / "labels" / "train" / "rock.txt"
        label.write_text("1 0.500000 0.500000 0.200000 0.200000\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            self.service.import_yolo("test-project")

    def test_missing_label_cannot_be_imported_as_reviewed_background(self):
        dataset = self.root / "missing-label-dataset"
        (dataset / "images" / "train").mkdir(parents=True)
        (dataset / "labels" / "train").mkdir(parents=True)
        Image.new("RGB", (32, 32), "#77664f").save(
            dataset / "images" / "train" / "missing.jpg"
        )
        Image.new("RGB", (32, 32), "#88755a").save(
            dataset / "images" / "train" / "empty.jpg"
        )
        (dataset / "labels" / "train" / "empty.txt").write_text("", encoding="utf-8")
        self.service.register_project(
            "missing-labels", "Missing labels", dataset, self.classes
        )
        self.service.import_yolo("missing-labels", imported_status="reviewed")
        items = self.service.list_images("missing-labels", limit=10)["items"]
        statuses = {item["rel_path"]: item["status"] for item in items}
        self.assertEqual(statuses["train/missing.jpg"], "unreviewed")
        self.assertEqual(statuses["train/empty.jpg"], "reviewed")

    def test_export_is_new_revision_and_round_trips_yolo(self):
        test_image = self.dataset / "images" / "test" / "background.jpg"
        Image.new("RGB", (48, 48), "#9b825e").save(test_image)
        (self.dataset / "labels" / "test" / "background.txt").write_text(
            "", encoding="utf-8"
        )
        self.service.import_yolo("test-project")
        items = self.service.list_images("test-project", limit=10)["items"]
        for item in items:
            detail = self.service.get_image("test-project", item["id"])
            boxes = []
            if item["rel_path"].endswith("rock.jpg"):
                boxes = [{"class_id": 0, "x1": 10, "y1": 5, "x2": 50, "y2": 25}]
            elif item["rel_path"].endswith("empty.jpg"):
                boxes = [{"class_id": 1, "x1": 8, "y1": 8, "x2": 32, "y2": 32}]
            self.service.save_annotations(
                "test-project",
                item["id"],
                boxes,
                expected_revision=detail["revision"],
                status="reviewed",
            )
        first = self.service.export_yolo("test-project", self.root / "exports")
        second = self.service.export_yolo("test-project", self.root / "exports")
        self.assertNotEqual(first["revision"], second["revision"])
        label = Path(first["path"]) / "labels" / "train" / "rock.txt"
        self.assertEqual(label.read_text(encoding="utf-8"), "0 0.300000 0.300000 0.400000 0.400000\n")
        empty = Path(first["path"]) / "labels" / "test" / "background.txt"
        self.assertTrue(empty.exists())
        self.assertEqual(empty.read_text(encoding="utf-8"), "")
        manifest = json.loads((Path(first["path"]) / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["images"], 3)
        self.assertEqual(manifest["boxes"], 2)
        self.assertTrue(manifest["train_ready"])
        self.assertEqual(manifest["split_counts"]["train"], 1)
        self.assertEqual(manifest["split_counts"]["val"], 1)
        self.assertEqual(manifest["split_box_counts"]["val"], 1)
        self.assertEqual(manifest["hash_algorithm"], "sha256")
        self.assertTrue(manifest["dataset_yaml_sha256"])
        self.assertTrue(all(item["output_sha256"] for item in manifest["items"]))
        self.assertTrue(all(item["label_sha256"] for item in manifest["items"]))
        self.assertNotIn("background", (Path(first["path"]) / "dataset.yaml").read_text(encoding="utf-8"))
        self.assertNotIn("path:", (Path(first["path"]) / "dataset.yaml").read_text(encoding="utf-8"))
        self.assertTrue((Path(first["path"]) / "images" / "test").is_dir())
        verify_export_manifest(Path(first["path"]) / "dataset.yaml", manifest)
        empty.write_text("tampered\n", encoding="utf-8")
        with self.assertRaises(SystemExit):
            verify_export_manifest(Path(first["path"]) / "dataset.yaml", manifest)

    def test_partial_export_is_explicitly_not_train_ready(self):
        test_image = self.dataset / "images" / "test" / "only.jpg"
        Image.new("RGB", (64, 64), "#b99d70").save(test_image)
        (self.dataset / "labels" / "test" / "only.txt").write_text("", encoding="utf-8")
        self.service.import_yolo("test-project")
        item = next(
            value
            for value in self.service.list_images("test-project", split="test", limit=10)["items"]
            if value["rel_path"] == "test/only.jpg"
        )
        detail = self.service.get_image("test-project", item["id"])
        self.service.save_annotations(
            "test-project",
            item["id"],
            [],
            expected_revision=detail["revision"],
            status="reviewed",
        )
        result = self.service.export_yolo("test-project", self.root / "partial-exports")
        self.assertFalse(result["train_ready"])
        self.assertEqual(result["split_counts"]["test"], 1)
        yaml_text = (Path(result["path"]) / "dataset.yaml").read_text(encoding="utf-8")
        self.assertIn("train: null", yaml_text)
        self.assertIn("val: null", yaml_text)
        self.assertIn("test: images/test", yaml_text)

    def test_nested_paths_are_preserved_and_collisions_are_atomic(self):
        for folder in ("a", "b"):
            image_dir = self.dataset / "images" / "train" / folder
            label_dir = self.dataset / "labels" / "train" / folder
            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            Image.new("RGB", (32, 32), "#a98f65").save(image_dir / "rock.jpg")
            (label_dir / "rock.txt").write_text("", encoding="utf-8")
        self.service.import_yolo("test-project")
        nested = [
            item
            for item in self.service.list_images("test-project", split="train", limit=20)["items"]
            if item["rel_path"] in {"train/a/rock.jpg", "train/b/rock.jpg"}
        ]
        for item in nested:
            detail = self.service.get_image("test-project", item["id"])
            self.service.save_annotations(
                "test-project",
                item["id"],
                [],
                expected_revision=detail["revision"],
                status="reviewed",
            )
        result = self.service.export_yolo("test-project", self.root / "nested-exports")
        export_root = Path(result["path"])
        self.assertTrue((export_root / "images" / "train" / "a" / "rock.jpg").is_file())
        self.assertTrue((export_root / "images" / "train" / "b" / "rock.jpg").is_file())

        collision_dataset = self.root / "collision"
        (collision_dataset / "images" / "train").mkdir(parents=True)
        (collision_dataset / "labels" / "train").mkdir(parents=True)
        for suffix in (".jpg", ".png"):
            Image.new("RGB", (20, 20), "#806b50").save(
                collision_dataset / "images" / "train" / f"same{suffix}"
            )
        (collision_dataset / "labels" / "train" / "same.txt").write_text("", encoding="utf-8")
        self.service.register_project(
            "collision-project", "Collision", collision_dataset, self.classes
        )
        self.service.import_yolo("collision-project")
        for item in self.service.list_images("collision-project", limit=10)["items"]:
            detail = self.service.get_image("collision-project", item["id"])
            self.service.save_annotations(
                "collision-project",
                item["id"],
                [],
                expected_revision=detail["revision"],
                status="reviewed",
            )
        collision_exports = self.root / "collision-exports"
        with self.assertRaises(ValueError):
            self.service.export_yolo("collision-project", collision_exports)
        self.assertEqual(list(collision_exports.iterdir()), [])

    def test_export_rejects_unreviewed_mode_and_cleans_failed_revision(self):
        with self.assertRaises(ValueError):
            self.service.export_yolo(
                "test-project", self.root / "unsafe-exports", reviewed_only=False
            )
        item = self.service.list_images("test-project", limit=10)["items"][0]
        detail = self.service.get_image("test-project", item["id"])
        self.service.save_annotations(
            "test-project",
            item["id"],
            [],
            expected_revision=detail["revision"],
            status="reviewed",
        )
        source = self.dataset / "images" / item["rel_path"]
        source.write_bytes(source.read_bytes() + b"changed")
        export_root = self.root / "failed-exports"
        with self.assertRaises(ValueError):
            self.service.export_yolo("test-project", export_root)
        self.assertEqual(list(export_root.iterdir()), [])

    def test_export_rejects_unmapped_split(self):
        dataset = self.root / "unsplit-dataset"
        (dataset / "images").mkdir(parents=True)
        (dataset / "labels").mkdir(parents=True)
        Image.new("RGB", (24, 24), "#725f48").save(dataset / "images" / "loose.jpg")
        (dataset / "labels" / "loose.txt").write_text(
            "0 0.5 0.5 0.5 0.5\n", encoding="utf-8"
        )
        self.service.register_project("unsplit-project", "Unsplit", dataset, self.classes)
        self.service.import_yolo("unsplit-project", imported_status="reviewed")
        output_root = self.root / "unsplit-exports"
        with self.assertRaises(ValueError):
            self.service.export_yolo("unsplit-project", output_root)
        self.assertEqual(list(output_root.iterdir()), [])

    def test_import_rejects_image_and_label_symlinks_outside_dataset(self):
        with tempfile.TemporaryDirectory() as external_temp:
            external = Path(external_temp)
            external_image = external / "outside.jpg"
            Image.new("RGB", (16, 16), "#44372c").save(external_image)
            image_link = self.dataset / "images" / "train" / "outside-link.jpg"
            try:
                image_link.symlink_to(external_image)
            except OSError as exc:
                self.skipTest(f"symlink creation unavailable: {exc}")
            with self.assertRaises(ValueError):
                self.service.import_yolo("test-project")
            image_link.unlink()

            local_image = self.dataset / "images" / "train" / "label-link.jpg"
            Image.new("RGB", (16, 16), "#554638").save(local_image)
            external_label = external / "outside.txt"
            external_label.write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
            label_link = self.dataset / "labels" / "train" / "label-link.txt"
            label_link.symlink_to(external_label)
            with self.assertRaises(ValueError):
                self.service.import_yolo("test-project")

    def test_path_containment_and_box_validation(self):
        with self.assertRaises(ValueError):
            self.service.safe_path(self.root.parent)
        sibling_images = self.root / "sibling-images"
        sibling_images.mkdir()
        with self.assertRaises(ValueError):
            self.service.register_project(
                "escape-project", "Escape", self.dataset, self.classes, image_dir="../sibling-images"
            )
        with self.assertRaises(ValueError):
            self.service.register_project(
                "gap-project", "Gap", self.dataset, {0: "object", 2: "gap"}
            )
        with self.assertRaises(ValueError):
            self.service.register_project(
                "test-project",
                "Changed taxonomy",
                self.dataset,
                {0: "renamed", 1: "dark_rock", 2: "metal_debris", 3: "shadow", 4: "background"},
            )
        with self.assertRaises(ValueError):
            AnnotationService(self.root, self.root.parent / "outside-annotations.db")
        with self.assertRaises(ValueError):
            BoxProposal(0, float("nan"), 0, 1, 1).validate(10, 10)
        with self.assertRaises(ValueError):
            BoxProposal(0, 5, 5, 4, 6).validate(10, 10)

    def test_versioned_api_health_and_revision_conflict(self):
        client = TestClient(create_app(self.service))
        health = client.get("/api/v1/health")
        self.assertEqual(health.status_code, 200)
        self.assertTrue(health.json()["ready"])
        self.assertIn("ultralytics_yolo", health.json()["adapters"])
        projects = client.get("/api/v1/projects")
        self.assertEqual(projects.status_code, 200)
        self.assertEqual(projects.json()[0]["id"], "test-project")
        item = self.service.list_images("test-project", limit=10)["items"][0]
        detail = self.service.get_image("test-project", item["id"])
        payload = {
            "expected_revision": detail["revision"],
            "status": "reviewed",
            "annotations": [{"class_id": 0, "x1": 2, "y1": 2, "x2": 20, "y2": 20}],
        }
        first = client.put(
            f"/api/v1/projects/test-project/images/{item['id']}/annotations", json=payload
        )
        self.assertEqual(first.status_code, 200)
        second = client.put(
            f"/api/v1/projects/test-project/images/{item['id']}/annotations", json=payload
        )
        self.assertEqual(second.status_code, 409)


class FutureSchemaTest(unittest.TestCase):
    def test_database_marked_current_but_missing_tables_is_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "corrupt.db"
            connection = sqlite3.connect(path)
            connection.execute("CREATE TABLE schema_meta(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            connection.execute(
                "INSERT INTO schema_meta VALUES('schema_version', ?)",
                (str(DATABASE_SCHEMA_VERSION),),
            )
            connection.commit()
            connection.close()
            with self.assertRaises(RuntimeError):
                AnnotationService(root, path)

    def test_v1_database_is_migrated_and_model_revision_is_backfilled(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "v1.db"
            connection = sqlite3.connect(path)
            connection.executescript(SCHEMA_V1)
            connection.execute(
                "INSERT OR REPLACE INTO schema_meta VALUES('schema_version', '1')"
            )
            connection.execute(
                """
                INSERT INTO model_registry(
                    id, name, adapter, weights_path, version, sha256, config_json,
                    active, created_at, updated_at
                ) VALUES('legacy-model', 'Legacy', 'test_fake', 'legacy.weights',
                         'abc123', 'abc123full', '{}', 1, 'now', 'now')
                """
            )
            connection.commit()
            connection.close()
            service = AnnotationService(root, path)
            self.assertEqual(service.database.schema_version(), DATABASE_SCHEMA_VERSION)
            model = service.get_model("legacy-model")
            self.assertTrue(model["revision_id"].startswith("legacy-model-legacy-"))
            revisions = service.list_model_revisions("legacy-model")
            self.assertEqual(revisions[0]["sha256"], "abc123full")

    def test_two_services_can_race_to_migrate_the_same_v1_database(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "shared-v1.db"
            connection = sqlite3.connect(path)
            connection.executescript(SCHEMA_V1)
            connection.execute(
                "INSERT OR REPLACE INTO schema_meta VALUES('schema_version', '1')"
            )
            connection.commit()
            connection.close()

            def open_service(_):
                return AnnotationService(root, path).database.schema_version()

            with ThreadPoolExecutor(max_workers=2) as executor:
                versions = list(executor.map(open_service, range(2)))
            self.assertEqual(versions, [DATABASE_SCHEMA_VERSION, DATABASE_SCHEMA_VERSION])

    def test_future_database_is_refused_before_schema_creation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "future.db"
            connection = sqlite3.connect(path)
            connection.execute("CREATE TABLE schema_meta(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            connection.execute("INSERT INTO schema_meta VALUES('schema_version', '999')")
            connection.commit()
            connection.close()
            with self.assertRaises(RuntimeError):
                AnnotationService(root, path)
            connection = sqlite3.connect(path)
            tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            connection.close()
            self.assertEqual(tables, {"schema_meta"})


if __name__ == "__main__":
    unittest.main()
