from __future__ import annotations

import hashlib
import threading
from pathlib import Path
from typing import Any

from ..models import BoxProposal
from .base import AdapterContext
from .registry import default_registry


class UltralyticsYoloAdapter:
    adapter_name = "ultralytics_yolo"
    adapter_version = "1"

    def __init__(self, context: AdapterContext):
        self.context = context
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        if self._model is None:
            weights = Path(self.context.model.weights_path)
            if not weights.is_file():
                raise FileNotFoundError(
                    f"Local YOLO weights not found: {weights}. Automatic downloads are disabled."
                )
            digest = hashlib.sha256()
            with weights.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != self.context.model.sha256:
                raise RuntimeError(
                    "Local YOLO weights changed after registration; register the new model revision first."
                )
            from ultralytics import YOLO

            self._model = YOLO(str(weights))
        return self._model

    def propose(
        self,
        image_path: Path,
        *,
        image_width: int,
        image_height: int,
        params: dict[str, Any],
    ) -> list[BoxProposal]:
        conf = float(params.get("conf", self.context.model.config.get("conf", 0.15)))
        iou = float(params.get("iou", self.context.model.config.get("iou", 0.45)))
        imgsz = int(params.get("imgsz", self.context.model.config.get("imgsz", 640)))
        max_det = int(params.get("max_det", self.context.model.config.get("max_det", 50)))
        max_area_ratio = float(params.get("max_area_ratio", 0.85))
        min_area_ratio = float(params.get("min_area_ratio", 0.00005))
        device = str(params.get("device", self.context.device))
        allowed_class_ids = {
            int(value)
            for value in params.get(
                "allowed_class_ids", self.context.model.config.get("allowed_class_ids", [0, 1, 2, 3])
            )
        }
        if not (0 <= conf <= 1 and 0 <= iou <= 1):
            raise ValueError("conf and iou must be between 0 and 1")
        if imgsz < 32 or imgsz > 4096 or max_det < 1 or max_det > 10000:
            raise ValueError("invalid imgsz or max_det")
        if not (0 <= min_area_ratio < max_area_ratio <= 1):
            raise ValueError("invalid area ratio thresholds")

        with self._lock:
            result = self._load().predict(
                source=str(image_path),
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                max_det=max_det,
                device=device,
                verbose=False,
                save=False,
            )[0]

        proposals: list[BoxProposal] = []
        if result.boxes is None:
            return proposals
        xyxy = result.boxes.xyxy.cpu().tolist()
        class_ids = result.boxes.cls.int().cpu().tolist()
        scores = result.boxes.conf.cpu().tolist()
        image_area = float(image_width * image_height)
        for coords, class_id, score in zip(xyxy, class_ids, scores, strict=True):
            if int(class_id) not in allowed_class_ids:
                continue
            x1, y1, x2, y2 = (float(value) for value in coords)
            x1, x2 = max(0.0, x1), min(float(image_width), x2)
            y1, y2 = max(0.0, y1), min(float(image_height), y2)
            if x2 <= x1 or y2 <= y1:
                continue
            ratio = ((x2 - x1) * (y2 - y1)) / image_area
            warning = None
            if ratio > max_area_ratio:
                warning = f"oversized_box:{ratio:.3f}"
            elif ratio < min_area_ratio:
                warning = f"tiny_box:{ratio:.6f}"
            proposal = BoxProposal(
                class_id=int(class_id),
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                confidence=float(score),
                source="auto",
                model_id=self.context.model.id,
                model_revision_id=self.context.model.revision_id,
                warning=warning,
            )
            proposal.validate(image_width, image_height)
            proposals.append(proposal)
        return proposals


default_registry.register("ultralytics_yolo", UltralyticsYoloAdapter)
