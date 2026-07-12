from __future__ import annotations

from pathlib import Path
from typing import Any

from ..models import BoxProposal
from .base import AdapterContext
from .registry import default_registry


class GroundingDinoAdapter:
    """Optional open-vocabulary adapter loaded only when transformers is installed.

    Model downloads are intentionally not initiated here. A local Hugging Face model
    directory must be registered as weights_path.
    """

    adapter_name = "grounding_dino"
    adapter_version = "1"

    def __init__(self, context: AdapterContext):
        self.context = context
        self._processor = None
        self._model = None

    def _load(self):
        model_path = Path(self.context.model.weights_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Local Grounding DINO model directory not found: {model_path}. "
                "Automatic downloads are disabled."
            )
        try:
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "Grounding DINO adapter requires the optional transformers dependency"
            ) from exc
        if self._processor is None:
            self._processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
            self._model = AutoModelForZeroShotObjectDetection.from_pretrained(
                model_path, local_files_only=True
            )
            self._model.to(self.context.device)
            self._model.eval()
        return self._processor, self._model

    def propose(
        self,
        image_path: Path,
        *,
        image_width: int,
        image_height: int,
        params: dict[str, Any],
    ) -> list[BoxProposal]:
        import torch
        from PIL import Image

        processor, model = self._load()
        prompts = params.get("prompts") or self.context.model.config.get("prompts")
        if not prompts:
            raise ValueError("Grounding DINO requires prompts with class_id and text")
        text = " . ".join(str(item["text"]).strip(" .") for item in prompts) + " ."
        threshold = float(params.get("threshold", 0.25))
        text_threshold = float(params.get("text_threshold", 0.2))
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, text=text, return_tensors="pt").to(self.context.device)
        with torch.no_grad():
            outputs = model(**inputs)
        result = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=threshold,
            text_threshold=text_threshold,
            target_sizes=[(image_height, image_width)],
        )[0]
        prompt_lookup = {str(item["text"]).lower(): int(item["class_id"]) for item in prompts}
        proposals: list[BoxProposal] = []
        labels = result.get("text_labels", result.get("labels", []))
        for box, score, label in zip(result["boxes"], result["scores"], labels, strict=True):
            phrase = str(label).lower()
            class_id = next(
                (value for key, value in prompt_lookup.items() if key in phrase or phrase in key),
                None,
            )
            if class_id is None:
                continue
            x1, y1, x2, y2 = (float(value) for value in box.tolist())
            proposal = BoxProposal(
                class_id=class_id,
                x1=max(0.0, x1),
                y1=max(0.0, y1),
                x2=min(float(image_width), x2),
                y2=min(float(image_height), y2),
                confidence=float(score),
                source="auto",
                model_id=self.context.model.id,
                model_revision_id=self.context.model.revision_id,
            )
            proposal.validate(image_width, image_height)
            proposals.append(proposal)
        return proposals


default_registry.register("grounding_dino", GroundingDinoAdapter)
