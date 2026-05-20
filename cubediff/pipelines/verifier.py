from typing import List, Optional, Sequence
import numpy as np
from PIL import Image


class CLIPVerifier:
    """Lightweight CLIP-based verifier. Lazily loads model to check if expected labels appear in an image.

    Usage: verifier = CLIPVerifier(model_name='openai/clip-vit-base-patch32', threshold=0.25)
           ok = verifier(face_name, face_img, expected_labels, scene_plan)
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", threshold: float = 0.25, device: Optional[str] = None):
        self.model_name = model_name
        self.threshold = float(threshold)
        self.device = device
        self._model = None
        self._processor = None

    def _ensure_model(self):
        if self._model is not None:
            return
        try:
            from transformers import CLIPProcessor, CLIPModel
            import torch
        except Exception as e:
            raise ImportError("Failed to import transformers.CLIP - install compatible transformers version") from e

        self._processor = CLIPProcessor.from_pretrained(self.model_name)
        self._model = CLIPModel.from_pretrained(self.model_name)
        if self.device is not None:
            try:
                self._model = self._model.to(self.device)
            except Exception:
                pass

    def __call__(self, face_name: str, face_img: np.ndarray, expected: Sequence[str], scene_plan=None) -> bool:
        """Return True if any expected label is detected with similarity above threshold.

        face_img: HxWxC uint8 numpy array
        expected: sequence of strings (labels)
        """
        if not expected:
            # Nothing to check, consider OK
            return True

        self._ensure_model()

        # Convert numpy to PIL
        if isinstance(face_img, np.ndarray):
            pil = Image.fromarray(face_img.astype('uint8'))
        else:
            pil = face_img

        # Prepare inputs
        text_inputs = list(expected)
        processor = self._processor

        inputs = processor(text=text_inputs, images=pil, return_tensors="pt", padding=True)
        device = self.device
        if device is not None:
            inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = self._model(**inputs)

        # Get image and text features
        image_features = outputs.image_embeds
        text_features = outputs.text_embeds

        import torch

        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity between image and each text embedding
        sims = (text_features @ image_features.T).squeeze(-1)
        max_sim = float(sims.max().item())

        return max_sim >= self.threshold
