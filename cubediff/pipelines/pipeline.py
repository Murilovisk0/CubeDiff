from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union
import torch
import numpy as np

from diffusers import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion.pipeline_output import BaseOutput
from ..modules.extra_channels import make_extra_channels_tensor
from ..modules.utils import patch_groupnorm, patch_unet, swap_transformer_blocks
from .postprocessing import postprocess_outputs


FACE_ORDER = ["front", "back", "left", "right", "top", "bottom"]


@dataclass
class SceneSemanticPlan:
    global_context: Dict[str, str] = field(default_factory=dict)
    face_artifacts: Dict[str, List[str]] = field(default_factory=dict)
    adjacency_relations: Dict[str, str] = field(default_factory=dict)
    face_overrides: Dict[str, str] = field(default_factory=dict)
    verification_targets: Dict[str, List[str]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SceneSemanticPlan":
        def _normalize_text_map(source: Any) -> Dict[str, str]:
            if not isinstance(source, Mapping):
                return {}
            return {str(key): str(value) for key, value in source.items()}

        def _normalize_list_map(source: Any) -> Dict[str, List[str]]:
            if not isinstance(source, Mapping):
                return {}

            normalized: Dict[str, List[str]] = {}
            for key, value in source.items():
                if value is None:
                    normalized[str(key)] = []
                elif isinstance(value, (list, tuple, set)):
                    normalized[str(key)] = [str(item) for item in value]
                else:
                    normalized[str(key)] = [str(value)]
            return normalized

        return cls(
            global_context=_normalize_text_map(data.get("global_context", {})),
            face_artifacts=_normalize_list_map(data.get("face_artifacts", {})),
            adjacency_relations=_normalize_text_map(data.get("adjacency_relations", {})),
            face_overrides=_normalize_text_map(data.get("face_overrides", {})),
            verification_targets=_normalize_list_map(data.get("verification_targets", {})),
        )


def _normalize_prompts(
    prompts: Optional[Union[str, Sequence[str], Mapping[str, str]]],
) -> List[str]:
    if prompts is None:
        return [""] * len(FACE_ORDER)
    if isinstance(prompts, str):
        return [prompts] * len(FACE_ORDER)
    if isinstance(prompts, Mapping):
        return [str(prompts.get(face, "")) for face in FACE_ORDER]

    prompt_list = list(prompts)
    if len(prompt_list) != len(FACE_ORDER):
        raise ValueError(f"Expected 6 prompts, got {len(prompt_list)}")
    return [str(prompt) for prompt in prompt_list]


def _format_context_block(title: str, values: Mapping[str, str]) -> str:
    if not values:
        return ""

    lines = [f"{title}:"]
    for key, value in values.items():
        lines.append(f"- {key.replace('_', ' ')}: {value}")
    return "\n".join(lines)


def _format_list_block(title: str, items: Sequence[str]) -> str:
    if not items:
        return ""
    return f"{title}: " + ", ".join(items)


def _compose_face_prompt(
    face_name: str,
    base_prompt: str,
    scene_plan: Optional[SceneSemanticPlan],
    verification_feedback: Optional[str] = None,
) -> str:
    if scene_plan is None:
        return base_prompt

    sections: List[str] = []

    context_block = _format_context_block("Global scene context", scene_plan.global_context)
    if context_block:
        sections.append(context_block)

    if base_prompt.strip():
        sections.append(f"Base prompt: {base_prompt.strip()}")

    face_override = scene_plan.face_overrides.get(face_name, "").strip()
    if face_override:
        sections.append(f"Face override: {face_override}")

    artifacts = scene_plan.face_artifacts.get(face_name, [])
    artifact_block = _format_list_block("Primary artifacts", artifacts)
    if artifact_block:
        sections.append(artifact_block)

    adjacency = scene_plan.adjacency_relations.get(face_name, "").strip()
    if adjacency:
        sections.append(f"Spatial continuity: {adjacency}")

    verification_targets = scene_plan.verification_targets.get(face_name, [])
    verification_block = _format_list_block("Verification targets", verification_targets)
    if verification_block:
        sections.append(verification_block)

    if verification_feedback:
        sections.append(f"Verification feedback: {verification_feedback.strip()}")

    return "\n".join(section for section in sections if section)


def _resolve_face_prompts(
    prompts: Optional[Union[str, Sequence[str], Mapping[str, str]]],
    scene_plan: Optional[SceneSemanticPlan],
    verification_feedback: Optional[Mapping[str, str]] = None,
) -> List[str]:
    base_prompts = _normalize_prompts(prompts)
    resolved_prompts: List[str] = []

    for face_name, base_prompt in zip(FACE_ORDER, base_prompts):
        resolved_prompts.append(
            _compose_face_prompt(
                face_name,
                base_prompt,
                scene_plan,
                verification_feedback.get(face_name) if verification_feedback else None,
            )
        )

    return resolved_prompts


@dataclass
class CubeDiffPipelineOutput(BaseOutput):
    faces: np.ndarray
    faces_cropped: np.ndarray
    equirectangular: np.ndarray


class CubeDiffPipeline(StableDiffusionPipeline):

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """
        Load CubeDiffPipeline from pretrained model and automatically apply CubeDiff patches.
        
        The pretrained model should already have the correct input conv layer (7 channels),
        but we still need to patch the attention mechanisms and group norms.
        """

        # Load the base pipeline
        pipeline = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        
        if pipeline.unet.config.in_channels != 7:
            # Is a base SD model, patch input conv as well
            patch_unet(pipeline.unet, in_channels=7)
        else:
            # Apply attention patches (swap BasicTransformerBlock -> CubeDiffTransformerBlock)
            swap_transformer_blocks(pipeline.unet)
        
        # Apply groupnorm patches (GroupNorm -> CubeDiffGroupNorm)
        patch_groupnorm(pipeline.vae)
    
        return pipeline

    @torch.no_grad()
    def __call__(
        self,
        prompts: Optional[Union[str, List[str], Mapping[str, str]]] = None,
        *,
        conditioning_image: torch.Tensor,  # (C,H,W)
        num_inference_steps: int = 50,
        generator: Optional[torch.Generator] = None,
        cfg_scale: float = 3.5,
        scene_plan: Optional[Union[SceneSemanticPlan, Mapping[str, Any]]] = None,
        face_verifier: Optional[
            Callable[[str, np.ndarray, Sequence[str], Optional[SceneSemanticPlan]], bool]
        ] = None,
        max_verification_rounds: int = 1,
    ):
        device = self._execution_device
        T = len(FACE_ORDER)

        if scene_plan is not None and not isinstance(scene_plan, SceneSemanticPlan):
            scene_plan = SceneSemanticPlan.from_dict(scene_plan)

        if isinstance(prompts, str):
            prompts = [prompts] * T

        def run_generation(current_prompts: Sequence[str]):
            text_inputs = self.tokenizer(
                current_prompts,
                max_length=self.tokenizer.model_max_length,
                padding="max_length",
                return_tensors="pt",
            )
            encoder_hidden_states = self.text_encoder(text_inputs.input_ids.to(device))[0]

            uncond_inputs = self.tokenizer(
                [""] * T,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            )
            uncond_embeddings = self.text_encoder(uncond_inputs.input_ids.to(device))[0]

            self.scheduler.set_timesteps(num_inference_steps, device=device)
            latents = torch.randn(
                (T, 4, self.unet.config.sample_size, self.unet.config.sample_size),
                generator=generator,
                device=device,
                dtype=self.unet.dtype,
            )
            latents *= self.scheduler.init_noise_sigma

            static_extra = make_extra_channels_tensor(
                1, self.unet.config.sample_size, self.unet.config.sample_size
            ).to(device, dtype=self.unet.dtype)

            conditioning_image_local = conditioning_image
            if conditioning_image_local.ndim == 3:
                conditioning_image_local = conditioning_image_local.unsqueeze(0)
            conditioning_image_local = conditioning_image_local.to(device, dtype=self.unet.dtype)
            ref_lat = self.vae.encode(conditioning_image_local).latent_dist.mean[0]
            ref_lat *= self.vae.config.scaling_factor

            for t in self.scheduler.timesteps:
                latents[0] = ref_lat
                latents_scaled = self.scheduler.scale_model_input(latents, t)
                latents_input = torch.cat([latents_scaled, static_extra], dim=1)

                noise_pred = self.unet(latents_input, t, encoder_hidden_states=encoder_hidden_states).sample
                noise_pred_uncond = self.unet(
                    latents_input,
                    t,
                    encoder_hidden_states=uncond_embeddings,
                    cross_attention_kwargs={"front_face_drop": True},
                ).sample

                combined = noise_pred_uncond + cfg_scale * (noise_pred - noise_pred_uncond)
                latents[1:] = self.scheduler.step(combined[1:], t, latents[1:]).prev_sample

            imgs = self.vae.decode(latents / self.vae.config.scaling_factor).sample
            imgs = (imgs / 2 + 0.5).clamp(0, 1)
            return postprocess_outputs(imgs)

        prompt_feedback: Dict[str, str] = {}
        max_rounds = max(1, int(max_verification_rounds) + 1)
        equirec = uncropped = cropped = None

        for attempt in range(max_rounds):
            effective_prompts = _resolve_face_prompts(prompts, scene_plan, prompt_feedback or None)
            equirec, uncropped, cropped = run_generation(effective_prompts)

            if face_verifier is None or scene_plan is None or attempt == max_rounds - 1:
                break

            failed_faces: Dict[str, str] = {}
            for face_name, face_img in zip(FACE_ORDER, cropped):
                expected = scene_plan.verification_targets.get(face_name) or scene_plan.face_artifacts.get(face_name, [])
                if face_verifier(face_name, face_img, expected, scene_plan):
                    continue

                if expected:
                    failed_faces[face_name] = (
                        f"Ensure {face_name} clearly contains: {', '.join(expected)}. "
                        "Keep the global lighting, palette and style stable across all faces."
                    )
                else:
                    failed_faces[face_name] = (
                        f"Improve semantic fidelity for the {face_name} face while preserving the same scene-wide style and lighting."
                    )

            if not failed_faces:
                break

            prompt_feedback.update(failed_faces)

        return CubeDiffPipelineOutput(
            faces=uncropped,
            faces_cropped=cropped,
            equirectangular=equirec,
        )
