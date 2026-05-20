import os
import os.path as osp
import json
import argparse
from PIL import Image

import torch

from DreamCube.app import build_pipeline, inference, postprocess_rgb, postprocess_depth


VIEW_KEYS = ["front", "right", "back", "left", "top", "bottom"]


def compose_prompts_from_plan(plan: dict) -> list:
    gc = plan.get("global_context", {})
    face_artifacts = plan.get("face_artifacts", {})

    def join_artifacts(key: str):
        items = face_artifacts.get(key, [])
        if isinstance(items, list):
            return ", ".join([str(i) for i in items if i])
        return str(items)

    prompts = []
    for k in VIEW_KEYS:
        parts = []
        # global era/style/lighting/mood if present
        for g in ("era", "style", "lighting", "mood", "atmosphere"):
            v = gc.get(g)
            if v:
                parts.append(str(v))

        # artifacts specific to this face
        artifacts = join_artifacts(k)
        if artifacts:
            parts.append(artifacts)

        # verification target or other hints
        vt = plan.get("verification_targets", {}).get(k)
        if vt:
            parts.append(str(vt))

        prompt = ". ".join([p.strip() for p in parts if p])
        if not prompt:
            prompt = "A detailed, photorealistic view"
        prompts.append(prompt)

    return prompts


def run(
    image_path: str,
    depth_path: str,
    plan_path: str,
    outdir: str,
    ckpt: str = "KevinHuang/DreamCube",
    device: str = None,
    height: int = 512,
    width: int = 512,
    steps: int = 50,
    guidance: float = 7.5,
):
    os.makedirs(outdir, exist_ok=True)

    with open(plan_path, 'r', encoding='utf-8') as f:
        plan = json.load(f)

    prompts = compose_prompts_from_plan(plan)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    print("Building DreamCube pipeline (this may download weights)...")
    pipe = build_pipeline(ckpt, device=device, local_files_only=False)
    print("Pipeline ready. Running inference...")

    preds = inference(
        pipe=pipe,
        image=image_path,
        depth=depth_path,
        prompts=prompts,
        height=height,
        width=width,
        output_type='np',
        num_inference_steps=steps,
        guidance_scale=guidance,
    )

    images_pred = preds.get('images')
    depths_pred = preds.get('depths')

    if images_pred is not None:
        post_rgb = postprocess_rgb(images_pred)
        equi = Image.fromarray(post_rgb['equi'][0])
        dice = Image.fromarray(post_rgb['dice'][0])
        equi.save(osp.join(outdir, 'output_equi_rgb.png'))
        dice.save(osp.join(outdir, 'output_dice_rgb.png'))

    if depths_pred is not None:
        post_depth = postprocess_depth(depths_pred)
        equi_depth_raw = Image.fromarray(post_depth['equi_depth_raw'][0])
        equi_depth_vis = Image.fromarray(post_depth['equi_depth_vis'][0])
        equi_depth_raw.save(osp.join(outdir, 'output_equi_depth_raw.png'))
        equi_depth_vis.save(osp.join(outdir, 'output_equi_depth_vis.png'))

    print(f"Saved outputs to {outdir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True, help='Input RGB image (front view)')
    parser.add_argument('--depth', required=True, help='Input depth image (front view)')
    parser.add_argument('--plan', required=True, help='JSON file with SceneSemanticPlan (global_context + face_artifacts)')
    parser.add_argument('--outdir', default='./outputs/dreamcube_from_plan', help='Output directory')
    parser.add_argument('--ckpt', default='KevinHuang/DreamCube', help='DreamCube checkpoint or HF repo')
    parser.add_argument('--device', default=None, help='Torch device string (e.g. cuda:0)')
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--guidance', type=float, default=7.5)

    args = parser.parse_args()

    run(
        image_path=args.image,
        depth_path=args.depth,
        plan_path=args.plan,
        outdir=args.outdir,
        ckpt=args.ckpt,
        device=args.device,
        height=args.height,
        width=args.width,
        steps=args.steps,
        guidance=args.guidance,
    )


if __name__ == '__main__':
    main()
