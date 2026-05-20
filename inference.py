import torch
import os
from PIL import Image
from torchvision import transforms
from cubediff.pipelines.pipeline import CubeDiffPipeline, SceneSemanticPlan

if __name__ == "__main__":
    # ============== USER CONFIGURATION ==============
    # Modify these variables directly in the script
    
    # Image filename
    IMAGE_FILENAME = "All_Gizah_Pyramids.jpg"  # Change this to your image filename
    
    # Six face prompts in the order: [front, back, left, right, top, bottom]
    PROMPTS = [
        "front face, great pyramids of Giza, central subject",
        "back face, temples and columns, ancient ruins",
        "left face, Nile river and palm trees",
        "right face, the Sphinx and desert dunes",
        "top face, clear Egyptian sky with warm haze",
        "bottom face, sandy desert ground, rocky texture",
    ]

    # Optional semantic planning block. Leave as None to use the legacy prompt flow.
    SCENE_PLAN = SceneSemanticPlan.from_dict({
        "global_context": {
            "era": "Ancient Egypt, around 1300 BCE",
            "climate": "arid desert, cloudless sky",
            "time_of_day": "late afternoon, warm golden hour",
            "style": "photorealistic, highly detailed, realistic textures",
            "lighting_direction": "northwest, soft directional sunlight",
            "palette": "ochre, sand, gold, deep sky blue",
            "restrictions": "no people, no animals, no vehicles, no modern objects",
        },
        "face_artifacts": {
            "front": ["great pyramids of Giza", "desert foreground"],
            "back": ["temples", "columns", "ancient stone ruins"],
            "left": ["Nile river", "palm trees", "green riverbanks"],
            "right": ["the Sphinx", "desert dunes"],
            "top": ["clear Egyptian sky", "warm atmospheric haze"],
            "bottom": ["sandy desert ground", "rocky texture"],
        },
        "adjacency_relations": {
            "left": "The Nile flows toward the front face and shares the same horizon line.",
            "front": "The pyramids connect naturally with the Nile and keep the same warm sunset light.",
            "right": "The Sphinx belongs in the same desert basin and matches the front face lighting.",
            "back": "Keep ruins, columns, and stone color consistent with the front face palette.",
            "top": "Sky color temperature matches the horizon on the lateral faces.",
            "bottom": "Ground texture stays sandy and arid across the panorama.",
        },
        "verification_targets": {
            "front": ["pyramids", "desert", "warm golden light"],
            "back": ["temples", "columns"],
            "left": ["Nile river", "palm trees"],
            "right": ["Sphinx", "sand dunes"],
            "top": ["clear sky"],
            "bottom": ["desert ground"],
        },
    })

    # Optional hook for a future VQA / image-text verifier. Return True when the face is acceptable.
    FACE_VERIFIER = None
    
    # Model checkpoint path
    CHECKPOINT = "hlicai/cubediff-512-multitxt"  # Change this to one of the three types of checkpoints
    # "hlicai/cubediff-512-singlecaption"
    # "hlicai/cubediff-512-imgonly"
    # "hlicai/cubediff-512-multitxt"

    # Output directory
    OUTPUT_DIR = f"output/{IMAGE_FILENAME}_generations/"  # Change this to your desired output directory
    
    # Classifier-free guidance scale
    CFG_SCALE = 3.5
    
    # ================================================

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------------- Load Pipeline ----------------

    print(f"Loading pipeline from: {CHECKPOINT}")
    pipe = CubeDiffPipeline.from_pretrained(
        CHECKPOINT,
    )
    pipe = pipe.to(device)
    print(f"Pipeline loaded successfully and moved to {device}")

    image_size = pipe.vae.config.sample_size

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"\n[INFO] Loading image: {IMAGE_FILENAME}")
    image = Image.open(IMAGE_FILENAME).convert("RGB")
    conditioning_image = transform(image)

    print("[INFO] Running inference...")
    output = pipe(
        prompts=PROMPTS,
        conditioning_image=conditioning_image.unsqueeze(0).to(device),
        num_inference_steps=50,
        cfg_scale=CFG_SCALE,
        scene_plan=SCENE_PLAN,
        face_verifier=FACE_VERIFIER,
    )

    # ---------------- Save Results ----------------
    
    # Save the face images
    for face_name, face_img in zip(["front", "back", "left", "right", "top", "bottom"], output.faces_cropped):
        face_img = Image.fromarray(face_img)
        face_img.save(os.path.join(OUTPUT_DIR, f"{face_name}.png"))
        print(f"[INFO] Saved {face_name} face to {OUTPUT_DIR}/{face_name}.png")
    
    # Save the equirectangular image
    equirec_img = Image.fromarray(output.equirectangular)
    equirec_img.save(os.path.join(OUTPUT_DIR, "equirectangular.png"))
    print(f"[INFO] Saved equirectangular image to {OUTPUT_DIR}/equirectangular.png")

    print("\n[INFO] Inference completed successfully!")
