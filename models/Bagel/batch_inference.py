"""
Batch inference script for BAGEL model.
Runs 5 prompts on 5 image pairs (clean/attacked) and saves outputs.

Directory structure expected:
    images/
        clean/
            1.png, 2.png, 3.png, 4.png, 5.png
        attacked/
            1.png, 2.png, 3.png, 4.png, 5.png

Output structure:
    outputs/
        prompt_1/
            clean_1.png
            attacked_1.png
        prompt_2/
            clean_2.png
            attacked_2.png
        ...
"""

import os
import argparse
import torch
import random
import numpy as np
from PIL import Image
from pathlib import Path

from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights
from accelerate.utils import BnbQuantizationConfig, load_and_quantize_model

from data.data_utils import add_special_tokens, pil_img2rgb
from data.transforms import ImageTransform
from inferencer import InterleaveInferencer
from modeling.autoencoder import load_ae
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM,
    SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer


# ============================================================================
# Configuration
# ============================================================================

PROMPTS = {
    1: "Render as a charcoal sketch drawing on textured paper.",
    2: "Change eye color to emerald green.",
    3: "Add a small red ball on the ground near the subject.",
    4: 'Add the text "Merry Christmas 2025" in large red and green font to the sign.',
    5: "The subject is now a hologram projection—50% transparent with light flickering.",
}

PROMPT_CATEGORIES = {
    1: "style",
    2: "detail_editing",
    3: "object_addition",
    4: "text",
    5: "semantic_edit",
}

# Each prompt maps to a specific image index (1-indexed)
PROMPT_TO_IMAGE = {
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
}


def set_seed(seed):
    """Set random seeds for reproducibility."""
    if seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_model(model_path: str, mode: int = 1):
    """Load the BAGEL model with specified quantization mode."""
    
    llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers -= 1

    vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))

    config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act='gelu_pytorch_tanh',
        latent_patch_size=2,
        max_latent_size=64,
    )

    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    vae_transform = ImageTransform(1024, 512, 16)
    vit_transform = ImageTransform(980, 224, 14)

    # Device mapping
    device_map = infer_auto_device_map(
        model,
        max_memory={i: "80GiB" for i in range(torch.cuda.device_count())},
        no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
    )

    same_device_modules = [
        'language_model.model.embed_tokens',
        'time_embedder',
        'latent_pos_embed',
        'vae2llm',
        'llm2vae',
        'connector',
        'vit_pos_embed'
    ]

    if torch.cuda.device_count() == 1:
        first_device = device_map.get(same_device_modules[0], "cuda:0")
        for k in same_device_modules:
            if k in device_map:
                device_map[k] = first_device
            else:
                device_map[k] = "cuda:0"
    else:
        first_device = device_map.get(same_device_modules[0])
        for k in same_device_modules:
            if k in device_map:
                device_map[k] = first_device

    # Load with specified mode
    if mode == 1:  # Full precision
        model = load_checkpoint_and_dispatch(
            model,
            checkpoint=os.path.join(model_path, "ema.safetensors"),
            device_map=device_map,
            offload_buffers=True,
            offload_folder="offload",
            dtype=torch.bfloat16,
            force_hooks=True,
        ).eval()
    elif mode == 2:  # NF4
        bnb_quantization_config = BnbQuantizationConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type="nf4"
        )
        model = load_and_quantize_model(
            model,
            weights_location=os.path.join(model_path, "ema.safetensors"),
            bnb_quantization_config=bnb_quantization_config,
            device_map=device_map,
            offload_folder="offload",
        ).eval()
    elif mode == 3:  # INT8
        bnb_quantization_config = BnbQuantizationConfig(
            load_in_8bit=True,
            torch_dtype=torch.float32
        )
        model = load_and_quantize_model(
            model,
            weights_location=os.path.join(model_path, "ema.safetensors"),
            bnb_quantization_config=bnb_quantization_config,
            device_map=device_map,
            offload_folder="offload",
        ).eval()
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Create inferencer
    inferencer = InterleaveInferencer(
        model=model,
        vae_model=vae_model,
        tokenizer=tokenizer,
        vae_transform=vae_transform,
        vit_transform=vit_transform,
        new_token_ids=new_token_ids,
    )

    return inferencer


def edit_image(inferencer, image: Image.Image, prompt: str, seed: int = 42,
               cfg_text_scale: float = 4.0, cfg_img_scale: float = 2.0,
               cfg_interval: float = 0.0, timestep_shift: float = 3.0,
               num_timesteps: int = 50, cfg_renorm_min: float = 0.0,
               cfg_renorm_type: str = "text_channel") -> Image.Image:
    """Run image editing inference."""
    
    set_seed(seed)
    
    image = pil_img2rgb(image)
    
    inference_hyper = dict(
        max_think_token_n=1024,
        do_sample=False,
        text_temperature=0.3,
        cfg_text_scale=cfg_text_scale,
        cfg_img_scale=cfg_img_scale,
        cfg_interval=[cfg_interval, 1.0],
        timestep_shift=timestep_shift,
        num_timesteps=num_timesteps,
        cfg_renorm_min=cfg_renorm_min,
        cfg_renorm_type=cfg_renorm_type,
    )
    
    result = inferencer(image=image, text=prompt, think=False, **inference_hyper)
    return result["image"]


def find_image(base_dir: str, image_idx: int) -> str:
    """Find image file with given index, supporting various extensions and naming patterns.
    
    Handles patterns like:
        - 1.png, 2.jpg
        - 1_skiing.png, 2_man.JPEG
    """
    extensions = ['.png', '.jpg', '.jpeg', '.webp', '.PNG', '.JPG', '.JPEG', '.WEBP']
    
    # List all files in directory
    try:
        files = os.listdir(base_dir)
    except FileNotFoundError:
        raise FileNotFoundError(f"Directory not found: {base_dir}")
    
    # Look for files starting with the index
    for f in files:
        # Check if filename starts with the index followed by underscore or dot
        if f.startswith(f"{image_idx}_") or f.startswith(f"{image_idx}."):
            # Verify it has a valid image extension
            if any(f.lower().endswith(ext.lower()) for ext in extensions):
                return os.path.join(base_dir, f)
    
    raise FileNotFoundError(f"No image found for index {image_idx} in {base_dir}")


def run_batch_inference(
    model_path: str,
    images_dir: str,
    output_dir: str,
    mode: int = 1,
    seed: int = 42,
):
    """Run batch inference on all prompt-image pairs."""
    
    print("=" * 60)
    print("BAGEL Batch Inference")
    print("=" * 60)
    
    # Setup paths
    clean_dir = os.path.join(images_dir, "clean")
    attacked_dir = os.path.join(images_dir, "attacked-bagelsiglip")
    
    # Validate directories exist
    for d in [clean_dir, attacked_dir]:
        if not os.path.exists(d):
            raise FileNotFoundError(f"Directory not found: {d}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    print(f"\nLoading model from: {model_path}")
    print(f"Quantization mode: {mode}")
    inferencer = load_model(model_path, mode)
    print("Model loaded successfully!\n")
    
    # Process each prompt
    for prompt_idx, prompt_text in PROMPTS.items():
        image_idx = PROMPT_TO_IMAGE[prompt_idx]
        category = PROMPT_CATEGORIES[prompt_idx]
        
        print(f"\n{'='*60}")
        print(f"Prompt {prompt_idx} ({category}): {prompt_text[:50]}...")
        print(f"Using image index: {image_idx}")
        print("=" * 60)
        
        # Create output subdirectory
        prompt_output_dir = os.path.join(output_dir, f"prompt_{prompt_idx}_{category}")
        os.makedirs(prompt_output_dir, exist_ok=True)
        
        # Process clean image
        try:
            clean_path = find_image(clean_dir, image_idx)
            print(f"\nProcessing clean image: {clean_path}")
            clean_image = Image.open(clean_path)
            
            clean_output = edit_image(inferencer, clean_image, prompt_text, seed=seed)
            clean_output_path = os.path.join(prompt_output_dir, f"clean_{image_idx}.png")
            clean_output.save(clean_output_path)
            print(f"Saved: {clean_output_path}")
            
        except Exception as e:
            print(f"Error processing clean image {image_idx}: {e}")
        
        # Process attacked image
        try:
            attacked_path = find_image(attacked_dir, image_idx)
            print(f"\nProcessing attacked image: {attacked_path}")
            attacked_image = Image.open(attacked_path)
            
            attacked_output = edit_image(inferencer, attacked_image, prompt_text, seed=seed)
            attacked_output_path = os.path.join(prompt_output_dir, f"attacked_{image_idx}.png")
            attacked_output.save(attacked_output_path)
            print(f"Saved: {attacked_output_path}")
            
        except Exception as e:
            print(f"Error processing attacked image {image_idx}: {e}")
    
    print("\n" + "=" * 60)
    print("Batch inference complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Batch inference for BAGEL model")
    parser.add_argument("--model_path", type=str, default="models/BAGEL-7B-MoT",
                        help="Path to BAGEL model")
    parser.add_argument("--images_dir", type=str, default="images",
                        help="Directory containing clean/ and attacked/ subdirectories")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Output directory for generated images")
    parser.add_argument("--mode", type=int, default=1, choices=[1, 2, 3],
                        help="Model loading mode: 1=full, 2=NF4, 3=INT8")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    run_batch_inference(
        model_path=args.model_path,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        mode=args.mode,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()