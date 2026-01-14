import torch
import torch.nn as nn
from torchvision import transforms
from transformers import AutoConfig
from safetensors.torch import load_file
import glob
import os
import gc
import sys
import json

# Get the directory where this script is located (surrogates/FeatureExtractors)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up to unified-model-attack root, then into models/Bagel
bagel_path = os.path.abspath(os.path.join(current_dir, "../../../../models/Bagel"))

if bagel_path not in sys.path:
    print(f"[INFO] Adding Bagel path to sys.path: {bagel_path}")
    sys.path.append(bagel_path)

from modeling.bagel.siglip_navit import SiglipVisionModel, SiglipVisionConfig
from data.data_utils import get_flattened_position_ids_extrapolate
from .Base import BaseFeatureExtractor

class BagelSiglipFeatureExtractor(BaseFeatureExtractor):
    def __init__(self, bagel_checkpoint_path=None, device='cuda'):
        if bagel_checkpoint_path is None:
            # Default to shared Bagel model in models folder
            bagel_checkpoint_path = os.path.abspath(os.path.join(current_dir, "../../../../models/Bagel/models/BAGEL-7B-MoT"))
        super(BagelSiglipFeatureExtractor, self).__init__()
        self.device = device
        
        # Ensure path is absolute
        if not os.path.isabs(bagel_checkpoint_path):
            # Assume relative to FOA-Attack root
            root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir))) 
            potential_path = os.path.join(root_dir, bagel_checkpoint_path)
            if os.path.exists(potential_path):
                bagel_checkpoint_path = potential_path
        
        print(f"[INFO] Initializing Bagel SigLIP from: {bagel_checkpoint_path}")

        # 1. Load Config
        try:
            full_config = AutoConfig.from_pretrained(bagel_checkpoint_path, trust_remote_code=True)
            self.config = full_config.vit_config
        except Exception as e:
            print(f"[WARN] AutoConfig failed. Attempting manual config load.")
            
            # Use SigLIP SO400M as base (matches hidden size 1152)
            self.config = SiglipVisionConfig.from_pretrained("google/siglip-so400m-patch14-384")

            # Default SO400M has 27 layers. We must match the checkpoint.
            self.config.num_hidden_layers = 26 
            
            self.config.image_size = 224
            self.config.rope = True

        self.patch_size = self.config.patch_size
        
        # 2. Initialize Model Structure
        self.model = SiglipVisionModel(config=self.config)
        
        # 3. Convert Conv2d to Linear (Bagel Requirement)
        self.model.vision_model.embeddings.convert_conv2d_to_linear(self.config)

        # 4. Load Weights from Bagel Checkpoint
        self._load_bagel_weights(bagel_checkpoint_path)

        # 5. Keep model in bfloat16 (required for flash_attn compatibility)
        self.model.to(self.device).bfloat16()
        self.model.eval()

        # 5. Define Normalization
        self.normalizer = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.Lambda(lambda img: torch.clamp(img, 0.0, 255.0) / 255.0), 
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def _load_bagel_weights(self, checkpoint_path):
        print("[INFO] Loading vision weights from Bagel checkpoint...")
        
        files = glob.glob(os.path.join(checkpoint_path, "*.safetensors"))
        if not files:
            files = glob.glob(os.path.join(checkpoint_path, "*.bin"))
            
        if not files:
            raise FileNotFoundError(f"No checkpoint files found in {checkpoint_path}")

        vision_state_dict = {}
        prefix = "vit_model." 

        for file in files:
            if file.endswith(".safetensors"):
                state_dict = load_file(file)
            else:
                state_dict = torch.load(file, map_location="cpu")
            
            keys_to_load = {k: v for k, v in state_dict.items() if k.startswith(prefix)}
            
            for k, v in keys_to_load.items():
                new_key = k[len(prefix):]
                vision_state_dict[new_key] = v
            
            del state_dict
            gc.collect()

        if len(vision_state_dict) == 0:
            raise RuntimeError("Found no 'vit_model' weights in checkpoint.")

        missing, unexpected = self.model.load_state_dict(vision_state_dict, strict=False)
        real_missing = [k for k in missing if "rope" not in k]
        
        if len(real_missing) > 0:
            print(f"[WARN] Missing keys: {real_missing[:5]} ...")
        else:
            print("[SUCCESS] All Bagel vision weights loaded successfully.")

    def forward(self, x):
        # Forward is often bypassed by EnsembleFeatureExtractor, but keeping it for compatibility
        return self.global_local_features(x)[0]

    def global_local_features(self, x):
        # Force print to check execution
        if not hasattr(self, "_has_printed"):
            print(f"[DEBUG] BagelSiglip is processing a batch of shape {x.shape}")
            self._has_printed = True

        x = x.to(self.device)
        x_norm = self.normalizer(x)

        # Convert to bfloat16 for flash_attn compatibility
        x_norm = x_norm.bfloat16()

        B, C, H, W = x_norm.shape

        p = self.patch_size
        h_patches = H // p
        w_patches = W // p
        n_patches = h_patches * w_patches

        x_patches = x_norm.view(B, C, h_patches, p, w_patches, p)
        x_patches = torch.einsum("bchpwq->bhwpqc", x_patches)
        packed_pixel_values = x_patches.reshape(-1, p * p * C)

        single_img_pos_ids = get_flattened_position_ids_extrapolate(
            H, W, p, max_num_patches_per_side=self.config.image_size // p
        ).to(self.device)

        packed_flattened_position_ids = single_img_pos_ids.repeat(B)

        cu_seqlens = torch.arange(0, (B + 1) * n_patches, n_patches, dtype=torch.int32, device=self.device)
        max_seqlen = n_patches

        hidden_states = self.model(
            packed_pixel_values=packed_pixel_values,
            packed_flattened_position_ids=packed_flattened_position_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen
        )

        # Convert back to float32 for compatibility with other models
        local_features = hidden_states.float().view(B, n_patches, -1)
        local_features = local_features / local_features.norm(dim=-1, keepdim=True)

        global_feature = local_features.mean(dim=1)
        global_feature = global_feature / global_feature.norm(dim=-1, keepdim=True)

        return global_feature, local_features