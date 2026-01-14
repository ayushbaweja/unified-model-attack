import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import argparse
import os
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass
from einops import rearrange
from torch import Tensor, nn
from safetensors.torch import load_file as load_sft
from typing import List


@dataclass
class AutoEncoderParams:
    resolution: int
    in_channels: int
    downsample: int
    ch: int
    out_ch: int
    ch_mult: List[int]
    num_res_blocks: int
    z_channels: int
    scale_factor: float
    shift_factor: float

def swish(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)

class AttnBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def attention(self, h_: Tensor) -> Tensor:
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
        b, c, h, w = q.shape
        q = rearrange(q, "b c h w -> b 1 (h w) c").contiguous()
        k = rearrange(k, "b c h w -> b 1 (h w) c").contiguous()
        v = rearrange(v, "b c h w -> b 1 (h w) c").contiguous()
        h_ = nn.functional.scaled_dot_product_attention(q, k, v)
        return rearrange(h_, "b 1 (h w) c -> b c h w", h=h, w=w, c=c, b=b)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.proj_out(self.attention(x))

class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = swish(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = swish(h)
        h = self.conv2(h)
        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        return x + h

class Downsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: Tensor):
        pad = (0, 1, 0, 1)
        x = nn.functional.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x

class Upsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor):
        x = nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
        return x

class Encoder(nn.Module):
    def __init__(self, resolution, in_channels, ch, ch_mult, num_res_blocks, z_channels):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.conv_in = nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)
        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        block_in = self.ch
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in)
                curr_res = curr_res // 2
            self.down.append(down)
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, 2 * z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))
        h = hs[-1]
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        return h

class Decoder(nn.Module):
    def __init__(self, ch, out_ch, ch_mult, num_res_blocks, in_channels, resolution, z_channels):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in)
                curr_res = curr_res * 2
            self.up.insert(0, up)
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z: Tensor) -> Tensor:
        h = self.conv_in(z)
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        return h

class DiagonalGaussian(nn.Module):
    def __init__(self, sample: bool = True, chunk_dim: int = 1):
        super().__init__()
        self.sample = sample
        self.chunk_dim = chunk_dim
    def forward(self, z: Tensor) -> Tensor:
        mean, logvar = torch.chunk(z, 2, dim=self.chunk_dim)
        if self.sample:
            std = torch.exp(0.5 * logvar)
            return mean + std * torch.randn_like(mean)
        else:
            return mean

class AutoEncoder(nn.Module):
    def __init__(self, params: AutoEncoderParams):
        super().__init__()
        self.encoder = Encoder(
            resolution=params.resolution, in_channels=params.in_channels,
            ch=params.ch, ch_mult=params.ch_mult, num_res_blocks=params.num_res_blocks,
            z_channels=params.z_channels,
        )
        self.decoder = Decoder(
            resolution=params.resolution, in_channels=params.in_channels,
            ch=params.ch, out_ch=params.out_ch, ch_mult=params.ch_mult,
            num_res_blocks=params.num_res_blocks, z_channels=params.z_channels,
        )
        self.reg = DiagonalGaussian()
        self.scale_factor = params.scale_factor
        self.shift_factor = params.shift_factor

    def encode(self, x: Tensor) -> Tensor:
        z = self.reg(self.encoder(x))
        z = self.scale_factor * (z - self.shift_factor)
        return z

    def decode(self, z: Tensor) -> Tensor:
        z = z / self.scale_factor + self.shift_factor
        return self.decoder(z)

    def forward(self, x: Tensor) -> Tensor:
        return self.decode(self.encode(x))

def load_ae(local_path: str) -> AutoEncoder:
    ae_params = AutoEncoderParams(
            resolution=256, in_channels=3, downsample=8, ch=128, out_ch=3,
            ch_mult=[1, 2, 4, 4], num_res_blocks=2, z_channels=16,
            scale_factor=0.3611, shift_factor=0.1159,
    )
    ae = AutoEncoder(ae_params)
    if local_path is not None:
        sd = load_sft(local_path)
        ae.load_state_dict(sd, strict=False)
        print("Model loaded successfully.")
    return ae, ae_params


class ImageTransform:
    def __init__(self, max_image_size=512, min_image_size=256, image_stride=32):
        self.max_image_size = max_image_size
        self.min_image_size = min_image_size
        self.image_stride = image_stride

    def __call__(self, pil_image):
        import numpy as np
        import torch
        # Resize to 256x256 for standard VAE input
        img = pil_image.resize((256, 256)) 
        x = np.array(img).astype(np.float32) / 127.5 - 1.0
        return torch.from_numpy(x).permute(2, 0, 1) # [C, H, W]

def pgd_attack_vae(
    model: torch.nn.Module, 
    images: torch.Tensor, 
    epsilon: float, 
    alpha: float, 
    num_iter: int,
    device: str
) -> torch.Tensor:
    model.eval()
    images = images.to(device)
    
    # 1. Random Start
    delta = torch.zeros_like(images).uniform_(-epsilon, epsilon)
    delta = torch.clamp(images + delta, -1, 1) - images
    delta.requires_grad = True

    for i in range(num_iter):
        adv_images = images + delta
        
        # 2. Forward
        reconstructions = model(adv_images)
        
        # 3. Maximize MSE
        loss = F.mse_loss(reconstructions, images, reduction='sum')
        
        # 4. Update
        grad = torch.autograd.grad(loss, delta)[0]
        with torch.no_grad():
            delta = delta + alpha * grad.sign()
            delta = torch.clamp(delta, -epsilon, epsilon)
            delta = torch.clamp(images + delta, -1, 1) - images
            delta.requires_grad = True # Reset for next iter

    return (images + delta).detach()

def process_batch(args, model, transform, image_files, device):
    for img_path in tqdm(image_files, desc="Attacking images"):
        try:
            pil_img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Skipping {img_path}: {e}")
            continue

        img_tensor = transform(pil_img).unsqueeze(0).to(device)

        adv_tensor = pgd_attack_vae(
            model=model,
            images=img_tensor,
            epsilon=args.epsilon,
            alpha=args.alpha,
            num_iter=args.iter,
            device=device
        )
        
        save_name = img_path.stem + f"_eps{args.epsilon}_adv.png"
        save_path = Path(args.output_dir) / save_name
        
        adv_img_np = (adv_tensor * 0.5 + 0.5).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
        adv_img_np = (adv_img_np * 255).astype(np.uint8)[0]
        
        Image.fromarray(adv_img_np).save(save_path)

def main():
    parser = argparse.ArgumentParser(description="PGD Attack on Bagel VAE")
    
    default_input = "/ssdscratch/abaweja7/unified-model-attack/attacks/FOA-Attack/resources/images/bigscale/test"
    default_model = "/ssdscratch/abaweja7/unified-model-attack/models/Bagel/models/BAGEL-7B-MoT/ae.safetensors"

    parser.add_argument("--input_dir", type=str, default=default_input, help="Path to directory containing images")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save adversarial images")
    parser.add_argument("--model_path", type=str, default=default_model, help="Path to model checkpoint")
    
    parser.add_argument("--epsilon", type=float, default=0.03, help="Perturbation magnitude")
    parser.add_argument("--alpha", type=float, default=0.01, help="Step size")
    parser.add_argument("--iter", type=int, default=40, help="Number of PGD iterations")
    
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}...")

    print(f"Loading model from {args.model_path}...")
    ae, _ = load_ae(args.model_path) 
    ae.to(device)
    ae.eval()

    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    image_files = list(input_path.glob("*.jpg")) + list(input_path.glob("*.png")) + list(input_path.glob("*.jpeg")) + list(input_path.glob("*.JPEG")) + list(input_path.glob("*.JPG"))
    
    if not image_files:
        print(f"No images found in {input_path}")
        return

    transform = ImageTransform()
    process_batch(args, ae, transform, image_files, device)
    print(f"Done. Results in {output_path}")

if __name__ == "__main__":
    main()