import torch
from surrogates import (
    BlipFeatureExtractor,
    ClipFeatureExtractor,
    EnsembleFeatureLoss,
    VisionTransformerFeatureExtractor,
)
from utils import get_list_image, save_list_images
from tqdm import tqdm
from attacks import SpectrumSimulationAttack, SSA_CommonWeakness
from torchvision import transforms
import os

images = get_list_image("./dataset/test-img")
resizer = transforms.Resize((512, 512))
images = [resizer(i).unsqueeze(0) for i in images]

blip = BlipFeatureExtractor().eval().cuda().requires_grad_(False)
clip = ClipFeatureExtractor().eval().cuda().requires_grad_(False)
vit = VisionTransformerFeatureExtractor().eval().cuda().requires_grad_(False)
models = [vit, blip, clip]

def ssa_cw_count_to_index(count, num_models=len(models), ssa_N=20):
    max_count = ssa_N * num_models
    count = count % max_count
    count = count // ssa_N
    return count

attack_strengths = {
    'weak': 8 / 255,
    'mild': 16 / 255,
    'medium': 32 / 255,
    'strong': 64 / 255,
    'very_strong': 128 / 255,
}

base_dir = "./attack_img_encoder_misdescription/"
if not os.path.exists(base_dir):
    os.mkdir(base_dir)

for strength_name, epsilon in attack_strengths.items():
    print(f"\n{'='*60}")
    print(f"Generating attacks with strength: {strength_name} (ε={epsilon:.4f})")
    print(f"{'='*60}\n")
    
    output_dir = os.path.join(base_dir, strength_name)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    ssa_cw_loss = EnsembleFeatureLoss(
        models, 
        ssa_cw_count_to_index, 
        feature_loss=torch.nn.MSELoss()
    )
    
    attacker = SSA_CommonWeakness(
        models,
        epsilon=epsilon,
        step_size=epsilon / 32,  # Adaptive step size: epsilon/32
        total_step=500,
        criterion=ssa_cw_loss,
    )
    
    img_id = 0
    for i, x in enumerate(tqdm(images, desc=f"Processing {strength_name}")):
        x = x.cuda()
        ssa_cw_loss.set_ground_truth(x)
        adv_x = attacker(x, None)
        save_list_images(adv_x, output_dir, begin_id=img_id)
        img_id += x.shape[0]
    
    print(f"Completed {strength_name}: saved {img_id} images to {output_dir}")

print(f"\n{'='*60}")
print("All attack strengths completed!")
print(f"{'='*60}")