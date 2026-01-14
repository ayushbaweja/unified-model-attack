import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import sys

# Add path to find modules
sys.path.append(os.getcwd())
from surrogates.FeatureExtractors.BagelSiglip import BagelSiglipFeatureExtractor

def load_image_as_tensor(path, device):
    """Loads image and returns it as a [0, 255] tensor, matching attack input."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    img = Image.open(path).convert('RGB')
    # Convert to tensor but keep [0, 255] range to match BagelSiglip expectation
    img_t = transforms.ToTensor()(img) * 255.0 
    return img_t.unsqueeze(0).to(device)

def main():
    # --- UPDATED DEVICE ---
    device = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
    print(f"Using device: {device}")

    # 1. Load the Bagel Evaluator
    print("Loading Bagel Model for evaluation...")
    # Ensure this path matches your setup inside FOA-Attack folder
    bagel_path = "Bagel-main/models/BAGEL-7B-MoT" 
    
    try:
        model = BagelSiglipFeatureExtractor(bagel_checkpoint_path=bagel_path, device=device)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    
    # 2. Setup Paths (Update these to your specific output files)
    target_img_path = "/ssdscratch/abaweja7/FOA-Attack/resources/images/target_images/1/4.jpg"
    
    # Path to an image generated WITHOUT Bagel (Standard Attack)
    adv_standard_path = "/ssdscratch/abaweja7/FOA-Attack/foa_attack_output/img/b41f202919935424a9a3a8c16e0fb36b/test/5_woman.png"
    
    # Path to an image generated WITH Bagel (Modified Attack)
    adv_bagel_path = "/ssdscratch/abaweja7/FOA-Attack/foa_attack_output_bageladded/img/e3c7b3d6305bc6b8122c444e8bdfb283/test/5_woman.png"

    # 3. Calculate Features
    print("Computing features...")
    with torch.no_grad():
        # --- FIX: Removed unpacking (, _) ---
        try:
            print("Encoding Target...")
            tgt_feat = model(load_image_as_tensor(target_img_path, device))
            
            print("Encoding Standard Attack...")
            std_feat = model(load_image_as_tensor(adv_standard_path, device))
            
            print("Encoding Bagel Attack...")
            bagel_feat = model(load_image_as_tensor(adv_bagel_path, device))
        except FileNotFoundError as e:
            print(e)
            return

    # 4. Calculate Cosine Similarity
    sim_standard = F.cosine_similarity(std_feat, tgt_feat).item()
    sim_bagel = F.cosine_similarity(bagel_feat, tgt_feat).item()

    print("\nResults (Higher is better/more targeted):")
    print(f"Similarity (Standard Attack): {sim_standard:.4f}")
    print(f"Similarity (Bagel Attack):    {sim_bagel:.4f}")
    
    if sim_bagel > sim_standard:
        print(f"\n SUCCESS: Bagel encoder improved alignment by {sim_bagel - sim_standard:.4f}")
    else:
        print("\n FAIL: Bagel encoder did not improve alignment.")

if __name__ == "__main__":
    main()