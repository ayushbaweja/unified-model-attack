# This script runs the Attack-Bard attack using uv for dependency management

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored messages
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ATTACK_BARD_DIR="$PROJECT_ROOT/attacks/Attack-Bard"
RESOURCES_DIR="$PROJECT_ROOT/resources"
DEFAULT_INPUT_DIR="$RESOURCES_DIR/test-images"
DEFAULT_OUTPUT_DIR="$PROJECT_ROOT/results/attack-bard"

# Parse arguments
INPUT_DIR="${1:-$DEFAULT_INPUT_DIR}"
OUTPUT_DIR="${2:-$DEFAULT_OUTPUT_DIR}"

print_info "Attack-Bard Attack Runner (via uv)"
print_info "=================================="
print_info ""
print_info "Input images directory: $INPUT_DIR"
print_info "Output directory: $OUTPUT_DIR"
print_info ""

# Check if input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    print_error "Input directory does not exist: $INPUT_DIR"
    print_info "Usage: $0 [input_dir] [output_dir]"
    print_info "Default input: $DEFAULT_INPUT_DIR"
    print_info "Default output: $DEFAULT_OUTPUT_DIR"
    exit 1
fi

# Check if there are images in the input directory
IMAGE_COUNT=$(find "$INPUT_DIR" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.JPG" -o -iname "*.JPEG" -o -iname "*.PNG" \) | wc -l)
if [ "$IMAGE_COUNT" -eq 0 ]; then
    print_error "No images found in $INPUT_DIR"
    print_info "Supported formats: .jpg, .jpeg, .png, .JPG, .JPEG, .PNG"
    exit 1
fi

print_info "Found $IMAGE_COUNT images in input directory"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Check for uv installation
if ! command -v uv &> /dev/null; then
    print_warning "uv is not installed."
    print_info "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Source cargo env if it was just installed there, or assume it's in path
    if [ -f "$HOME/.cargo/env" ]; then
        . "$HOME/.cargo/env"
    fi
fi

# Verify uv is now available
if ! command -v uv &> /dev/null; then
    print_error "Failed to install or locate uv. Please install it manually: https://docs.astral.sh/uv/"
    exit 1
fi

print_success "uv is ready."

# Create temporary attack script with custom paths AND inline metadata
TEMP_ATTACK_SCRIPT="$OUTPUT_DIR/attack_script_temp.py"

print_info "Preparing attack script with inline dependency metadata..."

cat > "$TEMP_ATTACK_SCRIPT" << 'EOFPYTHON'
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch",
#     "torchvision",
#     "torchaudio",
#     "transformers",
#     "accelerate",
#     "pillow",
#     "tqdm",
#     "scipy",
#     "numpy",
#     "markupsafe<2.1.0",
#     "opencv-python-headless",
# ]
# ///

import sys
import os

# Add Attack-Bard directory to Python path
# We receive this as the 3rd argument
if len(sys.argv) > 3:
    attack_bard_dir = sys.argv[3]
    sys.path.insert(0, attack_bard_dir)

import torch
# We wrap imports in try/except to give better errors if paths are wrong
try:
    from surrogates import (
        BlipFeatureExtractor,
        ClipFeatureExtractor,
        EnsembleFeatureLoss,
        VisionTransformerFeatureExtractor,
    )
    from utils import get_list_image, save_list_images
    from attacks import SpectrumSimulationAttack, SSA_CommonWeakness
except ImportError as e:
    print(f"\n[ERROR] Could not import Attack-Bard modules: {e}")
    print(f"Make sure {sys.argv[3]} points to the correct directory containing 'surrogates', 'utils', etc.")
    sys.exit(1)

from tqdm import tqdm
from torchvision import transforms

# Get paths from command line arguments
input_dir = sys.argv[1]
output_dir = sys.argv[2]

print(f"\n{'='*60}")
print(f"Attack-Bard: Image Encoder Misdescription Attack")
print(f"Python Version: {sys.version.split()[0]}")
print(f"{'='*60}\n")

# Load images
print("Loading images...")
try:
    images = get_list_image(input_dir)
except Exception as e:
    print(f"Error loading images: {e}")
    sys.exit(1)

print(f"Initial load count: {len(images)}")
if len(images) > 0:
    print(f"Sample type: {type(images[0])}")

resizer = transforms.Resize((512, 512))

# Process images (Handling Tensors directly)
processed_images = []
for img in images:
    # If it is a Tensor, we check dimensions. Standard image tensor is [C, H, W]
    if isinstance(img, torch.Tensor):
        # Handle Grayscale: [1, H, W] -> [3, H, W]
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        # Handle RGBA: [4, H, W] -> [3, H, W]
        elif img.shape[0] == 4:
            img = img[:3, :, :]
            
        # Resize works directly on tensors
        img = resizer(img)
        
        # Add batch dimension [1, 3, 512, 512]
        processed_images.append(img.unsqueeze(0))
    else:
        # Fallback if get_list_image somehow returns PIL images (unlikely given your error)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = resizer(img)
        processed_images.append(transforms.ToTensor()(img).unsqueeze(0))

images = processed_images
print(f"Processed {len(images)} images")

# Initialize surrogate models
print("\nInitializing surrogate models...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

print("- Loading BLIP model...")
blip = BlipFeatureExtractor().eval().to(device).requires_grad_(False)
print("- Loading CLIP model...")
clip = ClipFeatureExtractor().eval().to(device).requires_grad_(False)
print("- Loading Vision Transformer model...")
vit = VisionTransformerFeatureExtractor().eval().to(device).requires_grad_(False)
models = [vit, blip, clip]
print("Models initialized successfully")

def ssa_cw_count_to_index(count, num_models=len(models), ssa_N=20):
    max_count = ssa_N * num_models
    count = count % max_count
    count = count // ssa_N
    return count

# Attack strength configurations
attack_strengths = {
    'weak': 8 / 255,
    'mild': 16 / 255,
    'medium': 32 / 255,
    'strong': 64 / 255,
    'very_strong': 128 / 255,
}

# Create base output directory
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Run attacks at different strengths
for strength_name, epsilon in attack_strengths.items():
    print(f"\n{'='*60}")
    print(f"Generating attacks with strength: {strength_name} (ε={epsilon:.4f})")
    print(f"{'='*60}\n")

    strength_output_dir = os.path.join(output_dir, strength_name)
    if not os.path.exists(strength_output_dir):
        os.makedirs(strength_output_dir)

    ssa_cw_loss = EnsembleFeatureLoss(
        models,
        ssa_cw_count_to_index,
        feature_loss=torch.nn.MSELoss()
    )

    attacker = SSA_CommonWeakness(
        models,
        epsilon=epsilon,
        step_size=epsilon / 32,
        total_step=500,
        criterion=ssa_cw_loss,
    )

    img_id = 0
    for i, x in enumerate(tqdm(images, desc=f"Processing {strength_name}")):
        x = x.to(device)
        ssa_cw_loss.set_ground_truth(x)
        adv_x = attacker(x, None)
        # Assuming save_list_images handles tensor to image conversion and saving
        save_list_images(adv_x, strength_output_dir, begin_id=img_id)
        img_id += x.shape[0]

    print(f"Completed {strength_name}: saved {img_id} images to {strength_output_dir}")

print(f"\n{'='*60}")
print("All attack strengths completed!")
print(f"Output saved to: {output_dir}")
print(f"{'='*60}\n")
EOFPYTHON

print_success "Attack script prepared with inline dependencies."
print_info ""

print_info "Starting Attack-Bard attack..."
print_info "uv will now resolve dependencies and run the script (Python 3.10)..."
print_info ""

uv run --python 3.10 "$TEMP_ATTACK_SCRIPT" "$INPUT_DIR" "$OUTPUT_DIR" "$ATTACK_BARD_DIR"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    print_success "Attack completed successfully!"
    print_info ""
    print_info "Results saved to: $OUTPUT_DIR"
    
    # Clean up temporary script
    rm -f "$TEMP_ATTACK_SCRIPT"
else
    print_error "Attack failed with exit code $EXIT_CODE"
    # We keep the temp script for debugging if it failed
    print_warning "Temporary script preserved at: $TEMP_ATTACK_SCRIPT"
    exit $EXIT_CODE
fi