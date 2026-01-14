#!/bin/bash
# This script runs the FOA-Attack (Feature Optimal Alignment Attack)

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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
FOA_ATTACK_DIR="$PROJECT_ROOT/attacks/FOA-Attack"
RESOURCES_DIR="$PROJECT_ROOT/resources"
DEFAULT_INPUT_DIR="$RESOURCES_DIR/test-images"
DEFAULT_OUTPUT_DIR="$PROJECT_ROOT/results/foa-attack"

# Parse arguments
INPUT_DIR="${1:-$DEFAULT_INPUT_DIR}"
OUTPUT_DIR="${2:-$DEFAULT_OUTPUT_DIR}"

print_info "FOA-Attack Runner"
print_info "================="
print_info ""
print_info "Clean images (input): $INPUT_DIR"
print_info "Target images: $FOA_ATTACK_DIR/resources/images/target_images"
print_info "Output directory: $OUTPUT_DIR"
print_info ""

# Check if input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    print_error "Input directory does not exist: $INPUT_DIR"
    print_info "Usage: $0 [input_dir] [output_dir]"
    exit 1
fi

# Check if there are images in the input directory
IMAGE_COUNT=$(find "$INPUT_DIR" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | wc -l)
if [ "$IMAGE_COUNT" -eq 0 ]; then
    print_error "No images found in $INPUT_DIR"
    exit 1
fi

print_info "Found $IMAGE_COUNT images in input directory"

# Create output directory and required subdirectories
mkdir -p "$OUTPUT_DIR"

# Set up the data directories for FOA-Attack
# Clean images = input images to be perturbed
# Target images = existing target images in FOA-Attack (used for feature alignment)
CLEAN_DIR="$FOA_ATTACK_DIR/resources/images/bigscale/test"

# Create clean directory if it doesn't exist
mkdir -p "$CLEAN_DIR"

# Copy input images to clean directory only
print_info "Setting up clean image directory..."
cp "$INPUT_DIR"/*.{jpg,jpeg,png,JPG,JPEG,PNG} "$CLEAN_DIR/" 2>/dev/null || true

# Change to FOA-Attack directory
cd "$FOA_ATTACK_DIR"

# Dependencies list
DEPS="torch torchvision torchaudio transformers accelerate pillow tqdm scipy numpy hydra-core omegaconf pytorch-lightning opencv-python-headless safetensors"

# Check for Python virtual environment
if [ -d ".venv" ]; then
    print_info "Activating existing virtual environment..."
    source .venv/bin/activate
elif command -v uv &> /dev/null; then
    print_info "Creating virtual environment with uv..."
    uv venv .venv --python 3.10
    source .venv/bin/activate
    print_info "Installing dependencies..."
    uv pip install $DEPS
else
    print_warning "No virtual environment found and uv not available."
    print_info "Please ensure required packages are installed."
fi

# Check if flash-attn is installed (required for BagelSiglip)
if ! python -c "import flash_attn" 2>/dev/null; then
    print_warning "flash-attn not found. Installing (this may take 20-30 minutes to build)..."
    if command -v uv &> /dev/null; then
        uv pip install flash-attn --no-build-isolation
    else
        pip install flash-attn --no-build-isolation
    fi
    print_success "flash-attn installed successfully."
fi

# Add models path to Python path
export PYTHONPATH="$PROJECT_ROOT/models/Bagel:$PYTHONPATH"

print_info "Starting FOA-Attack..."
print_info ""

# Run the attack with overrides for output directory
python generate_adversarial_samples_foa_attack.py \
    data.output="$OUTPUT_DIR" \
    data.cle_data_path="resources/images/bigscale" \
    data.tgt_data_path="resources/images/target_images" \
    data.num_samples=$IMAGE_COUNT \
    wandb.enabled=false

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    print_success "FOA-Attack completed successfully!"
    print_info ""
    print_info "Results saved to: $OUTPUT_DIR"
else
    print_error "FOA-Attack failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi
