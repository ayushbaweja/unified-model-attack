#!/bin/bash
# This script runs the PGD (Projected Gradient Descent) attack on Bagel VAE

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
PGD_DIR="$PROJECT_ROOT/attacks/PGD"
RESOURCES_DIR="$PROJECT_ROOT/resources"
MODELS_DIR="$PROJECT_ROOT/models"
DEFAULT_INPUT_DIR="$RESOURCES_DIR/test-images"
DEFAULT_OUTPUT_DIR="$PROJECT_ROOT/results/pgd"
DEFAULT_MODEL_PATH="$MODELS_DIR/Bagel/models/BAGEL-7B-MoT/ae.safetensors"

# Parse arguments
INPUT_DIR="${1:-$DEFAULT_INPUT_DIR}"
OUTPUT_DIR="${2:-$DEFAULT_OUTPUT_DIR}"
EPSILON="${3:-0.06}"
ALPHA="${4:-0.01}"
ITERATIONS="${5:-40}"
ANALYZE="${6:-true}"
LOSS="${7:-pixel}"
LATENT_MODE="${8:-mean}"
VAR_WEIGHT="${9:-1.0}"

# Build analyze flag
ANALYZE_FLAG=""
if [ "$ANALYZE" = "true" ] || [ "$ANALYZE" = "1" ] || [ "$ANALYZE" = "yes" ]; then
    ANALYZE_FLAG="--analyze"
fi

# Determine output subdirectory name
if [ "$LOSS" = "latent" ]; then
    OUTPUT_SUBDIR="eps_${EPSILON}_${LOSS}_${LATENT_MODE}"
else
    OUTPUT_SUBDIR="eps_${EPSILON}_${LOSS}"
fi

print_info "PGD Attack on Bagel VAE"
print_info "======================="
print_info ""
print_info "Input images directory: $INPUT_DIR"
print_info "Output directory: $OUTPUT_DIR/$OUTPUT_SUBDIR"
print_info "Model path: $DEFAULT_MODEL_PATH"
print_info "Epsilon: $EPSILON"
print_info "Alpha: $ALPHA"
print_info "Iterations: $ITERATIONS"
print_info "Loss space: $LOSS"
if [ "$LOSS" = "latent" ]; then
    print_info "Latent mode: $LATENT_MODE"
    print_info "Var weight: $VAR_WEIGHT"
fi
if [ -n "$ANALYZE_FLAG" ]; then
    print_info "VAE analysis: enabled"
else
    print_info "VAE analysis: disabled"
fi
print_info ""

# Check if input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    print_error "Input directory does not exist: $INPUT_DIR"
    print_info "Usage: $0 [input_dir] [output_dir] [epsilon] [alpha] [iterations] [analyze] [loss] [latent_mode] [var_weight]"
    print_info ""
    print_info "Arguments:"
    print_info "  input_dir    - Directory containing input images (default: resources/test-images)"
    print_info "  output_dir   - Base output directory (default: results/pgd)"
    print_info "  epsilon      - L-inf perturbation budget (default: 0.06)"
    print_info "  alpha        - PGD step size (default: 0.01)"
    print_info "  iterations   - Number of PGD iterations (default: 40)"
    print_info "  analyze      - Run VAE analysis: true/false (default: true)"
    print_info "  loss         - Loss space: pixel/latent (default: pixel)"
    print_info "  latent_mode  - Latent attack mode: mean/mean_var/mean_neg_var (default: mean, only for loss=latent)"
    print_info "  var_weight   - Weight for variance term (default: 1.0, only for loss=latent)"
    exit 1
fi

# Check if model exists
if [ ! -f "$DEFAULT_MODEL_PATH" ]; then
    print_warning "Model file not found at: $DEFAULT_MODEL_PATH"
    print_info "Please ensure the Bagel model is downloaded."
fi

# Check if there are images in the input directory
IMAGE_COUNT=$(find "$INPUT_DIR" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | wc -l)
if [ "$IMAGE_COUNT" -eq 0 ]; then
    print_error "No images found in $INPUT_DIR"
    exit 1
fi

print_info "Found $IMAGE_COUNT images in input directory"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Change to PGD directory
cd "$PGD_DIR"

# Check for uv or use existing Python
if command -v uv &> /dev/null; then
    print_info "Running with uv..."

    # Create inline script with dependencies
    TEMP_SCRIPT="$OUTPUT_DIR/pgd_runner.py"

    cat > "$TEMP_SCRIPT" << 'EOFPYTHON'
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch",
#     "torchvision",
#     "pillow",
#     "tqdm",
#     "numpy",
#     "einops",
#     "safetensors",
#     "packaging",
# ]
# ///

import sys
import os

# Add PGD directory to path
pgd_dir = sys.argv[1]
sys.path.insert(0, pgd_dir)

# Import and run the main function from pgd_attack
from pgd_attack import main
import sys

# The remaining args are passed to argparse
sys.argv = ['pgd_attack.py'] + sys.argv[2:]
main()
EOFPYTHON

    # Build latent mode flags
    LATENT_FLAGS=""
    if [ "$LOSS" = "latent" ]; then
        LATENT_FLAGS="--latent_mode $LATENT_MODE --var_weight $VAR_WEIGHT"
    fi

    uv run --python 3.10 "$TEMP_SCRIPT" "$PGD_DIR" \
        --input_dir "$INPUT_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --model_path "$DEFAULT_MODEL_PATH" \
        --epsilon "$EPSILON" \
        --alpha "$ALPHA" \
        --iter "$ITERATIONS" \
        --loss "$LOSS" \
        $LATENT_FLAGS \
        $ANALYZE_FLAG

    EXIT_CODE=$?
    rm -f "$TEMP_SCRIPT"
else
    print_info "Running with system Python..."

    # Build latent mode flags
    LATENT_FLAGS=""
    if [ "$LOSS" = "latent" ]; then
        LATENT_FLAGS="--latent_mode $LATENT_MODE --var_weight $VAR_WEIGHT"
    fi

    python pgd_attack.py \
        --input_dir "$INPUT_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --model_path "$DEFAULT_MODEL_PATH" \
        --epsilon "$EPSILON" \
        --alpha "$ALPHA" \
        --iter "$ITERATIONS" \
        --loss "$LOSS" \
        $LATENT_FLAGS \
        $ANALYZE_FLAG

    EXIT_CODE=$?
fi

if [ $EXIT_CODE -eq 0 ]; then
    print_success "PGD Attack completed successfully!"
    print_info ""
    print_info "Results saved to: $OUTPUT_DIR/$OUTPUT_SUBDIR"
else
    print_error "PGD Attack failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi
