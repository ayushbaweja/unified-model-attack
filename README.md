# Unified Model Attack

A repository for adversarial attacks on vision-language models, with a focus on attacking the Bagel model.

## Repository Structure

```
unified-model-attack/
├── attacks/
│   ├── Attack-Bard/          # SSA-based attack using BLIP, CLIP, ViT ensemble
│   ├── FOA-Attack/           # Feature Optimal Alignment attack with BagelSiglip
│   └── PGD/                  # PGD attack on Bagel VAE
├── models/
│   └── Bagel/                # Bagel model files (shared across attacks)
├── resources/
│   └── test-images/          # Test images for benchmarking
├── results/                  # Output directory for attack results
│   ├── attack-bard/
│   ├── foa-attack/
│   └── pgd/
└── scripts/                  # Run scripts for each attack
    ├── run_attack_bard.sh
    ├── run_foa_attack.sh
    └── run_pgd.sh
```

## Prerequisites

- Python 3.10+
- CUDA-capable GPU
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Installing uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Setup

### 1. Clone the repository

```bash
git clone <repository-url>
cd unified-model-attack
```

### 2. Download the Bagel model

The Bagel model is hosted on Hugging Face. Use the provided download script:

```bash
cd models/Bagel
pip install huggingface_hub
python model-download.py
```

This downloads the `ByteDance-Seed/BAGEL-7B-MoT` model to `models/Bagel/models/BAGEL-7B-MoT/`.

**Required files:**
- `ae.safetensors` - VAE weights (for PGD attack)
- `*.safetensors` - Model checkpoint shards (for FOA-Attack's BagelSiglip)
- `config.json` - Model configuration

**Note:** The full model is ~28GB. Ensure sufficient disk space before downloading.

### 3. Add test images

Place your test images (`.jpg`, `.jpeg`, `.png`) in:
```
resources/test-images/
```

## Running Attacks

All scripts automatically handle dependencies using `uv` and save results to the `results/` directory.

### PGD Attack (Bagel VAE)

Attacks the Bagel VAE encoder to maximize reconstruction error.

```bash
# Basic usage (uses default test images)
./scripts/run_pgd.sh

# Custom input/output directories
./scripts/run_pgd.sh <input_dir> <output_dir>

# Full options
./scripts/run_pgd.sh <input_dir> <output_dir> <epsilon> <alpha> <iterations>
```

**Parameters:**
- `epsilon`: Perturbation magnitude (default: 0.03)
- `alpha`: Step size (default: 0.01)
- `iterations`: Number of PGD iterations (default: 40)

**Output:** `results/pgd/<image>_eps<epsilon>_adv.png`

---

### FOA Attack (Feature Optimal Alignment)

Uses an ensemble of feature extractors including BagelSiglip to generate transferable adversarial examples.

```bash
# Basic usage
./scripts/run_foa_attack.sh

# Custom directories
./scripts/run_foa_attack.sh <input_dir> <output_dir>
```

**Features:**
- Ensemble of CLIP, BLIP, ViT, and BagelSiglip models
- 300 attack iterations per image
- Source and target crop augmentation

**Output:** `results/foa-attack/img/<hash>/test/<image>.png`

---

### Attack-Bard (SSA Attack)

Spectrum Simulation Attack using an ensemble of BLIP, CLIP, and ViT models.

```bash
# Basic usage
./scripts/run_attack_bard.sh

# Custom directories
./scripts/run_attack_bard.sh <input_dir> <output_dir>
```

**Features:**
- Multiple attack strengths: weak, mild, medium, strong, very_strong
- 500 iterations per strength level
- Ensemble of BLIP-2, CLIP, and ViT-Large models

**Output:** `results/attack-bard/<strength>/<image>_adv.png`

## Attack Comparison

| Attack | Target | Method | Iterations | Surrogates |
|--------|--------|--------|------------|------------|
| PGD | Bagel VAE | Gradient-based | 40 | Bagel VAE |
| FOA | VLM encoders | Feature alignment | 300 | CLIP, BLIP, ViT, BagelSiglip |
| Attack-Bard | VLM encoders | Spectrum simulation | 500 | BLIP-2, CLIP, ViT-Large |

## Configuration

### FOA-Attack

Configuration files are in `attacks/FOA-Attack/config/`:
- `ensemble_3models.yaml`: Main configuration
- Backbone models, attack parameters, and data paths

### Attack-Bard

Attack strengths and parameters are defined in `attacks/Attack-Bard/attack_img_encoder_misdescription.py`

## Troubleshooting

### flash-attn installation

FOA-Attack requires `flash-attn` for BagelSiglip. If installation fails:

```bash
cd attacks/FOA-Attack
source .venv/bin/activate
pip install flash-attn --no-build-isolation
```

### CUDA out of memory

Reduce batch size in the respective config files or process fewer images at once.

### Missing model weights

Ensure the Bagel model is downloaded:

```bash
cd models/Bagel
python model-download.py
```

The model will be saved to `models/Bagel/models/BAGEL-7B-MoT/`.
