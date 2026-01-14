# Attack-Bard (Refactored - Minimal Version)

This is a refactored minimal version of the Attack-Bard codebase, containing only the essential components needed to generate adversarial images using the image encoder misdescription attack.

## Overview

This attack generates adversarial examples that can mislead vision-language models by attacking white-box surrogate vision encoders (BLIP, CLIP, ViT) using the Spectrum Simulation Attack (SSA) combined with Common Weakness targeting.

Original research: [How Robust is Google's Bard to Adversarial Image Attacks?](https://arxiv.org/abs/2309.11751)

## Directory Structure

```
Attack-Bard/
├── attack_img_encoder_misdescription.py  # Main attack script
├── attacks/                              # Attack implementations
│   ├── AdversarialInput/
│   │   ├── AdversarialInputBase.py      # Base attacker class
│   │   ├── CommonWeakness.py            # Common weakness attacks
│   │   ├── SpectrumSimulationAttack.py  # SSA implementation
│   │   └── utils.py                     # Attack utilities
│   └── utils.py                          # General attack utilities
├── surrogates/                           # Surrogate models
│   └── FeatureExtractors/
│       ├── Base.py                       # Ensemble feature extractor
│       ├── Blip.py                       # BLIP feature extractor
│       ├── Clip.py                       # CLIP feature extractor
│       └── ViT.py                        # Vision Transformer extractor
├── utils/                                # Utilities
│   └── ImageHandling.py                  # Image loading/saving
└── dataset/                              # Test images and targets
```

## Usage

Generate adversarial images with different attack strengths:

```bash
python attack_img_encoder_misdescription.py
```

The script will:
- Load test images from `./dataset/test-img/`
- Generate adversarial examples at 5 strength levels (weak, mild, medium, strong, very_strong)
- Save results to `./attack_img_encoder_misdescription/{strength}/`

### Attack Parameters

The script generates attacks at different epsilon values:
- **weak**: ε=8/255
- **mild**: ε=16/255
- **medium**: ε=32/255
- **strong**: ε=64/255
- **very_strong**: ε=128/255

## Requirements

- PyTorch
- torchvision
- transformers (for BLIP, CLIP models)
- timm (for Vision Transformer)
- PIL
- tqdm

## Citation

```bibtex
@article{dong2023robust,
  title={How Robust is Google's Bard to Adversarial Image Attacks?},
  author={Dong, Yinpeng and Chen, Huanran and Chen, Jiawei and Fang, Zhengwei and Yang, Xiao and Zhang, Yichi and Tian, Yu and Su, Hang and Zhu, Jun},
  journal={arXiv preprint arXiv:2309.11751},
  year={2023}
}
```

## Acknowledgement

Original code based on [MiniGPT4](https://github.com/Vision-CAIR/MiniGPT-4) and [AdversarialAttacks](https://github.com/huanranchen/AdversarialAttacks). 



