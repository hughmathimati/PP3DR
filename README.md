# Project structure:
## ViT:
- **Old trainers**: Training scripts for previous (failed) ViTTT distillation attempts
- **datasets**: Image-only dataset classes for ViTTT distillation
- **images**: Visualisations of features produced by DINOv3 and ViTTT, for two test images (cat + Sintel)
- **models**:
  - **BidirectionalLaCT.py**: Variant 1 Bidirectional LaCT (update last layer only) from https://github.com/JunchenLiu77/ViTTT/tree/main
  - **Dinov3.py**: Utilities for working with DINOv3 from HuggingFace
  - **ViTTT**: TTT feature extractor
  - **pos_embed**: 2D RoPE embedding class (taken from DINOv3)
- Remaining two scripts are 1) the current train script, and 2) the script used for extracting partial losses and model weights from a training checkpoint

## PP3DR (name subject to change):
- **models**:
  - **BidirectionalLaCT.py**: Same as above
  - **Dinov3.py**: Same as above
  - **ViTTT**: Same as above
  - **pos_embed**: Same as above, plus custom 3D partial RoPE implementation
  - **PP3DR.py**: End-to-end TTT for dynamic 3D reconstruction
  - **PP3DR_Dino.py**: Baseline using DINOv3 rather than TTT for feature extraction