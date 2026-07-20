# Project structure:
## ViT:
- ### Old trainers:
  - Training scripts for previous (failed) ViTTT distillation attempts
- ### datasets:
  - Image-only dataset classes for ViTTT distillation
- ### images:
  - Visualisations of features produced by DINOv3 and ViTTT, for two test images (cat + Sintel)
- ### models:
  - **BidirectionalLaCT.py**: Variant 1 Bidirectional LaCT (update last layer only) from https://github.com/JunchenLiu77/ViTTT/tree/main
  - **Dinov3.py**: Utilities for working with DINOv3 from HuggingFace
  - **ViTTT**: TTT feature extractor
  - **pos_embed**: 2D RoPE embedding class (taken from DINOv3)
- **Remaining three scripts are:**
- The train script used for the 24-block variant
- The train script used for the 12-block variant (which actually outperformed the 24-block variant in both train *and* val loss)
- The script used for extracting partial losses and model weights from a training checkpoint

## PP3DR (name subject to change):
- ### datasets
  - **dataset_base.py**: The (virtual/abstract) base dataset class, which includes basic init and helper functions and handles `__getitem__()`
  - **sintel_io.py**: Provided by the Sintel dataset to aid in processing their files
  - The rest of the files are datasets implemented for their respective file structures.
- **models**:
  - **BidirectionalLaCT.py**: Same as above, plus custom global/local LaCT blocks
  - **Dinov3.py**: Same as above
  - **ViTTT**: Same as above
  - **pos_embed**: Same as above, plus custom 2D and 3D RoPE implementations
  - **PP3DR.py**: End-to-end TTT for dynamic 3D reconstruction
  - **PP3DR_Dino.py**: Baseline using DINOv3 rather than TTT for feature extraction
  - **PP3DR_depth_focal.py**: Variant which predicts intrinsics instead of XY rays
- **PP3DR_trainer.py**: Self-explanatory.
- **PP3DR_loss.py**: Self-explanatory.
- **PP3DR_finetune.py**: Train script for fine-tuning PP3DR, which involves longer sequences and unfreezes the underlying ViTTT.
- **PP3DR_depth_focal_loss.py**: Adapted loss for the depth-focal variant of the model.
- **visualise.py**: Script for Viser visualisation of model predictions
- **visualise_gt.py**: Script for Viser visualisation of GT dataset scenes.