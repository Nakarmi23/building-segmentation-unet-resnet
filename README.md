# Building Segmentation with U-Net and ResU-Net

Building footprint segmentation from aerial imagery using U-Net and ResU-Net architectures on the WHU Building Dataset.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [Outputs](#outputs)
- [Citation](#citation)

## Overview

This project implements semantic segmentation models for extracting building footprints from aerial images. Two deep learning architectures are implemented:
- **U-Net**: Classic encoder-decoder architecture for image segmentation
- **ResU-Net**: U-Net with residual blocks for improved feature learning

## Dataset

### WHU Building Dataset

The project uses the **WHU Aerial Imagery Building Dataset**, which contains:
- High-resolution aerial imagery (0.3m spatial resolution)
- Binary building/non-building labels
- Training, validation, and test splits

**Download Link:** [WHU Building Dataset](https://gpcv.whu.edu.cn/data/3.%20The%20cropped%20aerial%20image%20tiles%20and%20raster%20labels.zip)

**Dataset Reference:**
```
Ji, S., Wei, S., & Lu, M. (2018).
Fully Convolutional Networks for Multisource Building Extraction from an Open Aerial and Satellite Imagery Data Set.
IEEE Transactions on Geoscience and Remote Sensing.
```

## Project Structure

```
building-segmentation-unet-resnet/
├── data/
│   ├── raw/                    # Raw dataset (TIF format)
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── train/                  # Converted PNG format
│   ├── val/
│   └── test/
├── src/
│   ├── datasets/               # Dataset loaders and transforms
│   │   └── whu_dataset.py
│   ├── models/                 # Model architectures
│   │   ├── unet.py
│   │   ├── unet_blocks.py
│   │   ├── resunet.py
│   │   └── resunet_blocks.py
│   ├── losses/                 # Loss functions
│   │   ├── dice_loss.py
│   │   └── combined_loss.py
│   ├── metrics/                # Evaluation metrics
│   │   └── segmentation.py
│   ├── utils/                  # Utility functions
│   │   └── metric_logger.py
│   ├── scripts/                # Helper scripts
│   │   ├── convert_tif_to_png.py
│   │   ├── check_data.py
│   │   ├── visualize_predictions.py
│   │   └── visualize_predictions_res.py
│   ├── train.py                # Training script for U-Net
│   ├── train_res.py            # Training script for ResU-Net
│   ├── one_batch_overfit_test.py  # Model sanity check
│   ├── metrics_visualization.ipynb
│   └── metrics_visualization_res.ipynb
├── checkpoints/                # Saved model checkpoints
│   ├── unet/
│   └── resunet/
├── outputs/                    # Training outputs
│   ├── metrics/                # Training/validation metrics (CSV)
│   └── preds/                  # Prediction visualizations
│       ├── unet/
│       └── resunet/
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.10+ (for CUDA PyTorch support)
- CUDA-capable GPU (recommended)
- 8GB+ RAM

## Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd building-segmentation-unet-resnet
```

2. **Create a virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**

**For CPU only:**
```bash
pip install -r requirements.txt
```

**For GPU (CUDA support - Recommended):**

If you have a CUDA-capable GPU, install PyTorch with CUDA support first:

```bash
# Check CUDA version
nvidia-smi

# Install PyTorch with CUDA support (example for CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Then install remaining dependencies
pip install numpy>=1.23 Pillow>=9.4 matplotlib>=3.7 pandas>=2.0
```

For other CUDA versions, visit [PyTorch Get Started](https://pytorch.org/get-started/locally/) to get the correct installation command.

**Verify GPU availability:**
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

## Usage

### Step 1: Download and Prepare Dataset

1. Download the WHU Building Dataset:
   ```bash
   # Download from: https://gpcv.whu.edu.cn/data/3.%20The%20cropped%20aerial%20image%20tiles%20and%20raster%20labels.zip
   ```

2. Extract and organize the dataset:
   ```bash
   # Extract the downloaded zip file
   # Copy the contents to data/raw/
   # Expected structure:
   # data/raw/train/{image,label}/
   # data/raw/val/{image,label}/
   # data/raw/test/{image,label}/
   ```

### Step 2: Convert Dataset Format

Convert TIF images to PNG format:
```bash
python -m src.scripts.convert_tif_to_png
```

This creates the processed dataset in `data/train/`, `data/val/`, and `data/test/`.

### Step 3: Verify Dataset

Check if the data is loaded correctly:
```bash
python -m src.scripts.check_data
```

This will display sample images and masks to verify the data pipeline.

### Step 4: Model Sanity Check

Run the one-batch overfit test to ensure the model can learn:
```bash
python -m src.one_batch_overfit_test
```

This trains the model on a single batch to verify the model architecture and training loop are working correctly.

### Step 5: Train Models

**Train U-Net:**
```bash
python -m src.train
```

**Train ResU-Net:**
```bash
python -m src.train_res
```

Training hyperparameters can be modified in the respective training scripts:
- Image size: 256×256 (default)
- Batch size: 4 (default)
- Learning rate: 1e-4 (default)
- Epochs: 20 (default)

### Step 6: Visualize Results

**Visualize U-Net predictions:**
```bash
python -m src.scripts.visualize_predictions
```
This saves visualization images to `outputs/preds/unet/`

**Visualize ResU-Net predictions:**
```bash
python -m src.scripts.visualize_predictions_res
```
This saves visualization images to `outputs/preds/resunet/`

### Step 7: Analyze Metrics

Use the Jupyter notebooks to analyze training metrics:
```bash
jupyter notebook src/metrics_visualization.ipynb          # For U-Net
jupyter notebook src/metrics_visualization_res.ipynb      # For ResU-Net
```

## Models

### U-Net
- Classic encoder-decoder architecture
- 4 downsampling and 4 upsampling stages
- Skip connections for feature preservation
- ~31M parameters

### ResU-Net
- U-Net with residual blocks
- Improved gradient flow
- Better feature learning
- ~34M parameters

### Loss Function
Combined BCE and Dice Loss:
- Binary Cross-Entropy (BCE) for pixel-wise classification
- Dice Loss for handling class imbalance

### Metrics
- **Dice Score**: Overlap between prediction and ground truth
- **IoU (Intersection over Union)**: Jaccard index
- **Pixel Accuracy**: Percentage of correctly classified pixels

## Outputs

After training, you'll find:

```
checkpoints/
├── unet/
│   └── best_model.pth          # Best U-Net model checkpoint
└── resunet/
    └── best_model.pth          # Best ResU-Net model checkpoint

outputs/
├── metrics/
│   ├── unet_metrics.csv        # U-Net training/validation metrics
│   └── resunet_metrics.csv     # ResU-Net training/validation metrics
└── preds/
    ├── unet/                   # U-Net prediction visualizations
    └── resunet/                # ResU-Net prediction visualizations
```

### Checkpoint Contents
Each `.pth` file contains:
- Model state dict
- Optimizer state dict
- Epoch number
- Best validation loss/metrics

### Metrics CSV Format
Columns: `epoch`, `train_loss`, `val_loss`, `val_dice`, `val_iou`, `val_pixel_acc`

## Citation

If you use the WHU Building Dataset, please cite:

```bibtex
@article{ji2018fully,
  title={Fully Convolutional Networks for Multisource Building Extraction from an Open Aerial and Satellite Imagery Data Set},
  author={Ji, Shunping and Wei, Shiqing and Lu, Meng},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  volume={57},
  number={1},
  pages={574--586},
  year={2019},
  publisher={IEEE}
}
```

## Troubleshooting

**Out of Memory Error:**
- Reduce batch size in training scripts
- Reduce image size
- Use gradient accumulation

**Poor Performance:**
- Train for more epochs
- Adjust learning rate
- Check data augmentation settings
- Verify dataset quality

**Data Loading Issues:**
- Ensure dataset structure matches expected format
- Verify image and mask file names match
- Check file permissions

---

**Note:** This project is for educational and research purposes.
