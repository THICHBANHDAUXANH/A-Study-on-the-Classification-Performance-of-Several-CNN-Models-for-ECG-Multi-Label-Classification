# ECG Multi-Label Classification

[![Hugging Face](https://img.shields.io/badge/HuggingFace-Checkpoints-yellow)](https://huggingface.co/fhaosss/CNN_model_for_ptb-xl_classification)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](requirements.txt)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-red)](requirements.txt)

This repository contains the report code for ECG multi-label classification on PTB-XL using image-based CNN models.

## Updates

- Report pipeline cleaned for GitHub release.
- Pretrained CNN checkpoints are hosted on Hugging Face.
- Signal EDA is kept as a notebook for easier inspection and rerunning.

## Highlights

- PTB-XL 5-superclass classification: `CD`, `HYP`, `MI`, `NORM`, `STTC`.
- Signal preprocessing: moving-average smoothing, 50 Hz notch filtering, and 0.5 Hz high-pass filtering.
- CNN baselines: ResNet-50, DenseNet-121, and Inception-v3.
- Grad-CAM visualization for ECG image interpretation.

## Installation

### Requirements

- Python 3.9+
- PyTorch 2.1+
- torchvision
- wfdb
- numpy, pandas, scipy, scikit-learn, matplotlib

### Environment Installation

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Datasets

Please download PTB-XL and place it under `data/`:

```text
data
├── ptbxl_database.csv
├── scp_statements.csv
└── records100
```

You can change dataset paths in [config.py](src/config.py).

## Model Zoo

| Model | Test | Epoch | Precision | Recall | F1-score | Params | config | download |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| ResNet-50 | PTB-XL 5-label | 50 | 66.38 | 68.73 | 67.39 | 23.5M | [config](model/train/resnet50.py) | [HF](https://huggingface.co/fhaosss/CNN_model_for_ptb-xl_classification/blob/main/best_resnet50_ecg_model.pth) |
| DenseNet-121 | PTB-XL 5-label | 50 | 66.94 | 65.56 | 64.35 | 8.0M | [config](model/train/densenet121.py) | [HF](https://huggingface.co/fhaosss/CNN_model_for_ptb-xl_classification/blob/main/best_densenet121_ecg_model.pth) |
| Inception-v3 | PTB-XL 5-label | 50 | 72.58 | 66.96 | 69.16 | 27.2M | [config](model/train/inception_v3.py) | [HF](https://huggingface.co/fhaosss/CNN_model_for_ptb-xl_classification/blob/main/best_inception_v3_ecg_model.pth) |

## Data Preparation

Run preprocessing and export arrays:

```bash
.venv/bin/python src/preprocess_and_render.py
.venv/bin/python src/export_numpy_arrays.py
```

Generated files follow this flow:

```text
PTB-XL WFDB records + metadata
  -> outputs/preprocessed_signals + outputs/ecg_images + outputs/labels
  -> outputs/arrays/ecg_images_array.npy + outputs/arrays/ecg_labels_array.npy
```

## Model Training

Train each CNN backbone from scratch:

```bash
.venv/bin/python model/train/resnet50.py
.venv/bin/python model/train/densenet121.py
.venv/bin/python model/train/inception_v3.py
```

Checkpoints and histories are saved to:

```text
outputs/models
```

## EDA

Signal preprocessing evidence and report figures are in:

```text
notebooks/signal_preprocessing_eda.ipynb
```

## Visualization

Run Grad-CAM after placing checkpoints in `outputs/models/`:

```bash
.venv/bin/python src/gradcam_resnet50.py
.venv/bin/python src/gradcam_densenet121.py
.venv/bin/python src/gradcam_inception_v3.py
```

Results are saved by model:

```text
outputs/gradcam/resnet50
outputs/gradcam/densenet121
outputs/gradcam/inception_v3
```

## Report Assets

Only images used in the report are kept in [report_assets](report_assets). Generated outputs are ignored by git.
