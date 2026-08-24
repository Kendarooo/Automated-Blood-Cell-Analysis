# Automated Blood Cell Analysis

End-to-end machine learning pipeline for automated analysis of peripheral blood smear images. The project detects, characterizes, and classifies three major blood-cell categories from the BCCD dataset: **red blood cells (RBC)**, **white blood cells (WBC)**, and **platelets**.

The system combines object detection, deep feature extraction, supervised classification, experiment tracking, and statistical inference in a modular Python pipeline.

> Academic project developed for **EL5857 Aprendizaje Automático** at the Tecnológico de Costa Rica (TEC). This public repository preserves the original development history while presenting the project as a portfolio artifact.

## Pipeline

1. **Cell detection — YOLO26s**
   - Fine-tunes a pretrained YOLO26s detector on BCCD images.
   - Localizes RBC, WBC, and platelets.
   - Supports confidence-threshold calibration and IoU-based evaluation.

2. **Cell extraction and augmentation**
   - Crops detected cells to a uniform `224 x 224` representation.
   - Applies geometric, photometric, and noise augmentations to reduce overfitting.

3. **Deep feature extraction — ResNet18**
   - Uses a frozen pretrained ResNet18 as a feature extractor.
   - The default configuration truncates the network at `layer3`.
   - Features are normalized before classifier training.

4. **Supervised classification**
   - **ANN:** custom PyTorch neural network with an explicit training loop.
   - **SVM:** scikit-learn implementation with configurable kernels and hyperparameters.
   - Hyperparameter experiments can be tracked with Weights & Biases (W&B).

5. **Statistical inference**
   - Builds reference cell proportions from the training split.
   - Compares predictions against the baseline using statistical hypothesis testing.
   - Produces an alert when the configured significance criterion is met.

## Key result

Detector confidence calibration found a best threshold of **0.4**, reaching **F1 = 0.7958** in the documented evaluation workflow.

The repository also includes tooling for label comparison, confusion matrices, per-image evaluation, ANN/SVM experiments, and end-to-end inference. Generated experiment artifacts are stored under `outputs/` and are intentionally excluded from version control.

## Technology stack

- Python 3.12+
- PyTorch / Torchvision
- Ultralytics YOLO
- scikit-learn
- OpenCV
- NumPy / Pandas / SciPy
- Weights & Biases
- PyTest
- Ruff

## Repository structure

```text
.
├── configs/                 # Pipeline and sweep configuration
├── data/                    # BCCD dataset metadata/data
├── src/
│   ├── data/                # Dataset loading
│   ├── detection/           # YOLO training and inference
│   ├── evaluation/          # Threshold calibration and label evaluation
│   ├── experiments/         # ANN/SVM experiment orchestration
│   ├── features/            # ResNet18 feature extraction and normalization
│   ├── inference/           # End-to-end inference and statistical analysis
│   ├── models/              # ANN and SVM models
│   ├── training/            # Training utilities
│   └── utils/               # Shared utilities and W&B integration
├── tests/                   # Unit/integration tests and pipeline benchmark
├── dataset_view.py          # BCCD dataset explorer
├── RUNBOOK.md               # Full reproducibility workflow
├── requirements.txt
└── T2.pdf                   # Original academic assignment specification
```

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If experiment tracking is enabled in `configs/config.yaml`, authenticate with W&B:

```bash
wandb login
```

Review the configuration before running the pipeline:

```bash
cat configs/config.yaml
```

## Explore the dataset

The repository includes `dataset_view.py`, a lightweight tool for inspecting the BCCD images and annotations and preparing the dataset when needed.

```bash
python dataset_view.py
```

## Tests

Run the complete test suite:

```bash
pytest tests/ -v
```

Examples of focused tests:

```bash
pytest tests/test_extractor.py -v
pytest tests/test_transforms.py -v
pytest tests/test_ann_training.py -v
pytest tests/test_statistical_inference.py -v
```

## Run the pipeline

Train the complete detector → feature extraction → ANN → SVM → baseline workflow:

```bash
python main.py --stage all --config configs/config.yaml
```

Individual stages can also be executed separately. The complete command sequence, expected artifacts, inference workflow, and final evaluation procedure are documented in [`RUNBOOK.md`](RUNBOOK.md).

## Configuration highlights

The default configuration uses:

- YOLO input size: `320`
- Detection confidence threshold: `0.4`
- Cell crop size: `224 x 224`
- ResNet18 truncation: `layer3`
- ANN hidden layers: `[128, 64]`
- ANN dropout: `0.6`
- SVM default kernel: `linear`
- Statistical significance level: `alpha = 0.05`

## Academic context

This project originated as **Tarea 2 — Datos, aprendizaje supervisado y una aplicación médica** for EL5857. The original assignment specification is retained as `T2.pdf` for context. The implementation in this repository extends the provided starting point into a modular training, evaluation, and inference pipeline with automated tests and experiment tracking.
