# 🚗 Road Accident Severity Prediction using CNN

<p align="center">
  <img src="https://img.shields.io/badge/Test%20Accuracy-91.09%25-brightgreen?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Weighted%20F1-0.912-blue?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/GPU-RTX%205080-76b900?style=for-the-badge&logo=nvidia"/>
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch"/>
</p>

A deep-learning pipeline that classifies road-accident severity as **No
Accident**, **Minor Accident**, or **Severe Accident** from CCTV images using
**EfficientNetV2-S**, transfer learning, mixed-precision training, and
Grad-CAM interpretability.

The v3 repository preserves the original 989-image project and adds a
reproducible base-R evaluation and a Streamlit application. It does not use or
retrain on the separate, larger CCTV dataset developed later.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Results](#-results)
- [Dataset](#-dataset)
- [Model Architecture](#-model-architecture)
- [Training Strategy](#-training-strategy)
- [Project Structure](#-project-structure)
- [Setup and Installation](#-setup-and-installation)
- [Usage](#-usage)
- [R Evaluation](#-r-evaluation)
- [Streamlit Dashboard](#-streamlit-dashboard)
- [Classical ML Baselines](#-classical-ml-baselines)
- [Interpretability](#-interpretability)
- [Authors](#-authors)
- [License](#-license)

---

## 🔍 Overview

Existing accident-response systems are largely **reactive**: emergency
services are dispatched only after someone reports an incident. This project
explores an automated, camera-based severity classifier that analyzes CCTV
images without additional sensors.

**Key design decisions:**

- EfficientNetV2-S initialized with ImageNet weights
- Two-phase transfer learning
- CCTV-oriented augmentations such as rain, fog, motion blur, and occlusion
- Inverse-frequency class weighting and label smoothing
- Grad-CAM visualization for qualitative inspection of model activations

This is a research and demonstration project. Predictions and visual
explanations should not be used as the sole basis for emergency response.

---

## 📊 Results

### Results reproduced from the v3 checkpoint and saved test predictions

The R evaluator in this repository was run on the saved 101-image test set.
These are the current, reproducible figures for this v3 repository:

| Metric | Value |
|--------|-------|
| Test images | 101 |
| Test accuracy | **91.09%** |
| Weighted F1-score | **0.9120** |
| Macro F1-score | **0.9124** |
| Macro one-vs-rest ROC AUC | **0.9814** |
| Best-checkpoint validation accuracy | **90.48%** |
| Best-checkpoint epoch | **22** |

### Per-class results

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| No Accident | 0.958 | 0.868 | 0.911 | 53 |
| Minor Accident | 0.793 | 0.958 | 0.868 | 24 |
| Severe Accident | 0.958 | 0.958 | 0.958 | 24 |

The test accuracy bootstrap 95% confidence interval is **85.15%–96.04%**;
the macro-F1 interval is **0.855–0.961**. The test set is small, so these
intervals are informative about image-sample uncertainty but do not establish
performance on independent videos or in deployment.

### Historical figures in the predecessor README

The original repository README reported **92.08% test accuracy** and **0.9212
weighted F1**, along with epoch-level training metrics. Those headline values
do not match the checkpoint and prediction CSV committed in v3. To avoid
presenting irreproducible numbers as the v3 results, the tables above are
computed from the v3 artifacts. The predecessor README also claimed perfect
severe-class precision and recall; the saved v3 predictions show 0.958 for
both metrics.

For reference, the predecessor README's per-class table reported:

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| No Accident | 0.94 | 0.91 | 0.92 | 53 |
| Minor Accident | 0.85 | 0.92 | 0.88 | 24 |
| Severe Accident | 0.96 | 0.96 | 0.96 | 24 |
| Weighted average | 0.92 | 0.92 | 0.9212 | 101 |

Its historical training-curve table was:

| Epoch | Train loss | Train accuracy | Validation loss | Validation accuracy |
|-------|-----------:|---------------:|----------------:|--------------------:|
| 1 | 1.3099 | 30.36% | 1.1593 | 43.54% |
| 5 (unfreeze) | 0.8312 | 62.08% | 0.7666 | 73.47% |
| 10 | 0.7176 | 69.37% | 0.6697 | 85.03% |
| 23 (reported best) | 0.5461 | 84.89% | 0.5786 | 93.20% |
| 30 | 0.5440 | 86.64% | 0.5773 | 91.84% |

These are retained from the predecessor README for project history; they are
not the metrics of the epoch-22 checkpoint and test predictions evaluated in
v3. The predecessor README's sentence claiming perfect severe-class precision
and recall also conflicts with its own per-class table.

---

## 🗂️ Dataset

**Source:** [Accident Detection from CCTV Footage on
Kaggle](https://www.kaggle.com/datasets/ckay16/accident-detection-from-cctv-footage) —
989 JPEG frames from road-surveillance cameras.

The source dataset has two folders, **Accident** and **Non Accident**. Accident
frames are split into two project labels using image sharpness:

| Project class | Source | Labeling method |
|---------------|--------|-----------------|
| No Accident | Non Accident frames | Direct mapping |
| Minor Accident | Accident frames | Higher half of Laplacian-variance sharpness scores |
| Severe Accident | Accident frames | Lower half of Laplacian-variance sharpness scores |

**Important label limitation:** the source dataset does not provide verified
minor/severe impact labels for these images. The sharpness-based split is a
proxy used by this project, not ground-truth accident severity. Image blur can
reflect camera motion, compression, or lighting rather than impact severity;
interpret class-specific results accordingly.

### Split statistics

| Split | No Accident | Minor | Severe | Total |
|-------|------------:|------:|-------:|------:|
| Train | 395 | 173 | 173 | **741** |
| Validation | 79 | 34 | 34 | **147** |
| Test | 53 | 24 | 24 | **101** |
| **Total** | **527** | **231** | **231** | **989** |

### Download and prepare the dataset

```powershell
python -m pip install kaggle
kaggle datasets download -d ckay16/accident-detection-from-cctv-footage
Expand-Archive .\accident-detection-from-cctv-footage.zip -DestinationPath .\dataset3
```

Configure Kaggle credentials according to the
[Kaggle CLI documentation](https://github.com/Kaggle/kaggle-api). Alternatively,
download the dataset manually. The source folders should be under `dataset3/`
alongside the notebook. Run the notebook's dataset inspection and processing
cells to create `data/processed/`.

The raw dataset and processed images are not stored in this GitHub repository.
Restore them locally if you want to rebuild the data or see source test images
in the dashboard.

---

## 🧠 Model Architecture

```text
Input image (224 x 224 x 3)
        |
        v
EfficientNetV2-S backbone, ImageNet initialized
        |
        v
Global average pooled 1280-dimensional feature vector
        |
        v
Linear(1280 -> 512) -> BatchNorm -> GELU -> Dropout(0.35)
        |
        v
Linear(512 -> 256) -> BatchNorm -> GELU -> Dropout(0.175)
        |
        v
Linear(256 -> 3) -> Softmax probabilities
```

Output classes, in checkpoint order:
`No_Accident`, `Minor_Accident`, `Severe_Accident`.

---

## 🏋️ Training Strategy

### Phase 1 — classifier-head training

- Freeze the EfficientNetV2-S backbone and train the classifier head.
- Head learning rate: `3e-4`.
- This lets the randomly initialized head adapt before backbone fine-tuning.

### Phase 2 — full fine-tuning

- Unfreeze the backbone and train with differential learning rates.
- Backbone learning rate: `5e-5`.
- Head learning rate: `3e-4`.

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW (`beta1=0.9`, `beta2=0.999`, weight decay `1e-4`) |
| Learning-rate schedule | Cosine Annealing Warm Restarts (`T0=10`) |
| Loss | Cross-entropy with label smoothing `0.1` and class weights |
| Gradient clipping | Maximum norm `1.0` |
| Batch size | 64 |
| Maximum epochs | 30; early stopping, patience 7 |
| Precision | Automatic mixed precision when supported |
| Input size | 224 × 224 |

The saved v3 checkpoint is from epoch 22 with 90.48% validation accuracy.
Training from the notebook is optional; the dashboard uses the existing
checkpoint and does not retrain the model.

### Training-only augmentations

```python
A.HorizontalFlip(), A.VerticalFlip(), A.RandomRotate90()
A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15)
A.OneOf([MotionBlur, GaussianBlur, MedianBlur])
A.OneOf([RandomBrightnessContrast, HueSaturationValue, CLAHE])
A.RandomRain(), A.RandomFog()
A.GaussNoise(), A.CoarseDropout()
A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

---

## 📁 Project Structure

```text
.
├── road_accident_severity_cnn_v2_source.ipynb  # Output-free training notebook
├── dataset3/                                   # Local raw dataset; not committed
├── data/processed/                             # Local processed splits; not committed
├── checkpoints/
│   └── best_model.pth                          # Original trained checkpoint
├── r_analysis/
│   ├── predictions.csv                         # Saved test predictions and probabilities
│   ├── summary_metrics.csv
│   ├── metrics_by_class.csv
│   ├── confusion_matrix.csv
│   ├── error_analysis.csv
│   ├── group_analysis.csv
│   ├── evaluation_report.png                   # R-generated report displayed in app
│   └── generate_report.R                       # Base-R evaluator
├── .streamlit/config.toml                      # Formal light theme
├── streamlit_app.py                            # Dashboard
├── model_inference.py                          # Checkpoint inference and explanations
├── rscript.R                                   # R evaluation entry point
├── requirements-dashboard.txt
└── README.md
```

The original executed notebook is retained locally but excluded from the
repository because saved cell outputs embed many source-dataset images. The
output-free notebook copy is included instead.

---

## ⚙️ Setup and Installation

### Requirements

- Python 3.9 or newer
- R for the evaluation report (base R; no additional R packages required)
- A CUDA-capable GPU is optional; inference falls back to CPU
- A virtual environment is recommended

The project was developed on Windows with Python 3.10, PyTorch 2.7.1, CUDA
12.8, and an RTX 5080.

### Install Python dependencies

For GPU training, install the PyTorch build suitable for your CUDA setup. For
example:

```powershell
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Install the notebook and dashboard dependencies:

```powershell
python -m pip install timm albumentations opencv-python matplotlib seaborn `
  scikit-learn tqdm grad-cam pandas pillow
python -m pip install -r requirements-dashboard.txt
```

Or run Cell 1 in the training notebook for its training dependencies.

### Windows note

Set `NUM_WORKERS = 0` in the notebook's configuration cell (the default).
Increase it only after confirming that your local Jupyter and Windows setup
supports multiprocessing reliably.

---

## 🚀 Usage

### 1. Download and prepare the dataset

Place the source dataset under `dataset3/` beside the notebook. See the
[Dataset](#-dataset) section for the source and class-label caveat.

### 2. Run the training notebook (optional)

Open `road_accident_severity_cnn_v2_source.ipynb` in Jupyter and run the cells
from top to bottom:

| Notebook section | Purpose |
|------------------|---------|
| 1 | Install dependencies |
| 2 | Imports and GPU setup |
| 3 | Configuration and paths |
| 4 | Inspect `dataset3` structure |
| 5 | Build the three-class processed dataset |
| 6 | Augmentation pipelines |
| 7 | Dataset class and DataLoaders |
| 8 | Visualize a sample batch |
| 9 | Build EfficientNetV2-S model |
| 10 | Loss, optimizer, and scheduler |
| 11 | Training loop |
| 12 | Training curves |
| 13 | Load best model and test inference |
| 14 | Confusion matrix |
| 15 | ROC curves |
| 16 | Grad-CAM heatmaps |
| 17 | Confidence distribution |
| 18 | Predict a single image |
| 19 | Export TorchScript and ONNX |

Training requires the source dataset and can take time. It is not required to
run the Streamlit app or the R analysis.

### 3. Predict a single image in the notebook

After running the relevant model and prediction cells:

```python
result = predict_single("path/to/your/image.jpg")
# Example shape:
# {'No_Accident': 0.03, 'Minor_Accident': 0.08, 'Severe_Accident': 0.89}
```

### 4. Load the saved model in the notebook

Run the notebook's model-definition cell first so the `model` architecture is
defined, then load the checkpoint:

```python
import torch

checkpoint = torch.load("checkpoints/best_model.pth", map_location="cpu")
model.load_state_dict(checkpoint["model_state"])
print(checkpoint["epoch"], checkpoint["val_acc"], checkpoint["class_names"])
```

### 5. Generate the R report and launch the dashboard

See [R Evaluation](#-r-evaluation) and
[Streamlit Dashboard](#-streamlit-dashboard).

---

## 📈 R Evaluation

The base-R evaluator is an additional analysis use case in this project. It
does not retrain, tune, or modify the CNN. It validates and summarizes the
saved held-out predictions exported from the notebook.

From the project directory, run:

```powershell
& "C:\Program Files\R\R-4.6.1\bin\Rscript.exe" rscript.R
```

If `Rscript` is available on `PATH`, you can instead run:

```powershell
Rscript rscript.R
```

The entry point calls `r_analysis/generate_report.R`. The evaluator checks the
input schema, supported labels, unique filenames, finite class probabilities,
probability bounds, and that each row's class probabilities sum to one. It
then creates:

- Accuracy and stratified-bootstrap 95% confidence intervals for accuracy and
  macro F1 (1,000 resamples)
- Macro precision, recall, and F1; weighted F1; and one-vs-rest ROC AUC
- Per-class precision, recall, specificity, F1, support, and ROC AUC
- Multiclass Brier score
- A confusion matrix
- Per-image error details, including predicted confidence, true-class
  probability, top-two probability margin, and high-confidence errors
- Class-level error rates and the most common actual-to-predicted error
- A four-panel PNG showing the confusion matrix, ROC curves, per-class
  performance, and confidence by outcome

Outputs are saved under `r_analysis/`. The report image is
[`r_analysis/evaluation_report.png`](r_analysis/evaluation_report.png); the app
displays this plot in its **R report** tab. For compatibility, the evaluator
also refreshes `r_analysis/r_evaluation.png`.

The current saved test set contains 101 predictions and gives 91.09% accuracy
and 0.9120 weighted F1. The bootstrap intervals quantify uncertainty over this
image sample only; they do not establish video-independent generalization.

---

## 🖥️ Streamlit Dashboard

The Streamlit app provides a formal, light-themed interface to the original
model and saved evaluation:

- **Overview** — current held-out metrics, per-class performance, and
  confusion matrix
- **Test examples** — a random held-out example from each actual class, with
  the image and Grad-CAM displayed side by side; use **Show another random
  set** to rotate examples
- **Error explorer** — filter individual predictions by actual class,
  predicted class, outcome, confidence, and filename; inspect individual
  mistakes and download filtered rows
- **Group analysis** — compare error rates and common confusions by actual
  class
- **R report** — view the R-generated PNG and download summary metrics
- **Predict & explain** — upload an image and run the existing checkpoint for
  class probabilities, Grad-CAM, and input-gradient saliency

The original dataset does not contain verified source-video identifiers, so
the app groups errors by actual class instead of suggesting unsupported
video-level patterns. Test-image galleries and misclassification previews
require local images under `data/processed/test/<class>/`. Dataset images are
excluded from this repository; without them, the app still displays evaluation
results and accepts uploaded images.

Install the dashboard dependencies if necessary:

```powershell
python -m pip install -r requirements-dashboard.txt
```

Launch Streamlit from the project directory:

```powershell
streamlit run streamlit_app.py
```

The dashboard loads `checkpoints/best_model.pth`; it does not retrain the
model. Grad-CAM and input-gradient saliency are visual diagnostics, not causal
evidence or proof that the model uses valid accident features.

---

## 📉 Classical ML Baselines

The predecessor README also reported results for five models trained on a
structured tabular accident dataset (driver age, experience, time of day,
cause, and other features):

| Model | Reported test accuracy | Reported weighted F1 |
|-------|------------------------:|---------------------:|
| SVM | 61.2% | 0.58 |
| Logistic Regression | 68.4% | 0.66 |
| Decision Tree | 77.3% | 0.75 |
| Random Forest | 84.1% | 0.83 |
| XGBoost | 86.9% | 0.86 |

These are historical figures from the predecessor README, not regenerated by
the v3 evaluation. The tabular models use a different dataset and modality
from the CCTV-image CNN, so this is not a direct head-to-head comparison.

---

## 🔍 Interpretability

Grad-CAM maps are generated from the final EfficientNetV2-S feature block.
The notebook saves example visualizations, and the dashboard generates maps
for held-out images and user-uploaded images. They can help inspect which
image regions influence a prediction, but they do **not** prove that the model
attends to crash zones or learned causal accident features.

---

## 👨‍💻 Authors
**Kartik Singh Rana** — A010175723015<br>
B.Sc in Artificial Intelligence & Data Science (Hons.)<br>
Amity Institute of Information Technology

---

## 📄 License

The predecessor README states that the project is licensed under the MIT
License. However, the predecessor GitHub repository currently has no detected
license file, so this v3 repository does not add or claim a license. Confirm
the license with the project authors before redistributing or reusing the
project.
