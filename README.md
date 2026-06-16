# Heart Disease Classification — Neural Network (Keras / TensorFlow)

A feed-forward neural network that classifies patients into one of **five heart-disease
severity levels (0–4)** using the UCI Cleveland Heart Disease dataset. Built with
TensorFlow/Keras, with a focus on a **leakage-free preprocessing pipeline** and
**class-imbalance handling via SMOTE**.


---

## Problem

The [UCI Cleveland Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease)
contains 13 clinical features (age, sex, chest-pain type, resting blood pressure,
cholesterol, etc.). The target ranges from `0` (no presence) to `4` (most severe). The
goal is a multi-class classifier over all five levels — harder than the common binary
"disease / no disease" framing, because the higher-severity classes are rare.

## Dataset

- **Source:** UCI Machine Learning Repository — Heart Disease (Cleveland subset).
- **Rows:** 303 (a handful contain missing values marked `?`, which are dropped).
- **Features (13):** age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang,
  oldpeak, slope, ca, thal
- **Target:** class_attbr ∈ {0, 1, 2, 3, 4}

The data file is included for reproducibility and is publicly available from UCI for
research and educational use.

## Approach & methodology

The pipeline is deliberately built to avoid **data leakage**:

1. **Split first.** Stratified train / validation / test split (~80 / 10 / 10) happens
   *before* any scaling or resampling.
2. **SMOTE on the training set only.** Synthetic minority-class samples are generated
   after the split and only on training data.
3. **Scaler fit on the training set only.** StandardScaler is fit on training data and
   merely applied to validation and test.

### Model

| Layer            | Units | Notes                                          |
|------------------|-------|------------------------------------------------|
| Dense (hidden 1) | 22    | ReLU, He-normal init, L2 regularisation (0.01) |
| Dropout          | —     | rate 0.2                                        |
| Dense (hidden 2) | 19    | ReLU, He-normal init                            |
| Dense (output)   | 5     | Softmax (5 classes)                             |

- **Loss:** sparse categorical cross-entropy
- **Optimizer:** Adam (lr = 0.001)
- **Callbacks:** EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
- **Epochs:** up to 300 (early-stopped)

## Results

| Metric              | Value (unseeded run) |
|---------------------|----------------------|
| Train accuracy      | ~80%   |
| Validation accuracy | ~67%   |
| Test accuracy       | ~67%   |

> **Note on variance.** The test set is only 30 samples, so each sample is worth ~3.3%
> of accuracy and results vary run-to-run. Fixed random seed is added to make 
> runs reproducible.

Running the script also produces training/validation curves, a confusion matrix, and a
per-class classification report.

## Project structure

```
Heart_Disease_Neural_Network/
├── data/processed.cleveland.data   # UCI Cleveland dataset
├── models/                         # trained .keras files (generated, git-ignored)
├── scaler/                         # saved StandardScaler (generated, git-ignored)
├── src/main.py                     # load → preprocess → train → evaluate
├── requirements.txt
├── .gitignore
└── README.md
```

## How to run

From the repo root:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python src/main.py
```

## Tech stack

Python · TensorFlow / Keras · scikit-learn · imbalanced-learn (SMOTE) · pandas · NumPy ·
Matplotlib · seaborn

---

**Author:** Panagiotis Christofilopoulos — [GitHub](https://github.com/panalis) · [LinkedIn](https://linkedin.com/in/panagiotis-christofilopoulos-7715a2357)