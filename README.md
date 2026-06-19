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

### 1. Multiclass — single hold-out split (`multiclass.py`)

Split: train 237 / val 30 / test 30.

```
              precision  recall    f1     support
   0           0.8235   0.8750   0.8485     16
   1           0.5000   0.3333   0.4000      6
   2           0.2500   0.3333   0.2857      3
   3           0.2000   0.2500   0.2222      4
   4           0.0000   0.0000   0.0000      1
   accuracy                      0.6000     30
```

Train 71.09% | Val 63.33% | Test 60.00%

> **Note:** a single 30-sample test set has high variance — not a reliable estimate.
> Earlier runs of the same model ranged 60–70% on test. This is why the cross-validated
> result (section 2) is the one to trust.

---

### 2. Multiclass — 5-fold stratified cross-validation (`cross_val.py`)

297 patients, each tested exactly once (out-of-fold).

| Fold | Accuracy | Balanced acc | Macro-F1 |
|------|----------|--------------|----------|
| 1 | 46.67% | 26.58% | 0.260 |
| 2 | 46.67% | 24.35% | 0.236 |
| 3 | 47.46% | 30.01% | 0.300 |
| 4 | 59.32% | 37.81% | 0.363 |
| 5 | 55.93% | 35.60% | 0.355 |

| Metric | NN model | Majority baseline |
|--------|----------|-------------------|
| Accuracy | 51.21% ± 5.36% | 53.88% ± 0.44% |
| Balanced accuracy | 30.87% ± 5.14% | 20.00% ± 0.00% |
| Macro F1 | 0.303 ± 0.050 | 0.140 ± 0.001 |

Aggregated out-of-fold classification report (all 297 patients):

```
              precision  recall    f1     support
   0           0.8188   0.7625   0.7896    160
   1           0.2037   0.2037   0.2037     54
   2           0.2093   0.2571   0.2308     35
   3           0.2250   0.2571   0.2400     35
   4           0.0909   0.0769   0.0833     13
   accuracy                      0.5118    297
   macro avg   0.3095   0.3115   0.3095    297
   weighted    0.5333   0.5118   0.5216    297
```

**Reading:** raw accuracy (51%) sits *below* the 54% majority baseline, but the
imbalance-aware metrics tell the real story — balanced accuracy ~1.5× the baseline
and macro-F1 ~2.2× the baseline. The model learns genuine minority-class signal, but
classes 3 (35 samples) and 4 (13) are too sparse to grade reliably. Accuracy is the
wrong metric to lead with on imbalanced data.

---

### 3. Binary — single hold-out split (`binary.py`)

Target: `class_attbr > 0` → 0 = no disease, 1 = disease.
Output: `Dense(2, softmax)` + sparse categorical cross-entropy.

```
              precision  recall    f1     support
   0           0.8235   0.8750   0.8485     16
   1           0.8462   0.7857   0.8148     14
   accuracy                      0.8333     30
   macro avg   0.8348   0.8304   0.8316     30
   weighted    0.8341   0.8333   0.8328     30
```

Train 88.28% | Val 83.33% | Test 83.33%

> **Note:** still a single 30-sample test set — indicative, not exact. Both classes
> perform well (F1 0.85 and 0.81); unlike multiclass, no class collapses. See section 4
> for the cross-validated version, directly comparable to section 2.

---

### 4. Binary — 5-fold stratified cross-validation (`cross_val.py`, `MODE="binary"`)

297 patients, each tested exactly once (out-of-fold).

| Fold | Accuracy | Balanced acc | Macro-F1 |
|------|----------|--------------|----------|
| 1 | 88.33% | 88.62% | 0.883 |
| 2 | 80.00% | 79.91% | 0.799 |
| 3 | 74.58% | 74.54% | 0.745 |
| 4 | 79.66% | 79.22% | 0.794 |
| 5 | 81.36% | 81.37% | 0.813 |

| Metric | NN model | Majority baseline |
|--------|----------|-------------------|
| Accuracy | 80.79% ± 4.42% | 53.88% ± 0.44% |
| Balanced accuracy | 80.73% ± 4.56% | 50.00% ± 0.00% |
| Macro F1 | 0.807 ± 0.045 | 0.350 ± 0.002 |

Aggregated out-of-fold classification report (all 297 patients):

```
              precision  recall    f1     support
   0           0.8280   0.8125   0.8202    160
   1           0.7857   0.8029   0.7942    137
   accuracy                      0.8081    297
   macro avg   0.8069   0.8077   0.8072    297
   weighted    0.8085   0.8081   0.8082    297
```

**Reading:** the model beats the baseline on *every* metric by a wide margin
(accuracy 81% vs 54%, balanced accuracy 81% vs 50%, macro-F1 0.81 vs 0.35), and all
three model metrics sit together at ~0.81 — both classes predicted well, no collapsed
class. Strong, trustworthy performance; the data clearly supports this task.

---

### Headline comparison — cross-validation vs cross-validation

Both rows come from `cross_val.py` using the same 5-fold method (`MODE` switch), so
they are directly comparable.

| Task | Accuracy | Balanced acc | Macro-F1 | Baseline acc |
|------|----------|--------------|----------|--------------|
| Binary (2-class) | 80.79% ± 4.42% | 80.73% | 0.807 | 53.88% |
| Multiclass (5-class) | 51.21% ± 5.36% | 30.87% | 0.303 | 53.88% |

- **Binary** beats the baseline on every metric → real, strong, balanced signal.
- **Multiclass** falls below baseline on raw accuracy but runs ~1.5–2× the baseline on
  the imbalance-aware metrics → it learns genuine signal but cannot reliably grade
  severity; classes 3 and 4 (35 and 13 samples) are too sparse.

**Conclusion:** the Cleveland dataset is rich enough for binary screening but too
small and imbalanced for trustworthy 5-class severity grading. The value of the project
is the honest, leakage-free evaluation that makes this visible — not a single headline
accuracy figure.

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