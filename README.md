# Heart Disease Neural Network

A feedforward neural network that screens for heart disease and grades its severity on the
UCI Cleveland dataset. The project is built around a pipeline that is free of data leakage
and an honest, cross validated evaluation. It compares two framings of the same problem:
reliable binary screening (disease versus no disease) and the much harder five level
severity grading, then uses proper metrics and a baseline to show why one framing works
and the other does not.

> The headline of this project is not a single accuracy number. It is the finding that the
> classic Cleveland dataset supports binary screening well (about 81% cross validated
> accuracy) but is too small and imbalanced for trustworthy five level severity grading
> (about 51%, which is below the majority baseline). Showing that difference with correct
> methodology is the point.

## The dataset

The [UCI Cleveland Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease)
holds 303 patient records with 13 clinical features (age, sex, chest pain type, resting blood
pressure, cholesterol, and so on). A handful of rows contain missing values marked with `?`
and are dropped, leaving 297 clean records. The target, `class_attbr`, runs from 0 (no disease)
to 4 (most severe).

* **Binary framing:** 0 = no disease, 1 = disease present (any level above 0).
* **Multiclass framing:** the full 0 - 4 severity scale.

The data file is included in the repository for reproducibility and is publicly available from
UCI for research and educational use.

## Methodology

The pipeline is deliberately built to avoid data leakage, the most common way a model quietly
inflates its own scores:

1. **Split first.** A stratified split into training, validation, and test sets (about
   80 / 10 / 10) happens before any scaling or resampling, so the validation and test sets
   never influence preprocessing.
2. **SMOTE on the training set only.** Synthetic minority class samples are generated after the
   split and only on the training data, so no synthetic information leaks into validation or test.
3. **Scaler fit on the training set only.** `StandardScaler` learns its mean and standard
   deviation from the training data alone, then is applied to validation and test.

Every script sets a fixed random seed (42) so results are reproducible.

## Model architecture

The same compact network is used across every script, kept small on purpose for a dataset of
roughly 300 rows:

| Layer | Units | Notes |
|-------|-------|-------|
| Input | 13 | one per clinical feature |
| Dense (hidden 1) | 18 | ReLU, He normal init, L2 regularisation (0.01) |
| Dropout | | rate 0.2 |
| Dense (hidden 2) | 16 | ReLU, He normal init |
| Output | 5 or 2 | softmax (5 for multiclass, 2 for binary) |

* **Loss:** sparse categorical cross entropy
* **Optimizer:** Adam (learning rate 0.001)
* **Callbacks:** EarlyStopping, ModelCheckpoint (best validation loss), ReduceLROnPlateau
* **Training:** up to 300 epochs, batch size 32, stopped early when validation loss plateaus

## Evaluation

A single 30 sample test split is too small to trust. Early experiments swung between 60% and
70% purely on the luck of the draw. The honest evaluation therefore uses **5 fold stratified
cross validation**, where every patient is tested exactly once and results are reported as
mean plus or minus standard deviation. Two safeguards make the numbers meaningful on imbalanced
data:

* A **majority class baseline** (always predict the most common class), so every model number
  has a reference point.
* **Balanced accuracy and macro F1** alongside plain accuracy, because raw accuracy is
  misleading when one class dominates.

## Results

All numbers below come from the same 5 fold cross validation in `cross_val.py` (switched with a
single `MODE` variable), so the two framings are directly comparable. Every patient is tested
exactly once per run, and scores are reported as mean plus or minus standard deviation across
the five folds.

### Headline comparison

| Task | Accuracy | Balanced accuracy | Macro F1 | Baseline accuracy |
|------|----------|-------------------|----------|-------------------|
| Binary (disease versus no disease) | 80.79% ± 4.42% | 80.73% ± 4.56% | 0.807 ± 0.045 | 53.88% |
| Multiclass (five level severity) | 51.21% ± 5.36% | 30.87% ± 5.14% | 0.303 ± 0.050 | 53.88% |

* **Binary screening is strong and trustworthy.** The model beats the majority baseline on every
  metric by a wide margin (accuracy 81% versus 54%, balanced accuracy 81% versus 50%, macro F1
  0.81 versus 0.35).
* **Multiclass severity grading does not work reliably.** Raw accuracy (51%) sits *below* the
  majority baseline (54%), yet the model scores roughly 1.5 to 2 times the baseline on balanced
  accuracy and macro F1. It learns genuine signal but cannot grade severity dependably.

### Why accuracy alone is misleading here

Because every patient is tested exactly once across the five folds, the predictions can be
collected into a single report covering all 297 patients (an aggregated out of fold report).
This is where the gap between the two tasks becomes obvious.

Binary, per class:

```
              precision   recall   f1-score   support
   0            0.8280    0.8125    0.8202      160
   1            0.7857    0.8029    0.7942      137
```

Both classes score around 0.80. Performance is balanced, with no weak class.

Multiclass, per class:

```
              precision   recall   f1-score   support
   0            0.8188    0.7625    0.7896      160
   1            0.2037    0.2037    0.2037       54
   2            0.2093    0.2571    0.2308       35
   3            0.2250    0.2571    0.2400       35
   4            0.0909    0.0769    0.0833       13
```

Class 0 (no disease) is predicted well, but classes 1 through 4 collapse, and class 4 is almost
unlearnable with only 13 patients in the entire dataset. A model that simply predicted "no
disease" for everyone would already reach about 54% accuracy, which is why the raw 51% looks
deceptively close to the baseline. The per class view shows the real picture: the signal needed
for fine grained severity grading is not present in this amount of data.

The fold to fold spread tells the same story. Multiclass accuracy ranged from about 47% to 59%
across the five folds, so a single train and test split could have reported anything from the
high 40s to 70% by luck alone. Reporting the cross validated mean and standard deviation is what
keeps the result honest.

**Conclusion:** the Cleveland dataset is rich enough for binary screening but too small and
imbalanced for trustworthy five level severity grading. The value of the project is the honest,
leakage free evaluation that makes this visible, rather than any single headline accuracy.



## How to run

From the repository root:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Train the models (this also produces the accuracy and loss curves, the confusion matrix, and a
per class report):

```bash
python src/multiclass.py
python src/binary.py
```

Run the cross validated evaluation (open `cross_val.py` and set `MODE` to `"multiclass"` or
`"binary"`):

```bash
python src/cross_val.py
```

Predict on a new patient (edit the `new_patient` values). The model
and scaler files are git ignored, so run the training scripts above at least once before this:

```bash
python src/predict.py
```

## Tech stack

Python, TensorFlow and Keras, scikit-learn, imbalanced-learn (SMOTE), pandas, NumPy, Matplotlib,
seaborn.

## Author

Panagiotis Christofilopoulos
[GitHub](https://github.com/panalis) · [LinkedIn](https://linkedin.com/in/panagiotis-christofilopoulos-7715a2357)
