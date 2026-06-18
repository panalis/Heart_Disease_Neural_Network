import os
import numpy as np
import random
import tensorflow as tf
import pandas as pd, keras

from keras import layers
from keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['PYTHONHASHSEED'] = str(SEED)

MODE = "multiclass" # "multiclass" or "binary"

column_names = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "class_attbr"]

df = pd.read_csv("./data/processed.cleveland.data", header=None, names=column_names, na_values="?").dropna()

inputs = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]

X = df[inputs].values

if MODE == "multiclass":
    y = df["class_attbr"].astype(int).values
    n_classes = 5
else:
    y = (df["class_attbr"] > 0).astype(int).values
    n_classes = 2



def build(n_features, h1=18, h2=16, dropout=0.2):

    model = keras.Sequential([

        layers.InputLayer(shape=(n_features,)),

        layers.Dense(h1, activation="relu", kernel_initializer="he_normal", kernel_regularizer=keras.regularizers.l2(0.01)),
        layers.Dropout(dropout),
        layers.Dense(h2, activation="relu", kernel_initializer="he_normal"),

        layers.Dense(n_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"])
    return model

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# per-fold metric stores (NN model + majority-class baseline)
model_acc, model_bal, model_f1 = [], [], []
base_acc,  base_bal,  base_f1  = [], [], []

# out-of-fold predictions: every sample is in exactly one test fold,
# so collecting them gives one prediction per patient for an aggregate report
oof_true, oof_pred = [], []

for k, (train, test) in enumerate(skf.split(X, y), 1):

    Xtrain_raw, Xtest = X[train], X[test]
    ytrain_raw, ytest = y[train], y[test]

    # majority-class baseline: fit on the REAL training distribution (before SMOTE)
    dummy = DummyClassifier(strategy="most_frequent").fit(Xtrain_raw, ytrain_raw)
    ybase = dummy.predict(Xtest)

    # model pipeline: SMOTE + scaling, both fit on the train fold only
    Xtrain, ytrain = SMOTE(random_state=42, k_neighbors=3).fit_resample(Xtrain_raw, ytrain_raw)
    sc = StandardScaler().fit(Xtrain)
    Xtrain, Xtest_scaled = sc.transform(Xtrain), sc.transform(Xtest)

    model = build(Xtrain.shape[1])
    es = EarlyStopping(monitor="loss", patience=20, restore_best_weights=True, verbose=0)
    model.fit(Xtrain, ytrain, epochs=300, batch_size=32, callbacks=[es], verbose=0)
    ymodel = np.argmax(model.predict(Xtest_scaled, verbose=0), axis=1)

    # metrics for this fold
    model_acc.append(accuracy_score(ytest, ymodel))
    model_bal.append(balanced_accuracy_score(ytest, ymodel))
    model_f1.append(f1_score(ytest, ymodel, average="macro", zero_division=0))

    base_acc.append(accuracy_score(ytest, ybase))
    base_bal.append(balanced_accuracy_score(ytest, ybase))
    base_f1.append(f1_score(ytest, ybase, average="macro", zero_division=0))

    oof_true.extend(ytest)
    oof_pred.extend(ymodel)

    print(f"Fold {k}: acc={model_acc[-1]*100:5.2f}%  balanced_acc={model_bal[-1]*100:5.2f}%  macroF1={model_f1[-1]:.3f}")


def pct(v):   # mean ± std, formatted as a percentage
    return f"{np.mean(v)*100:5.2f}% ± {np.std(v)*100:4.2f}%"

def raw(v):   # mean ± std, raw 0-1 score (for F1)
    return f"{np.mean(v):.3f} ± {np.std(v):.3f}"

print(f"\nMODE = {MODE}  ({n_classes} classes)")
print("\n=================== 5-FOLD CROSS-VALIDATION ===================")
print(f"{'Metric':<20}{'NN model':<24}{'Majority baseline'}")
print(f"{'Accuracy':<20}{pct(model_acc):<24}{pct(base_acc)}")
print(f"{'Balanced accuracy':<20}{pct(model_bal):<24}{pct(base_bal)}")
print(f"{'Macro F1':<20}{raw(model_f1):<24}{raw(base_f1)}")
print("===============================================================")

print("\nAggregated out-of-fold classification report (all patients):")
print(classification_report(oof_true, oof_pred, digits=4, zero_division=0))