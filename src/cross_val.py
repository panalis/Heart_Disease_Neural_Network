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

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['PYTHONHASHSEED'] = str(SEED)

column_names = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "class_attbr"]

df = pd.read_csv("./data/processed.cleveland.data", header=None, names=column_names, na_values="?").dropna()

inputs = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]

X = df[inputs].values
y = df["class_attbr"].astype(int).values

def build(n_features, h1=18, h2=16, dropout=0.2):

    model = keras.Sequential([

        layers.InputLayer(input_shape=(n_features,)),

        layers.Dense(h1, activation="relu", kernel_initializer="he_normal", kernel_regularizer=keras.regularizers.l2(0.01)),
        layers.Dropout(dropout),
        layers.Dense(h2, activation="relu", kernel_initializer="he_normal"),

        layers.Dense(5, activation="softmax"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss="sparse_categorical_crossentropy", 
        metrics=["accuracy"])
    return model

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
accs = []

for k, (train, test) in enumerate(skf.split(X, y), 1):

    Xtrain, Xtest, ytrain, ytest = X[train], X[test], y[train], y[test]
    Xtrain, ytrain = SMOTE(random_state=42, k_neighbors=3).fit_resample(Xtrain, ytrain)   # train fold only
    
    sc = StandardScaler().fit(Xtrain)  # fit on train fold only

    Xtrain, Xtest = sc.transform(Xtrain), sc.transform(Xtest)

    model = build(Xtrain.shape[1])
    es = EarlyStopping(monitor="loss", patience=20, restore_best_weights=True, verbose=0)
    model.fit(Xtrain, ytrain, epochs=300, batch_size=32, callbacks=[es], verbose=0)
    acc = model.evaluate(Xtest, ytest, verbose=0)[1]
    accs.append(acc); print(f"Fold {k}: {acc*100:.2f}%")

print(f"\nCV accuracy: {np.mean(accs)*100:.2f}% ± {np.std(accs)*100:.2f}%")