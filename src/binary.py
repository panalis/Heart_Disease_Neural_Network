import os
import random
import tensorflow as tf
import pandas as pd
import numpy as np
import keras
import matplotlib.pyplot as plt
import joblib
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras import layers
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from imblearn.over_sampling import SMOTE

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['PYTHONHASHSEED'] = str(SEED)

column_names = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "class_attbr"]

# load and clean data
data = pd.read_csv("./data/processed.cleveland.data", header=None, names=column_names, na_values="?")
clean_data = data.dropna() 

# showcase results
print("Original Data: ", data.shape)
print("Clean Data: ", clean_data.shape)

inputs = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]

# separation x=input and y=target 
X = clean_data[inputs]
y = (clean_data["class_attbr"] > 0).astype(int)    # binary classification

# train/validate/test split BEFORE scaling
X_train_raw, X_temp_raw, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_val_raw, X_test_raw, y_val, y_test = train_test_split(X_temp_raw, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

# apply SMOTE only on training data
sm = SMOTE(random_state=42, k_neighbors=3)
X_train_raw_sm, y_train_sm = sm.fit_resample(X_train_raw, y_train)

print("Before SMOTE:", np.bincount(y_train))
print("After SMOTE:", np.bincount(y_train_sm))

# scaling AFTER split — fit only on training data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw_sm)
X_val   = scaler.transform(X_val_raw)
X_test  = scaler.transform(X_test_raw)

print("Training shape: \n", X_train.shape)
print("Validation shape: \n", X_val.shape)
print("Test shape: \n", X_test.shape)


# neural network modeling
model = keras.Sequential([

        layers.InputLayer(shape=(X_train.shape[1],), name="input_layer"),

        layers.Dense(18, activation="relu", name="hidden_layer1", kernel_initializer="he_normal", kernel_regularizer=keras.regularizers.l2(0.01)),
        layers.Dropout(0.2), 
        #layers.BatchNormalization(),
        layers.Dense(16, activation="relu", name="hidden_layer2", kernel_initializer="he_normal"),

        layers.Dense(2, activation="softmax", name="output_layer")
    ])

model.compile(
    optimizer = keras.optimizers.Adam(learning_rate = 0.001), # adam adapts learning rate
    loss = "sparse_categorical_crossentropy",                 # categorical crossentropy (0-1)
    metrics = ["accuracy"]                                    # metric for classification
)

print(model.summary())

early = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True, verbose=1)
checkpoint = ModelCheckpoint("./models/binary_best_model.keras", monitor="val_loss", save_best_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, verbose=1)
callbacks = [early, checkpoint, reduce_lr]

history = model.fit(
    X_train, y_train_sm,
    validation_data=(X_val, y_val),
    epochs=300,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

train_loss = history.history["loss"][-1]
train_acc  = history.history["accuracy"][-1]
val_loss   = history.history["val_loss"][-1]
val_acc    = history.history["val_accuracy"][-1]

print(f"Final train loss: {train_loss:.4f}, train acc: {train_acc:.4f}")
print(f"Final val   loss: {val_loss:.4f}, val   acc: {val_acc:.4f}")

# Accuracy
plt.figure(figsize=(10,4))
plt.plot(history.history["accuracy"], label="train_acc")
plt.plot(history.history["val_accuracy"], label="val_acc")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# Loss
plt.figure(figsize=(10,4))
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f} ({test_acc * 100:.2f}%)")

# Predictions
y_probs = model.predict(X_test)             # probabilities for each class
y_pred = np.argmax(y_probs, axis=1)         # choose the class with highest probability

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix (Test set)")
plt.show()

# Classification report
report = classification_report(y_test, y_pred, digits=4, zero_division=0)
print("Classification report:\n", report)

# Save the final model and scaler
model.save("./models/binary_final_model.keras", include_optimizer=False)
joblib.dump(scaler, "./scaler/binary_scaler.save")

# FINAL SUMMARY (all accuracies in %)

# Train & validation accuracy (last epoch)
train_acc_pct = train_acc * 100
val_acc_pct = val_acc * 100

# Test accuracy %
test_acc_pct = test_acc * 100

# Classification report overall accuracy
overall_acc_pct = accuracy_score(y_test, y_pred) * 100

print("\n================ FINAL SUMMARY ================")
print(f"Train Accuracy:          {train_acc_pct:.2f}%")
print(f"Validation Accuracy:     {val_acc_pct:.2f}%")
print(f"Test Accuracy:           {test_acc_pct:.2f}%")
print(f"Overall Accuracy (CR):   {overall_acc_pct:.2f}%")
print("================================================\n")



