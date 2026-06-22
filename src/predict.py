import os
import sys
import joblib
import numpy as np
import pandas as pd
from keras.models import load_model

# resolve all relative paths from the project root, regardless of where this runs
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 13 features IN THE SAME ORDER used during training
FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal"]

# human-readable labels for the multiclass severity target
SEVERITY = {
    0: "No disease",
    1: "Disease - level 1",
    2: "Disease - level 2",
    3: "Disease - level 3",
    4: "Disease - level 4",
}

# ---------------------------------------------------------------------------
# NEW PATIENT - edit these 13 values to test a different person.
# (order does not matter here since it is a dict, but every key must be present)
# ---------------------------------------------------------------------------
new_patient = {
    "age": 28, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1,
    "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3,
    "ca": 0, "thal": 6,
}


X_new = pd.DataFrame([new_patient], columns=FEATURES)

def load_pair(model_path, scaler_path):
    """Load a (model, scaler) pair, or return (None, None) if not yet trained."""
    if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
        return None, None
    return load_model(model_path), joblib.load(scaler_path)


# load both trained models (produced by binary.py and multiclass.py)
bin_model, bin_scaler = load_pair("./models/binary_final_model.keras",
                                  "./scaler/binary_scaler.save")
mc_model, mc_scaler = load_pair("./models/multiclass_final_model.keras",
                                "./scaler/multiclass_scaler.save")

if bin_model is None and mc_model is None:
    sys.exit(
        "No trained models found.\n"
        "Run these once to create them, then re-run this script:\n"
        "  python src/binary.py\n"
        "  python src/multiclass.py"
    )

print("\n=== Prediction for new patient ===")
print(X_new.to_string(index=False))
print()

# ---- Binary screening (the reliable result) ----
if bin_model is not None:
    p = bin_model.predict(bin_scaler.transform(X_new), verbose=0)[0]
    label = "Disease present" if np.argmax(p) == 1 else "No disease"
    print(f"Binary screening  : {label}  (confidence {p.max() * 100:.1f}%)")
else:
    print("Binary screening  : [binary model not found - run src/binary.py]")

# ---- Multiclass severity (indicative only) ----
if mc_model is not None:
    p = mc_model.predict(mc_scaler.transform(X_new), verbose=0)[0]
    cls = int(np.argmax(p))
    probs = ", ".join(f"{i}:{p[i] * 100:.0f}%" for i in range(len(p)))
    print(f"Severity (5-class): {SEVERITY[cls]}  (confidence {p.max() * 100:.1f}%)")
    print(f"                    probabilities -> {probs}")
else:
    print("Severity (5-class): [multiclass model not found - run src/multiclass.py]")

print(
    "\nNote: the binary screener is the trustworthy output (~81% CV accuracy).\n"
    "The 5-class severity is shown for completeness but is NOT reliable on this\n"
    "small / imbalanced dataset (~51% CV accuracy) - treat it as indicative only."
)
