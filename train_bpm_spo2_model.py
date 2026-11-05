import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# ---------------------------
# 1️⃣ CREATE / LOAD DATASET
# ---------------------------

# Synthetic dataset (realistic ranges)
np.random.seed(42)
n_samples = 1200

# Heart Rate (BPM)
bpm = np.random.normal(loc=75, scale=12, size=n_samples)

# Oxygen Saturation (SpO2)
spo2 = np.random.normal(loc=96, scale=2.5, size=n_samples)

# Label logic (medical-inspired)
labels = []
for b, s in zip(bpm, spo2):
    if s < 92 or b < 50 or b > 110:
        labels.append(1)   # APNEA
    else:
        labels.append(0)   # NORMAL

labels = np.array(labels)

# Feature matrix
X = np.column_stack((bpm, spo2))
y = labels

# ---------------------------
# 2️⃣ TRAIN / TEST SPLIT
# ---------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# ---------------------------
# 3️⃣ TRAIN MODEL
# ---------------------------

model = RandomForestClassifier(
    n_estimators=120,
    max_depth=6,
    random_state=42
)

model.fit(X_train, y_train)

# ---------------------------
# 4️⃣ EVALUATION
# ---------------------------

y_pred = model.predict(X_test)

print("\n✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))
print("\n🧮 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ---------------------------
# 5️⃣ SAVE MODEL
# ---------------------------

joblib.dump(model, "apnea_model.pkl")
print("\n💾 Model saved as apnea_model.pkl")
