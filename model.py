import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# =========================================================
# CONFIG
# =========================================================
DATASET_PATH = "iot_traffic_large.csv"
MODEL_OUT = "traffic_model.pkl"
SCALER_OUT = "scaler.pkl"
FEATURES_OUT = "feature_names.pkl"

# =========================================================
# LOAD DATASET
# =========================================================
print("Loading dataset...")
df = pd.read_csv(DATASET_PATH)

print("Initial shape:", df.shape)

# =========================================================
# TARGET ENCODING
# =========================================================
traffic_map = {
    "Low Traffic": 0,
    "Medium Traffic": 1,
    "High Traffic": 2,
    "Heavy Traffic": 3
}

df["traffic_status"] = df["traffic_status"].map(traffic_map)

# Drop invalid rows
df = df.dropna(subset=["traffic_status"])

# =========================================================
# FEATURE SELECTION (RAW)
# =========================================================
base_features = [
    "speed(km/h)",
    "Ax(m/s2)", "Ay(m/s2)", "Az(m/s2)",
    "Gx(rad/s)", "Gy(rad/s)", "Gz(rad/s)",
    "front_distance(m)",
    "back_distance(m)",
    "jerk",
    "temperature(°C)",
    "humidity(%)",
    "latitude",
    "longitude"
]

df = df[base_features + ["traffic_status"]]

# =========================================================
# FEATURE ENGINEERING
# =========================================================
df["acc_magnitude"] = np.sqrt(
    df["Ax(m/s2)"]**2 +
    df["Ay(m/s2)"]**2 +
    df["Az(m/s2)"]**2
)

df["gyro_magnitude"] = np.sqrt(
    df["Gx(rad/s)"]**2 +
    df["Gy(rad/s)"]**2 +
    df["Gz(rad/s)"]**2
)

df["speed_distance_ratio"] = (
    df["speed(km/h)"] / (df["front_distance(m)"] + 1)
)

# =========================================================
# FINAL FEATURE SET
# =========================================================
X = df.drop("traffic_status", axis=1)
y = df["traffic_status"]

feature_names = list(X.columns)

# =========================================================
# TRAIN / TEST SPLIT
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================================================
# SCALING
# =========================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================================================
# MODEL TRAINING
# =========================================================
print("Training RandomForest model...")

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_scaled, y_train)

# =========================================================
# EVALUATION
# =========================================================
y_pred = model.predict(X_test_scaled)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# =========================================================
# SAVE ARTIFACTS
# =========================================================
joblib.dump(model, MODEL_OUT)
joblib.dump(scaler, SCALER_OUT)
joblib.dump(feature_names, FEATURES_OUT)

print("\nArtifacts saved:")
print(f"- {MODEL_OUT}")
print(f"- {SCALER_OUT}")
print(f"- {FEATURES_OUT}")
