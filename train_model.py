import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Sample: Create your own dataset here or replace this
data = pd.DataFrame([
    {"bpm": 70, "spo2": 98, "status": "stable"},
    {"bpm": 85, "spo2": 95, "status": "stable"},
    {"bpm": 100, "spo2": 93, "status": "at_risk"},
    {"bpm": 120, "spo2": 88, "status": "danger"},
    {"bpm": 130, "spo2": 85, "status": "danger"},
    {"bpm": 95, "spo2": 90, "status": "at_risk"},
])

# Add deviation features if needed
data["bpm_delta"] = data["bpm"] - 80
data["spo2_delta"] = data["spo2"] - 97

X = data[["bpm", "spo2", "bpm_delta", "spo2_delta"]]
y = data["status"]

# Train/test split (optional for validation)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "ria_model_v2.pkl")
print("✅ Model saved as ria_model_v2.pkl")
