from flask import Flask, request, jsonify
import numpy as np
import joblib
import firebase_admin
from firebase_admin import credentials, db
import os
import json

app = Flask(__name__)

# ✅ Load Firebase credentials from ENV
firebase_creds = os.environ.get("FIREBASE_CREDENTIALS")
cred = credentials.Certificate(json.loads(firebase_creds))
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://revixapp-ecd88-default-rtdb.firebaseio.com/'
})

# ✅ Load the model
model = joblib.load("ria_model_v2.pkl")

@app.route('/')
def home():
    return "✅ RIA Backend is running"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    bpm = data['bpm']
    spo2 = data['spo2']
    user_id = data['userId']

    features = np.array([[bpm, spo2]])
    prediction = model.predict(features)[0]
    confidence = model.predict_proba(features).max()

    # Log to Firebase
    ref = db.reference(f'cloud_logs/{user_id}')
    ref.push({
        'bpm': bpm,
        'spo2': spo2,
        'prediction': prediction,
        'confidence': round(float(confidence), 4)
    })

    return jsonify({
        'status': prediction,
        'confidence': round(float(confidence), 4)
    })

if __name__ == '__main__':
    app.run(debug=True)
