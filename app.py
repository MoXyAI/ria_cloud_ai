from flask import Flask, request, jsonify
import joblib
import numpy as np
import firebase_admin
from firebase_admin import credentials, db

# Init Flask app
app = Flask(__name__)

# Load model
model = joblib.load('ria_model_v2.pkl')  # Trained with bpm, spo2, and optionally more

# Firebase setup (optional for baseline)
cred = credentials.Certificate('credentials_austin.json')
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://your-firebase-url.firebaseio.com'
})

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        bpm = data.get('bpm')
        spo2 = data.get('spo2')
        user_id = data.get('userId')

        # Optionally pull user baseline from Firebase
        ref = db.reference(f'/{user_id}/baseline')
        baseline = ref.get() or {'avg_bpm': 75, 'avg_spo2': 97}

        bpm_delta = bpm - baseline['avg_bpm']
        spo2_delta = spo2 - baseline['avg_spo2']

        # Prediction input
        X = np.array([[bpm, spo2, bpm_delta, spo2_delta]])
        prediction = model.predict(X)[0]
        confidence = max(model.predict_proba(X)[0])

        return jsonify({
            'status': prediction,
            'confidence': round(confidence, 2)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the app
if __name__ == '__main__':
    app.run(debug=True)

