from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load your trained model
model = joblib.load('random_forest_model.joblib')  # Replace with your filename

@app.route('/', methods=['GET'])
def home():
    return "ML API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    try:
        temp = float(data['temperature'])
        humidity = float(data['humidity'])
        gas = int(data['gas'])

        features = np.array([[temp, humidity, gas]])
        prediction = model.predict(features)[0]

        return jsonify({
            'status': 'success',
            'prediction': prediction
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
