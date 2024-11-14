from Pipeline import extract_features
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Configs
model_path = './model/modelv3.keras'
model = tf.keras.models.load_model(model_path)
label_classes = ['Bronchial', 'asthma', 'copd', 'healthy', 'pneumonia']

@app.route("/predict-audio", methods=['POST'])
def predict_audio():
  data = request.get_json()
    
  if 'audio_path' not in data:
    return jsonify({
        "status": 400,
        "message": "Bad Request, No Audio Path Provided",
        "err": {
            "data": {
              "code": -1
            }
        }
    }), 400
    
  audio_path = data['audio_path']
  features = extract_features(audio_path)
    
  if features is not None:
    features = np.expand_dims(features, axis=(0, 2))
    prediction = model.predict(features)
    prediction_probabilities = {label.lower(): float(prob) for label, prob in zip(label_classes, prediction[0])}

    return jsonify({
      "status": 200,
      "message": "OK",
      "data": prediction_probabilities
    })
  else:

    return jsonify({
        "status": 500,
        "message": "Something Wrong with model",
        "err": {
            "data": {
              "code": -2
            }
        }
    }), 500

app.run(host='0.0.0.0', port=10110)