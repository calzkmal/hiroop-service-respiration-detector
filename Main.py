from Pipeline import FeatureExtractor
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import os
from datetime import datetime

app = Flask(__name__)

# Initialize the FeatureExtractor
extractor = FeatureExtractor()

# Configs
model_path = './model/model.keras'
model = tf.keras.models.load_model(model_path)
label_classes = ['Bronchial', 'asthma', 'copd', 'healthy', 'pneumonia']
ALLOWED_EXTENSIONS = {'wav'}

def allowed_file(filename):
    return '.' in filename and \
      filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/predict-audio", methods=['POST'])
def predict_audio():
  if 'file' not in request.files:

    return jsonify({
        "status": 400,
        "message": "Bad Request, No Audio Provided",
        "err": {
            "data": {
              "code": -1
            }
        }
    }), 400
  
  file = request.files['file']
  
  if file and allowed_file(file.filename):
    filename = datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3] + ".wav"
    file_path = os.path.join('./server_data', filename)
    file.save(file_path)

  features = extractor.get_features(file_path)
    
  if features is not None:
    # Reshape the data to match the model input shape (None, 162, 1)
    features = np.expand_dims(features, axis=(0, 2))

    # Make a prediction
    prediction = model.predict(features)

    # Convert prediction to a dictionary with probabilities and labels
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