from flask import Flask, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route("/")
def home():
    return "Welcome to the Parkinson Disease Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)

    # Extract feature values in order
    features = [
        "mdvp_fo_hz", "mdvp_fhi_hz", "mdvp_flo_hz", 
        "mdvp_jitter_percent", "mdvp_jitter_abs", 
        "mdvp_rap", "mdvp_ppq", "jitter_ddp", 
        "mdvp_shimmer", "mdvp_shimmer_db", 
        "shimmer_apq3", "shimmer_apq5", 
        "mdvp_apq", "shimmer_dda", 
        "nhr", "hnr", 
        "rpde", "dfa", "spread1", "spread2", 
        "d2", "ppe"
    ]

    input_data = [data[feature] for feature in features]
    input_array = np.array(input_data).reshape(1, -1)
    std_data = scaler.transform(input_array)
    prediction = model.predict(std_data)

    result = {
        "prediction": int(prediction[0]),
        "message": "The person has Parkinson's" if prediction[0] else "The person does not have Parkinson's"
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
