from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('flood_predictor_model.pkl')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)

    # Extract features from the incoming JSON request
    avg_temp = data['Avg_Temp']
    rainfall = data['Rainfall']
    humidity = data['Relative_Humidity']
    wind_speed = data['Wind_Speed']

    # Create a feature array for prediction
    features = np.array([[avg_temp, rainfall, humidity, wind_speed]])

    # Make a prediction
    prediction = model.predict(features)

    # Return the prediction as JSON
    return jsonify({'prediction': int(prediction[0])})


if __name__ == '__main__':
    app.run(debug=True)
