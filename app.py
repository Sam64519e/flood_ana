import streamlit as st
from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load the trained model
model = joblib.load('flood_predictor_model.pkl')

# Create a Flask app
flask_app = Flask(__name__)


@flask_app.route('/predict', methods=['POST'])
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


# Run the Flask app with Streamlit
from streamlit.server.server import Server


def main():
    st.title("Flood Predictor API")
    st.write("This app provides a RESTful API for flood predictions.")

    # Run Flask app
    flask_server = Server.get_current()._app
    flask_server.wsgi_app = flask_app


if __name__ == '__main__':
    main()
