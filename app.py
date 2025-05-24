import json
import pickle
import numpy as np 
import pandas as pd
from flask import Flask, request, jsonify, render_template

# Initialize the Flask application
app = Flask(__name__)

# Load the pre-trained model and scaler from pickle files
with open('XGB_pkl', 'rb') as model_file:
    pickled_model = pickle.load(model_file)

with open('scaling.pkl', 'rb') as scaler_file:
    scalar = pickle.load(scaler_file)

@app.route('/')
def home():
    """Render the home page."""
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    """API endpoint for making predictions based on JSON input."""
    data = request.json['data']
    
    # Convert input data to a NumPy array and reshape it
    input_data = np.array(list(data.values())).reshape(1, -1)
    print("Input data:", input_data)
    
    # Scale the input data
    scaled_data = scalar.transform(input_data)
    
    # Make prediction using the loaded model
    output = pickled_model.predict(scaled_data)
    print("Prediction output:", output[0])
    
    # Return the prediction as a JSON response
    return jsonify(output[0])

@app.route('/predict', methods=['POST'])
def predict():
    """Handle form submissions and return predictions."""
    # Extract data from the form and convert to float
    data = [float(x) for x in request.form.values()]
    
    # Scale the input data
    final_input = scalar.transform(np.array(data).reshape(1, -1))
    print("Scaled input data:", final_input)
    
    # Make prediction using the loaded model
    output = pickled_model.predict(final_input)[0]
    
    # Render the home page with the prediction result
    return render_template("home.html", prediction_text=f"The shear capacity is {output} KN")

if __name__ == "__main__":
    # Run the application in debug mode
    app.run(debug=True)



