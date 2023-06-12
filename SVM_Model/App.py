from flask import Flask, request, jsonify
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from sklearn import svm

# Load the trained model
with open('svm_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Create a Flask app
app = Flask(__name__)

# Define an API endpoint for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Extract the 'array' field from the input JSON
        input_array = data['array']

        # Validate the input array length
        if len(input_array) != 9:
            return jsonify({'error': 'Invalid input array length'})

        # Convert alphabet into numeric gender
        if input_array[0] == "male":
            input_array[0] = 1
        else:
            input_array[0] = 0

        # Convert alphabet into numeric pain
        if input_array[2] == "mild":
            input_array[2] = 0
        elif input_array[2] == 'normal':
            input_array[2] = 1
        else:
            input_array[2] = 2

        # Convert alphabet into numeric other symptoms
        for i in range(3, 9):
            if input_array[i] == "y":
                input_array[i] = 1
            else:
                input_array[i] = 0

        # Perform the same preprocessing as the training data (if any)
        # ...

        # Create a DataFrame from the preprocessed input array
        input_df = pd.DataFrame([input_array], columns=['Gender', 'Age', 'Pain Intensity', 'Swelling', 'Diab', 'Hypertensive', 'Cardiac', 'Liver Transplant', 'Kidney Transplant'])

        # Make predictions using the model
        predictions = model.predict(input_df)

        # Convert predictions to a list and return as JSON response
        response = {'predictions': predictions.tolist()}
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)})

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
