from flask import Flask, request, jsonify
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from sklearn.tree import DecisionTreeClassifier

# Load the trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Create a Flask app
app = Flask(__name__)

# Define an API endpoint for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Extract the 'array' field from the input JSON
    input_array = data['array']
    #convert alphabet into numeric gender
    if(input_array[0] == "male"):
        input_array[0] = 1
    else:
        input_array[0] = 0


    #convert alphabet into numeric pain
    if(input_array[2] == "mild"):
        input_array[0] = 0
    elif(input_array[2] == 'normal'):
        input_array[2] = 1
    else:
        input_array[2] = 2


    #convert alphabet into numeric other symptoms
    for i in range(3, 9):
        if(input_array[i] == "y"):
            input_array[i] = 1
        else:
            input_array[i] = 0
    print(input_array)
    # Create a DataFrame from the input array
    
    predictions = model.predict([input_array])

    # Convert predictions to a list and return as JSON response
    response = {
        'predictions': predictions.tolist()
    }
    return jsonify(response)

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000)
