from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import pandas as pd

app = Flask(__name__)

# Load the saved model
model = tf.keras.models.load_model(r"C:\Life Projects\nasa space app challenge\asteroids-diameter\modell.h5")

def predict_diameter(input_dict):
    # Create a DataFrame from the input dictionary
    df = pd.DataFrame([input_dict])
    
    # Perform one-hot encoding for categorical variables
    categorical_columns = ['condition_code', 'neo', 'pha', 'class']
    df_encoded = pd.get_dummies(df, columns=categorical_columns)
    
    # Ensure all expected columns are present
    expected_columns = ['a', 'e', 'i', 'om', 'w', 'q', 'ad', 'per_y', 'data_arc', 'n_obs_used',
       'H','albedo', 'moid', 'n', 'ma', 'approx_diameter',
       'condition_code_0', 'condition_code_1', 'condition_code_2',
       'condition_code_3', 'condition_code_4', 'condition_code_5',
       'condition_code_6', 'condition_code_7', 'condition_code_8',
       'condition_code_9', 'neo_N', 'neo_Y', 'pha_N', 'pha_Y', 'class_AMO',
       'class_APO', 'class_AST', 'class_ATE', 'class_CEN', 'class_IMB',
       'class_MBA', 'class_MCA', 'class_OMB', 'class_TJN', 'class_TNO']
    for col in expected_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    
    # Ensure correct order of columns
    df_encoded = df_encoded[expected_columns]
    
    # Convert DataFrame to numpy array and then to float32
    input_data = df_encoded.to_numpy().astype('float32')
    
    # Make the prediction
    prediction = model.predict(input_data)
    
    # Return the predicted diameter
    return prediction[0][0]

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request
    input_data = request.json
    
    # Ensure all required fields are present
    required_fields = ['a', 'e', 'i', 'om', 'w', 'q', 'ad', 'per_y', 'data_arc', 'n_obs_used',
                       'H', 'albedo', 'moid', 'n', 'ma', 'approx_diameter',
                       'condition_code', 'neo', 'pha', 'class']
    
    for field in required_fields:
        if field not in input_data:
            return jsonify({'error': f'Missing required field: {field}'}), 400
    
    # Make the prediction
    try:
        predicted_diameter = predict_diameter(input_data)
        return jsonify({'predicted_diameter': float(predicted_diameter)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)