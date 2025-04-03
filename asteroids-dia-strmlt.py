import numpy as np
import tensorflow as tf
import pandas as pd
import streamlit as st

# Load the saved model
try:
    model = tf.keras.models.load_model(r"modell.h5")
    model_loaded = True
except Exception as e:
    st.error(f"Error loading the model: {str(e)}")
    st.error("Please make sure the model file 'modell.h5' is present and not corrupted.")
    model_loaded = False

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

# Streamlit UI
st.title("Asteroid Diameter Prediction")

# Default values
default_values = {
    'a': 2.76916515450648, 'e': 0.07600902910070946, 'i': 10.59406704424526, 'om': 80.30553156826473, 'w': 73.597694115971,
    'q': 2.558683599692926, 'ad': 2.979646709320033, 'per_y': 4.60820180153985, 'data_arc': 8822, 'n_obs_used': 1002,
    'H': 3.34, 'albedo': 0.090, 'moid': 1.59478, 'n': 0.213885225911375, 'ma': 77.37209588584763, 'approx_diameter': 950,
    'condition_code': '0', 'neo': 'N', 'pha': 'N', 'class': 'MBA'
}

# Create input fields for numerical features
numerical_features = ['a', 'e', 'i', 'om', 'w', 'q', 'ad', 'per_y', 'data_arc', 'n_obs_used',
                      'H', 'albedo', 'moid', 'n', 'ma', 'approx_diameter']

input_data = {}
for feature in numerical_features:
    input_data[feature] = st.number_input(f"Enter {feature}", value=default_values[feature], format="%.6f")

# Create dropdown for categorical features
condition_codes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
input_data['condition_code'] = st.selectbox("Select condition code", condition_codes, index=condition_codes.index(default_values['condition_code']))

input_data['neo'] = st.selectbox("Is it a Near Earth Object (NEO)?", ['N', 'Y'], index=['N', 'Y'].index(default_values['neo']))
input_data['pha'] = st.selectbox("Is it a Potentially Hazardous Asteroid (PHA)?", ['N', 'Y'], index=['N', 'Y'].index(default_values['pha']))

asteroid_classes = ['AMO', 'APO', 'AST', 'ATE', 'CEN', 'IMB', 'MBA', 'MCA', 'OMB', 'TJN', 'TNO']
input_data['class'] = st.selectbox("Select asteroid class", asteroid_classes, index=asteroid_classes.index(default_values['class']))

if st.button("Predict Diameter"):
    if model_loaded:
        predicted_diameter = predict_diameter(input_data)
        st.success(f"Predicted diameter: {predicted_diameter:.2f} km")
    else:
        st.error("Cannot make predictions. The model failed to load.")

# Add some information about the features
st.sidebar.title("Feature Information")
st.sidebar.write("""
- a: Semi-major axis
- e: Eccentricity
- i: Inclination
- om: Longitude of the ascending node
- w: Argument of perihelion
- q: Perihelion distance
- ad: Aphelion distance
- per_y: Orbital period (years)
- data_arc: Data arc span
- n_obs_used: Number of observations used
- H: Absolute magnitude
- albedo: Albedo
- moid: Minimum orbit intersection distance
- n: Mean motion
- ma: Mean anomaly
- approx_diameter: Approximate diameter
""")