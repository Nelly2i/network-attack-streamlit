#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

# Loading the trained model
model = load_model('Leslie_network_attack_model.h5')

# Defining the selected feature names
top_10_features = ['dst host srv diff host rate', 'same srv rate', 'dst host same srv rate', 'count', 'dst host count',
                   'dst host same src port rate', 'diff srv rate', 'service_eco_i', 'src bytes', 'dst host diff srv rate']

# Streamlit app title
st.title("Network Attack Prediction App")

# Uploading dataset for prediction
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"], key='unique_uploader')

# Checking if a file is uploaded
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # One-hot encoding the 'service' column if it exists
    if 'service' in data.columns:
        service_encoded = pd.get_dummies(data['service'], prefix='service')

        # Remove the original 'service' column and add the one-hot encoded columns
        data = data.drop('service', axis=1)
        data = pd.concat([data, service_encoded], axis=1)

    # Displaying the first few rows of the uploaded data
    st.write("Uploaded Dataset Preview:")
    st.write(data.head())

    # Selecting only the feature columns from the uploaded dataset
    if set(top_10_features).issubset(data.columns):
        filtered_data = data[top_10_features]
        st.write("Filtered Data (Only Selected Features):")
        st.write(filtered_data.head())

        # Button to trigger prediction
        if st.button("Predict"):
            try:
                # Checking for non-numeric values and encoding them if necessary
                for col in filtered_data.columns:
                    if filtered_data[col].dtype == 'object':  # If the column is categorical
                        st.write(f"Encoding column: {col}")
                        filtered_data[col] = filtered_data[col].astype('category').cat.codes

                # Converting the filtered data to NumPy array and reshape for prediction
                data_for_prediction = np.array(filtered_data).astype(np.float32)

                # Reshaping the data to fit the model's expected input shape
                data_for_prediction = data_for_prediction.reshape(-1, 1, len(top_10_features))

                # Making predictions
                predictions = model.predict(data_for_prediction)

                # Converting predictions to class labels
                predicted_classes = np.argmax(predictions, axis=1)

                # Define the possible attack classes based on your model
                attack_classes = ['normal', 'back', 'ipsweep', 'portsweep', 'satan']

                # Mapping predicted class numbers to attack types
                predicted_attacks = [attack_classes[i] for i in predicted_classes]

                # Create a DataFrame with the Id and predicted attack types
                output_df = pd.DataFrame({
                    'Id': data.index + 1,  # Assuming the Id should just be the row number + 1
                    'type_of_attack': predicted_attacks
                })

                # Display the output as a table on the app
                st.write("Predicted Attack Types:")
                st.write(output_df)

                # Optionally: Allow users to download the predictions as a CSV
                st.download_button(label="Download Predictions as CSV", data=output_df.to_csv(index=False), mime='text/csv')

            except KeyError as e:
                st.error(f'Error: {e}. Please ensure the dataset contains the necessary features.')
            except Exception as e:
                st.error(f'An error occurred: {e}')

# Correlation Heatmap Option
if st.button("Show Correlation Heatmap"):
    correlation_matrix = filtered_data.corr()
    fig = px.imshow(correlation_matrix, text_auto=True)
    st.plotly_chart(fig)
