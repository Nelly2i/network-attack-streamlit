#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import plotly.express as px

# Loading the trained model
model = load_model('Leslie_network_attack_model.h5')

# Defining the selected feature names
top_10_features = [
    'dst host srv diff host rate', 'same srv rate', 'dst host same srv rate', 
    'count', 'dst host count', 'dst host same src port rate', 'diff srv rate', 
    'service_eco_i', 'src bytes', 'dst host diff srv rate'
]

# Streamlit app title
st.title("Network Attack Prediction App")

# Uploading dataset for prediction
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"], key='unique_uploader')

# Checking if a file is uploaded
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    try:
        # One-hot encoding the 'service' column
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
                    # Checking for non-numeric values and encode them if necessary
                    for col in filtered_data.columns:
                        if filtered_data[col].dtype == 'object':  # If the column is categorical
                            st.write(f"Encoding column: {col}")
                            filtered_data[col] = filtered_data[col].astype('category').cat.codes

                    # Converting the filtered data to NumPy array and reshape for prediction
                    data_for_prediction = np.array(filtered_data).astype(np.float32)

                    # Reshaping the data to match the input shape of the model (batch_size, sequence_length, num_features)
                    data_for_prediction = data_for_prediction.reshape(-1, 1, len(top_10_features))

                    # Making predictions
                    predictions = model.predict(data_for_prediction)

                    # Mapping prediction to attack types
                    attack_mapping = {0: 'ipsweep', 1: 'satan', 2: 'portsweep', 3: 'back', 4: 'normal'}
                    predicted_classes = np.argmax(predictions, axis=1)
                    predicted_attacks = [attack_mapping[cls] for cls in predicted_classes]

                    # Creating DataFrame for results
                    results = pd.DataFrame({
                        'Id': data.index,
                        'type_of_attack': predicted_attacks
                    })

                    # Display the results
                    st.write("Prediction Results:")
                    st.write(results)

                except KeyError as e:
                    st.error(f'Error: {e}. Please ensure the dataset contains the necessary features.')
                except Exception as e:
                    st.error(f'An error occurred: {e}')

        else:
            st.error("The uploaded dataset does not contain the required feature columns.")

    except Exception as e:
        st.error(f"An error occurred while processing the data: {e}")

# Correlation Heatmap using Plotly
if st.button("Show Correlation Heatmap"):
    try:
        # Select only numeric columns for correlation
        numeric_columns = data.select_dtypes(include=[np.number])

        # Compute the correlation matrix
        correlation_matrix = numeric_columns.corr()

        # Plotting the correlation heatmap using Plotly
        fig = px.imshow(correlation_matrix, text_auto=True)
        st.plotly_chart(fig)

    except Exception as e:
        st.error(f"An error occurred while generating the heatmap: {e
