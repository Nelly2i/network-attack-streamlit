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

    # One-hot encoding the 'service' column if it exists in the dataset
    if 'service' in data.columns:
        service_encoded = pd.get_dummies(data['service'], prefix='service')

        # Remove the original 'service' column and add the one-hot encoded columns
        data = data.drop('service', axis=1)
        data = pd.concat([data, service_encoded], axis=1)

    # Displaying the first few rows of the uploaded data
    st.write("Uploaded Dataset Preview:")
    st.write(data.head())

    # Selecting only the top 10 feature columns from the uploaded dataset
    if set(top_10_features).issubset(data.columns):
        filtered_data = data[top_10_features]
        st.write("Filtered Data (Only Selected Features):")
        st.write(filtered_data.head())
    else:
        st.error("The dataset does not contain the required top 10 feature columns.")

    # Button to trigger prediction
    if st.button("Predict"):
        try:
            # Checking for non-numeric values and encoding them if necessary
            for col in filtered_data.columns:
                if filtered_data[col].dtype == 'object':  # If the column is categorical
                    st.write(f"Encoding column: {col}")
                    filtered_data[col] = filtered_data[col].astype('category').cat.codes

            # Converting the filtered data to a NumPy array for prediction
            data_for_prediction = np.array(filtered_data).astype(np.float32)

            # Reshape the data for the model: (batch_size, 1, number_of_features)
            data_for_prediction = data_for_prediction.reshape(data_for_predicition.shape[0], 1, data_for_prediction.shape[1])

            # Making predictions using the model
            predictions = model.predict(data_for_prediction)

            # Converting predictions to class labels
            predicted_classes = np.argmax(predictions, axis=1)

            # Displaying predictions
            st.write(f'Predicted Classes: {predicted_classes}')
        
        except KeyError as e:
            st.error(f"Error: {e}. Please ensure the dataset contains the necessary features.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

# Optional: Plot Correlation Heatmap
import plotly.express as px

if st.button("Show Correlation Heatmap"):
    correlation_matrix = data.corr()
    fig = px.imshow(correlation_matrix, text_auto=True)
    st.plotly_chart(fig)
