#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import plotly.express as px
import plotly.graph_objs as go

# Load the trained model
model = load_model('Leslie_network_attack_model.h5')

# Define the selected feature names (your top 10 features)
top_10_features = ['dst host srv diff host rate', 'same srv rate', 'dst host same srv rate', 'count', 'dst host count',
                   'dst host same src port rate', 'diff srv rate', 'service_eco_i', 'src bytes', 'dst host diff srv rate']

# Streamlit app title
st.title("Network Attack Prediction App")

# Uploading dataset for predictions
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"], key='unique_uploader')

# Checking if a file is uploaded
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # Display the first few rows of the uploaded data
    st.write("Uploaded Dataset Preview:")
    st.write(data.head())

    try:
        # One-hot encoding the 'service' column
        if 'service' in data.columns:
            service_encoded = pd.get_dummies(data['service'], prefix='service')
            # Remove the original 'service' column and add the one-hot encoded columns
            data = data.drop('service', axis=1)
            data = pd.concat([data, service_encoded], axis=1)
        
        # Checking if the dataset has the necessary top 10 features
        if set(top_10_features).issubset(data.columns):
            filtered_data = data[top_10_features]
            st.write("Filtered Data (Only Selected Features):")
            st.write(filtered_data.head())
            
            # Button to trigger prediction
            if st.button("Predict"):
                # Ensure no categorical values are present
                for col in filtered_data.columns:
                    if filtered_data[col].dtype == 'object':
                        filtered_data[col] = filtered_data[col].astype('category').cat.codes

                # Convert the filtered data to NumPy array for prediction
                data_for_prediction = np.array(filtered_data).astype(np.float32)

                # Reshape the data if needed (based on the input shape required by the model)
                data_for_prediction = data_for_prediction.reshape(-1, 1, len(filtered_data.columns))

                # Make predictions using the model
                predictions = model.predict(data_for_prediction)

                # Convert predictions to class labels
                predicted_classes = np.argmax(predictions, axis=1)

                # Display predictions
                st.write(f'Predicted Classes: {predicted_classes}')
        else:
            st.error("The uploaded dataset does not contain all the required top 10 feature columns.")
    
    except KeyError as e:
        st.error(f"Error: {e}. Please ensure the dataset contains the necessary features.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Example: Correlation heatmap button
if st.button("Show Correlation Heatmap"):
    if uploaded_file is not None:
        correlation_matrix = data.corr()
        fig = px.imshow(correlation_matrix, text_auto=True)
        st.plotly_chart(fig)
    else:
        st.write("Please upload a dataset first to show the correlation heatmap.")
