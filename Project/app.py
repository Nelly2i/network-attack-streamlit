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
top_10_features = ['dst host srv diff host rate', 'same srv rate', 'dst host same srv rate', 'count', 'dst host count',
                   'dst host same src port rate', 'diff srv rate', 'service_eco_i', 'src bytes', 'dst host diff srv rate']

# Streamlit app title
st.title("Network Attack Prediction App")

# Uploading dataset for prediction
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"], key='unique_uploader')

# Attack mapping
attack_mapping = {0: 'ipsweep', 1: 'satan', 2: 'portsweep', 3: 'back', 4: 'normal'}

# Checking if a file is uploaded
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    try:
        # One-hot encoding the 'service' column if it exists in the dataset
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
                    # Converting the filtered data to NumPy array for prediction
                    data_for_prediction = np.array(filtered_data).astype(np.float32)

                    # Reshaping to match the model's expected input shape (batch_size, 1, num_features)
                    data_for_prediction = np.expand_dims(data_for_prediction, axis=1)

                    # Making predictions
                    predictions = model.predict(data_for_prediction)

                    # Converting predictions to class labels
                    predicted_classes = np.argmax(predictions, axis=1)
                    predicted_attacks = [attack_mapping[p] for p in predicted_classes]

                    # Displaying predictions
                    st.write("Predicted Classes:")
                    st.write(predicted_attacks)

                    # Saving predictions to a CSV file
                    result_df = pd.DataFrame({'Id': data.index + 1, 'type_of_attack': predicted_attacks})
                    result_df.to_csv('predicted_attacks.csv', index=False)
                    st.write("Download the predictions:", result_df)
                    st.download_button(label="Download CSV", data=result_df.to_csv(index=False), file_name='predicted_attacks.csv', mime='text/csv')

                except KeyError as e:
                    st.error(f'Error: {e}. Please ensure the dataset contains the necessary features.')
                except Exception as e:
                    st.error(f'An error occurred: {e}')

        else:
            st.error("The uploaded dataset does not contain the required feature columns.")

    except Exception as e:
        st.error(f"An error occurred: {e}")

# Example: Correlation heatmap
if st.button("Show Correlation Heatmap"):
    try:
        # Selecting only the numeric columns for correlation
        numeric_data = data.select_dtypes(include=[np.number])

        if not numeric_data.empty:
            correlation_matrix = numeric_data.corr()
            fig = px.imshow(correlation_matrix, text_auto=True)
            st.plotly_chart(fig)
        else:
            st.warning("The dataset does not contain numeric columns for correlation.")

    except Exception as e:
        st.error(f"An error occurred while generating the heatmap: {e}")
