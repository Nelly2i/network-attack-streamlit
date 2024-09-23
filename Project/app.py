#!/usr/bin/env python
# coding: utf-8

# In[2]:

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
uploaded_file1 = st.file_uploader("Upload your dataset (CSV)", type=["csv"], key='uploader1')

if uploaded_file1 is not None:
    # Reading the uploaded CSV file
    uploaded_data = pd.read_csv(uploaded_file1)
    
    # Displaying the first few rows of the uploaded data
    st.write("Uploaded Dataset Preview:")
    st.write(uploaded_data.head())
    
    # Selecting only the feature columns from the uploaded dataset
    if set(top_10_features).issubset(uploaded_data.columns):
        filtered_data = uploaded_data[top_10_features]
        st.write("Filtered Data (Only Selected Features):")
        st.write(filtered_data.head())
        
        # Button to trigger prediction
        if st.button("Predict"):
            # Preparing the data for prediction (reshape if necessary)
            data_for_prediction = np.array(filtered_data).reshape(filtered_data.shape[0], 1, filtered_data.shape[1])
            
            # Making predictions using the model
            predictions = model.predict(data_for_prediction)
            
            # Converting predictions to class labels
            predicted_classes = np.argmax(predictions, axis=1)
            
            # Displaying predictions
            st.write("Predictions:")
            st.write(predicted_classes)
    else:
        st.error("The uploaded dataset does not contain the required feature columns.")
else:
    st.write("Please upload a dataset for predictions.")


# In[3]:


import plotly.express as px
import plotly.graph_objs as go
import streamlit as st
import pandas as pd

# Loading the dataset
uploaded_file2 = st.file_uploader("Upload your dataset (CSV)", type=["csv"], key='uploader2')
if uploaded_file2 is not None:
    data = pd.read_csv(uploaded_file2)
    st.write('Dataset Preview')
    st.write(data.head())

# Example: Correlation heatmap
if st.button("Show Correlation Heatmap"):
    correlation_matrix = data.corr()
    fig = px.imshow(correlation_matrix, text_auto=True)
    st.plotly_chart(fig)


# In[ ]:




