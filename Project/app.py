#!/usr/bin/env python
# coding: utf-8

# In[2]:

import streamlit as st
#import seaborn as sns
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

if uploaded_file is not None:
    # Reading the uploaded CSV file
    uploaded_data = pd.read_csv(uploaded_file)
    
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
            try:
        #Applying one hot encoding to the 'service' column if it exists
               if 'service' in data.columns:
                   data = pd.get_dummies(data, columns=['service'])
            try:
        # Filtering out the selected features
                filtered_data = data[top_10_features]
            except Exception as e
                print(f'Error: {str(e)}')

        # Checking for non-numeric values and encode them if necessary
        for col in filtered_data.columns:
            if filtered_data[col].dtype == 'object':  # If the column is categorical
                st.write(f"Encoding column: {col}")
                filtered_data[col] = filtered_data[col].astype('category').cat.codes

        # Converting the filtered data to NumPy array and reshape for prediction
        data_for_prediction = np.array(filtered_data).astype(np.float32)

        # Reshaping the data if needed (based on the input shape required by the model)
        data_for_prediction = data_for_prediction.reshape(-1, len(selected_features))

        # Making predictions
        predictions = model.predict(data_for_prediction)


          
          # Preparing the data for prediction (reshape if necessary)
            
            # Making predictions using the model
           # predictions = model.predict(data_for_prediction)
            
         #Converting predictions to class labels
          predicted_classes = np.argmax(predictions, axis=1)
            
          # Displaying predictions
          st.write(f'Predicted_classes: {predicted_classes}')
except KeyError as e:
    st.error(f'Error: {e}. Please ensure the dataset contains the necessary features.')
except Exception as e:
    st.error(f'An error occurred: {e}')
            #st.write(predicted_classes)
  #  else:
        #st.error("The uploaded dataset does not contain the required feature columns.")
#else:
    #st.write("Please upload a dataset for predictions.")


# In[3]:


import plotly.express as px
import plotly.graph_objs as go

# Loading the dataset
#uploaded_file2 = st.file_uploader("Upload your dataset (CSV)", type=["csv"], key='uploader2')
#if uploaded_file2 is not None:
   # data = pd.read_csv(uploaded_file2)
    #st.write('Dataset Preview')
    #st.write(data.head())

# Example: Correlation heatmap
if st.button("Show Correlation Heatmap"):
    correlation_matrix = data.corr()
    fig = px.imshow(correlation_matrix, text_auto=True)
    st.plotly_chart(fig)


# In[ ]:




