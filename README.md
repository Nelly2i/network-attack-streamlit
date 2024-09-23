# Network Attack Prediction Using CNN

This project is aimed at building a predictive model to classify network attacks using a Convolutional Neural Network (CNN). The model is trained on a dataset with 4 types of network attacks and normal activity. The project also includes deploying the model as a Streamlit web app for easy interaction and predictions.

## Features

- Classification of network activities into 5 categories: ipsweep, satan, portsweep, back, and normal.
- Data preprocessing and feature selection using Random Forest.
- Model training using CNN.
- Model evaluation using precision, recall, and F1-score metrics.
- Deployment of the model using a Streamlit web app.
- Visualization of results using Plotly.

## Dataset

The dataset contains various features representing network traffic patterns. The target column has 5 unique attack types:
- ipsweep
- satan
- portsweep
- back
- normal (no attack)

### Key Features Used in Model

- dst_host_srv_diff_host_rate: % of connections to different hosts on the same service.
- same_srv_rate: % of connections to the same service.
- dst_host_same_srv_rate: % of connections to the same service on the destination host.
- count: Number of connections to the same host as the current connection in the past 2 seconds.
- dst_host_count: Number of connections to the same destination host as the current connection.
- dst_host_diff_srv_rate: % of connections to different services on the destination host.
- src_bytes: Bytes sent from the source to the destination.
- diff_srv_rate: % of connections to different services.
- dst_host_same_src_port_rate: % of connections from the same source port.

## Installation

1. Clone the repository:
   bash
   git clone https://github.com/Nelly2i/network-attack-streamlit.git
   cd network-attack-streamlit
   

2. Install the required dependencies:
   bash
   pip install -r requirements.txt
   

3. Run the Streamlit app:
   bash
   streamlit run app.py
   

## Usage

1. Upload a CSV file containing the relevant features.
2. The model will process the file and predict the type of network activity (attack type or normal).
3. Visualize correlation matrix and prediction results directly in the app.

## Model Architecture

- Input Layer: 10 selected features
- Conv1D Layers: Convolutional layers with ReLU activation
- MaxPooling Layers: Downsampling to reduce dimensionality
- Dense Layers: Fully connected layers
- Output Layer: Softmax for multi-class classification

## Evaluation

The model was evaluated using various metrics:
- Precision
- Recall
- F1-score

## Example Output

    precision    recall  f1-score   support

    ipsweep       1.00      1.00      1.00      1088
    satan       1.00      1.00      1.00       710
    portsweep       1.00      1.00      1.00       410
    back       1.00      1.00      1.00       268
    normal       0.98      1.00      0.99       122

    accuracy                           1.00      2598
    macro avg       1.00      1.00      1.00      2598
    weighted avg       1.00      1.00      1.00      2598


## Technologies Used

- Python: Primary programming language.
- TensorFlow/Keras: For building and training the CNN model.
- Streamlit: For deploying the model as a web app.
- Pandas & NumPy: For data manipulation and preprocessing.
- Plotly: For visualizations.

## Future Work

- Explore other deep learning architectures like LSTM or GRU for time-series network data.
- Integrate a more dynamic and interactive dashboard for users.
- Improve real-time predictions with streaming data.
- Increase the size of the training data

## Acknowledgements

- [Keras Documentation](https://keras.io)
- [Streamlit Documentation](https://docs.streamlit.io)
