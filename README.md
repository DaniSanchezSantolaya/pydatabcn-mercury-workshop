# Introduction

In this workshop, we will introduce mercury.robust and mercury.monitoring. With mercury.robust we can pass tests in our datasets and trained models to detect undesirable conditions such as high level of noise in the labels, label leaking in the features of the dataset, or model excessive sensitivity to drift in certain features. Mercury.monitoring contains components that can help us to detect data drift in our inference data and estimate the performance of our model when real labels are not yet available.

## Steps

### 1. Dataset Generation

Notebook "1_dataset_generation.ipynb" contains the code for dataset generation. In this notebook several robust data tests are passed.

### 2. Model Training

Notebook "2_model_training.ipynb" contains the code for training a model. In this notebook several robust model tests are passed.

### 3. Model Inference and monitoring

Notebook "3_model_inference_and_monitoring.ipynb" contains the code to perform data drift detection and performance estimation on inference data

### 4. Dashboard for monitoring

The "app" folder contains a streamlit application which contains an example of dashboard that shows the data drift and performance estimation over time. The notebooks must be executed previously in order to show the data in the dashboard. Then, the streamlit app can be run executing (streamlit must be installed first):
```
cd app
python -m streamlit run app.py
```