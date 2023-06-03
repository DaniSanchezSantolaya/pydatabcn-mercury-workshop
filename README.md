# Introduction

In this workshop, we will introduce mercury.robust and mercury.monitoring. With mercury.robust we can pass tests in our datasets and trained models to detect undesirable conditions such as high level of noise in the labels, label leaking in the features of the dataset, or model excessive sensitivity to drift in certain features. Mercury.monitoring contains components that can help us to detect data drift in our inference data and estimate the performance of our model when real labels are not yet available.

## Requirements

We will be using mercury.robust and mercury.monitoring. You can install them using pip

```
pip install mercury-robust mercury-monitoring
```

We will also use other data science libraries like scikit-learn, pandas or numpy.

## Steps

The Workshop consists on 4 steps, where the goal are:
- Generate a robust dataset
- Train a model which will pass some tests
- Use the model for inference monitor data drift respect with the training dataset. Estimate the performance of the model at inference time without the labels.
- Show an example of dashboard for data and model monitoring.

### 1. Dataset Generation

Notebook "1_dataset_generation.ipynb" contains the code for dataset generation. In this notebook several robust data tests are passed.

### 2. Model Training

Notebook "2_model_training.ipynb" contains the code for training a model. In this notebook several robust model tests are passed.

### 3. Model Inference and monitoring

Notebook "3_model_inference_and_monitoring.ipynb" contains the code to perform data drift detection and performance estimation on inference data

### 4. Dashboard for monitoring

The "dashboard" folder contains a streamlit application which contains an example of dashboard to show data drift and performance estimation over time. The notebooks must be executed previously in order to show the data in the dashboard. Then, the streamlit app can be run executing (streamlit must be installed first with `pip install streamlit`):
```
cd dashboard
python -m streamlit run app.py
```