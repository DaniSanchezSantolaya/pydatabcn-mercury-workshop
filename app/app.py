# Imports
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from utils import (
    compute_drift_and_performance_estimation_over_time, 
    load_model,
    load_training_dataset,
    build_table_drifted_features
)

# Constants
T_START = 2
T_END = 13
LABEL = "default.payment.next.month"
PATH_DATASET = "../dataset/"
PATH_INFERENCE_DATA = "../data/uci_credit_drifted_inference.csv"
PATH_MODEL = "../models/model.joblib" 
PATH_FEATURES = "../models/features.pkl"

# Load Training Dataset
df_train, df_test = load_training_dataset(PATH_DATASET)

# Load Inference Data
df_inference = pd.read_csv(PATH_INFERENCE_DATA)

# Load Model
_model, features = load_model(PATH_MODEL, PATH_FEATURES)

# Load Data Schema
from mercury.dataschema import DataSchema
_schema = DataSchema.load(PATH_DATASET + "schema.json")
del _schema.feats['default.payment.next.month']

# Calculate Drift and Performance Estimation
metrics, detectors = compute_drift_and_performance_estimation_over_time(_model, df_inference, df_train, df_test, _schema, features, LABEL, t_start=T_START, t_end=T_END)

# APP STARTS HERE

# Sidebar
with st.sidebar:

    show_drift_detection = st.checkbox('Show Drift Detection', value=True)
    show_performance_prediction = st.checkbox('Show Performance Prediction')

    t_end = st.slider('Time', T_START, T_END, value=10)
    
df_metrics_show = metrics.iloc[0:t_end]
df_metrics_show.index = df_metrics_show.index + 1

# Drift Detection
if show_drift_detection:
    tabs = st.tabs(["KS Feature Drift", "Domain Classifier Feature Drift", "KS Prediction Drift"])

    # KS Feature Drift
    with tabs[0]:

        # Drift Detected?
        df_drift_detected = df_metrics_show[df_metrics_show["feature_drift_ks_detection"] == True]
        t_drift_detected = df_drift_detected.index.values

        drift_detected = len(t_drift_detected) > 0
        st.metric(label="Drift Detected: ", value=drift_detected)

        # Chart Drift Score over time
        st.subheader("KS Feature Drift Score")
        st.line_chart(df_metrics_show, y="feature_drift_ks_score", use_container_width=True)

        # Features with Drift
        if drift_detected:
            df_drifted_features = build_table_drifted_features(detectors['ks_features'], t_drift_detected)
            df_drifted_features

        # Distribution of Features
        col1, col2 = st.columns([3,1])
        with col1:
            t_feature_show = st.selectbox("timestep: ", options=["-"] + list(np.arange(1, T_END+1)))
        with col2:
            feature_show = st.selectbox("feature: ", ["-"] + features)

        if t_feature_show != "-" and feature_show != "-":
            axes = detectors['ks_features'][t_feature_show].plot_distribution_feature(name_feature=feature_show)
            st.pyplot(axes.figure)

    # Domain Classifier Drift
    with tabs[1]:

        # Drift Detected?
        df_drift_detected = df_metrics_show[df_metrics_show["feature_drift_dc_detection"] == True]
        t_drift_detected = df_drift_detected.index.values

        drift_detected = len(t_drift_detected) > 0
        st.metric(label="Drift Detected: ", value=drift_detected)

        # Chart Drift Score over time
        st.subheader("Domain Classifier Feature Drift Score")
        st.line_chart(df_metrics_show, y="feature_drift_dc_score", use_container_width=True)

        
        # Distribution of Features
        t_feature_show = st.selectbox("timestep: ", options=["-"] + list(np.arange(1, T_END+1)), key="checkbox_dc_t")
        if t_feature_show != "-" :
            ax = detectors['domain_classifier_features'][t_feature_show].plot_feature_drift_scores()
            st.pyplot(ax.figure)
        feature_show = st.selectbox("feature: ", ["-"] + features, key="checkbox_df_feature")
        if t_feature_show != "-" and feature_show != "-":
            axes = detectors['domain_classifier_features'][t_feature_show].plot_distribution_feature(name_feature=feature_show)
            st.pyplot(axes.figure)

    # Predictions Drift
    with tabs[2]:

        # Drift Detected?
        df_drift_detected = df_metrics_show[df_metrics_show["prediction_drift_ks_detection"] == True]
        t_drift_detected = df_drift_detected.index.values

        drift_detected = len(t_drift_detected) > 0
        st.metric(label="Drift Detected: ", value=drift_detected)

        # Chart Drift Score
        st.subheader("KS Predictions Drift Score")
        st.line_chart(df_metrics_show, y="prediction_drift_ks_score", use_container_width=True)

        # Predictions Distribution
        t_feature_show = st.selectbox("timestep: ", options=["-"] + list(np.arange(1, T_END+1)), key="checkbox_predictions_t")
        if t_feature_show != "-" :
            axes = detectors['ks_predictions'][t_feature_show].plot_distribution_feature(name_feature="prediction")
            st.pyplot(axes.figure)


# Performance Prediction
if show_performance_prediction:

    if show_performance_prediction:
        show_real_performance = st.checkbox('Show real performance')

    st.subheader("Predicted Performance")

    predictions_chart_data = pd.DataFrame()
    predictions_chart_data["predicted_acc"] = df_metrics_show["performance_predictions"]
    if show_real_performance:
        predictions_chart_data["real_acc"] = df_metrics_show["performance_real"]


    performance_predicted = df_metrics_show["performance_predictions"]
    x = np.arange(len(performance_predicted)) + 1

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=performance_predicted, name="predicted", line_shape='hv'))

    if show_real_performance:
        performance_real = df_metrics_show["performance_real"]
        fig.add_trace(go.Scatter(x=x, y=performance_real, name="real", line_shape='hv'))

    fig.update_traces(hoverinfo='text+name', mode='lines+markers')
    fig.update_layout(legend=dict(y=0.5, traceorder='reversed', font_size=16))

    st.plotly_chart(fig, use_container_width=True)




