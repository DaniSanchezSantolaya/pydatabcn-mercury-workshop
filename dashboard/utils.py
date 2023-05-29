import pandas as pd
from joblib import load
import pickle
from sklearn.metrics import accuracy_score
import streamlit as st

from mercury.monitoring.drift.domain_classifier_drift_detector import DomainClassifierDrift
from mercury.monitoring.drift.ks_drift_detector import KSDrift
from mercury.monitoring.estimation.performance_predictor import PerformancePredictor

def load_model(path_model, path_features):

    # Load Model
    model = load(path_model)

    # Load Features
    if path_features is not None:
        with open(path_features, "rb") as fp:   # Unpickling
            features = pickle.load(fp)
    else:
        features = None
        
    return model, features

@st.cache_data
def load_training_dataset(path_dataset):

    df_train = pd.read_csv(path_dataset + "train.csv")
    df_test = pd.read_csv(path_dataset + "test.csv")

    return df_train, df_test

@st.cache_data
def drift_detection(df_source, df_inference, features, drift_detector_type="ks"):
    
    X_source = df_source[features].values
    X_inference = df_inference[features].values
    
    if drift_detector_type.lower() == "ks":
        drift_detector = KSDrift(X_source, X_inference, features=features, p_val=0.01)
    elif drift_detector_type.lower() == "domain_classifier":
        drift_detector = DomainClassifierDrift(
            X_source, X_inference, features=features, p_val=0.05, n_runs=3
        )
    else:
        raise ValueError("Invalid drift_detector_type. Select: 'ks' or 'domain_classifier'")
        
    drift_metrics = drift_detector.calculate_drift()
    return drift_detector, drift_metrics

@st.cache_data
def compute_drift_and_performance_estimation_over_time(_model, df_inference, df_train, df_test, _schema, features, label, t_start=1, t_end=7, seed=48, return_as_df=True):
    
    # Compute predictions train set
    y_pred_proba_train = _model.predict_proba(df_train[features])[:,1]
    
    # Look at each time step
    feature_drift_ks_score = []
    feature_drift_ks_detection = []
    
    feature_drift_dc_score = []
    feature_drift_dc_detection = []
    
    prediction_drift_ks_score = []
    prediction_drift_ks_detection = []
    
    performance_predictions = []
    performance_real = []
    
    detectors = {
        'ks_features': [],
        'domain_classifier_features': [],
        'ks_predictions': []
    }
    
    for t in range(t_start, t_end + 1):
        
        # Get Dataset for specific time
        df_inference_t = df_inference[df_inference["time"] == t]
        y_true = df_inference_t[label]
        df_inference_t = df_inference_t[features]

        # Get Model predictions
        y_pred = _model.predict(df_inference_t)
        y_pred_proba = _model.predict_proba(df_inference_t)[:,1]
        
        # KS Drift Over Features
        drift_detector, drift_metrics = drift_detection(
            df_source=df_train, df_inference=df_inference_t, features=features, drift_detector_type = "ks"
        )
        feature_drift_ks_score.append(drift_metrics["score"])
        feature_drift_ks_detection.append(drift_metrics["drift_detected"])
        detectors['ks_features'].append(drift_detector)
        
        # Domain Classifier Drift Over Features
        drift_detector, drift_metrics = drift_detection(
            df_source=df_train, df_inference=df_inference_t, features=features, drift_detector_type = "domain_classifier"
        )
        feature_drift_dc_score.append(drift_metrics["score"])
        feature_drift_dc_detection.append(drift_metrics["drift_detected"])
        detectors['domain_classifier_features'].append(drift_detector)
        
        # KS Drift Over Predictions
        df_pred_train = pd.DataFrame(y_pred_proba_train, columns=['prediction'])
        df_pred_inference = pd.DataFrame(y_pred_proba, columns=['prediction'])
        drift_detector, drift_metrics = drift_detection(
            df_source=df_pred_train, df_inference=df_pred_inference, features=['prediction'], drift_detector_type="ks"
        )
        prediction_drift_ks_score.append(drift_metrics["score"])
        prediction_drift_ks_detection.append(drift_metrics["drift_detected"])
        detectors['ks_predictions'].append(drift_detector)
        
        # Performance Estimation
        performance_predictor = PerformancePredictor(_model, metric_fn=accuracy_score, random_state=seed)
        performance_predictor.fit(X=df_test[features], y=df_test[label], X_serving=df_inference_t, dataset_schema=_schema)
        predicted_acc = performance_predictor.predict(df_inference_t[features])
        performance_predictions.append(predicted_acc[0])
        
        # Real Performance (we don't really have it, but we can compute for illustration purposes)
        real_performance = accuracy_score(y_true, y_pred)
        performance_real.append(real_performance)
        
    metrics = {
        'feature_drift_ks_score': feature_drift_ks_score,
        'feature_drift_ks_detection': feature_drift_ks_detection,
        'feature_drift_dc_score': feature_drift_dc_score,
        'feature_drift_dc_detection': feature_drift_dc_detection,
        'prediction_drift_ks_score': prediction_drift_ks_score,
        'prediction_drift_ks_detection': prediction_drift_ks_detection,
        'performance_predictions': performance_predictions,
        'performance_real': performance_real,
    }

    if return_as_df:
        df_metrics = pd.DataFrame()
        df_metrics['feature_drift_ks_score'] = metrics['feature_drift_ks_score']
        df_metrics['feature_drift_ks_detection'] = metrics['feature_drift_ks_detection']
        df_metrics['feature_drift_dc_score'] = metrics['feature_drift_dc_score']
        df_metrics['feature_drift_dc_detection'] = metrics['feature_drift_dc_detection']
        df_metrics['prediction_drift_ks_score'] = metrics['prediction_drift_ks_score']
        df_metrics['prediction_drift_ks_detection'] = metrics['prediction_drift_ks_detection']
        df_metrics['performance_predictions'] = metrics['performance_predictions']
        df_metrics['performance_real'] = metrics['performance_real']
        return df_metrics, detectors
    
    return metrics, detectors

def build_table_drifted_features(drift_detector, t_drift_detected):
    
    l_timesteps = []
    l_drifted_features = []
    for t in t_drift_detected:
        l_timesteps.append(t)
        l_drifted_features.append(drift_detector[t-1].get_drifted_features())
    df_drifted_features = pd.DataFrame()
    df_drifted_features["timestep"] = l_timesteps
    df_drifted_features["features"] = l_drifted_features

    return df_drifted_features