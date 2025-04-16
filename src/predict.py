import joblib
import logging

import pandas as pd

from utils import load_model_from_path, postprocess_features

logger = logging.getLogger(__name__)


def make_predictions(X_test, model_path,feature_pipeline_path,preprocess_path,train_features, dir_path=''):

    '''
    Predict the target variable using the trained model.
    Args:
        X_test: Test features DataFrame
        model_path: Path to the trained model
        feature_pipeline_path: Path to the feature pipeline
        preprocess_path: Path to the encoder pipeline
        train_features: List of features used for training
        dir_path: Directory path to load the models from
    Returns:
        y_pred: Predicted target variable
    '''
    
    # Load the feature pipeline
    feature_pipeline = load_model_from_path(feature_pipeline_path, dir_path)
    
    # Load the encoder pipeline
    preprocess_pipeline = load_model_from_path(preprocess_path, dir_path)

    # Load the trained model
    model = load_model_from_path(model_path, dir_path)

    X_test_transformed = X_test.copy()
    # Transform the test features using the feature pipeline
    X_test_transformed = feature_pipeline.transform(X_test_transformed)
    # Transform the test features using the encoder pipeline
    X_test_transformed = preprocess_pipeline.transform(X_test_transformed)

    # This is a workaround to fix the issue with the column names
    X_test_transformed = postprocess_features(X_test_transformed, X_test.index, preprocess_pipeline)
    logger.info(f"Transformed test features: {X_test_transformed.head()}")
    # Select the features for prediction
    X_test_transformed = X_test_transformed[train_features]
    # Make predictions
    y_pred = model.predict(X_test_transformed)
   
    return y_pred