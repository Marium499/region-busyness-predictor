import logging
import argparse
import os
from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import joblib
import uuid
import numpy as np
import pandas as pd

from features import select_features
from data_collection import load_data_from_path
from utils import load_config, save_model_to_path


logging.basicConfig(filename='logs/app.log', level=logging.DEBUG, format='%(asctime)s:%(levelname)s:%(name)s:%(message)s')

logger = logging.getLogger(__name__)

def validate_data(X, y):

    '''
    Validate the data before training the model.
    Check if the target variable and features are empty or have only one unique value.
    Args:
        X: Features DataFrame
        y: Target variable Series
    Returns:
        None
    '''

    # Check if the target variable is empty
    if y.empty:
        logger.error('Target variable is empty. Please check the data.')
        return
    # Check if the features are empty
    if X.empty:
        logger.error('Feature variables are empty. Please check the data.')
        return
    # Check if the target variable has only one unique value
    if y.nunique().eq(1).all():
        logger.error('Target variable has only one unique value. Please check the data.')
        return
    # Check if the features have only one unique value
    if X.nunique().eq(1).all():
        logger.error('Feature variables have only one unique value. Please check the data.')
        return
    
def run_grid_search(X_train, y_train, model_params, X_test, y_test, n_jobs=-1):

    '''
    Run GridSearchCV to find the best hyperparameters for the RandomForestRegressor model.
    Args:
        X_train: Training features
        y_train: Training target variable
        model_params: Hyperparameters for the RandomForestRegressor model
        X_test: Test features
        y_test: Test target variable
    Returns:
        grid_search: GridSearchCV object
        regr_best: Best RandomForestRegressor model
        regr_score_best: Best score of the RandomForestRegressor model on the test data
    '''

    logger.info(f'Using RandomForestRegressor with grid search. Params: {model_params}')
    try:
        grid_search = GridSearchCV(estimator=RandomForestRegressor(),
                        param_grid=model_params,
                        cv = 3,
                        n_jobs=n_jobs, verbose=1, scoring='r2')
        grid_search.fit(X_train, y_train)
        grid_score_best = grid_search.best_score_
        regr_best = grid_search.best_estimator_
        regr_score_best = regr_best.score(X_test, y_test)
        logger.info(f'grid search best score: {grid_score_best}')
        logger.info(f'rf best model: {regr_best}')
        logger.info(f'rf best score on test data: {regr_score_best}')
        return grid_search, regr_best, regr_score_best
    except Exception as e:
        logger.error(f'Error during grid search: {e}')
        raise e


def train_pipeline(df, train_features, target_features, model_params, test_size=0.33, use_grid_search=True, n_jobs=-1):
    
    '''
    Train a RandomForestRegressor model on the provided dataset.

    This function splits the data into training and test sets, validates inputs,
    optionally performs hyperparameter tuning via GridSearchCV, and saves the trained model(s).

    Args:
        df (pd.DataFrame): Input dataset containing features and target.
        train_features (List[str]): List of column names to be used as features.
        target_features (List[str]): List of column names to be used as target(s).
        model_params (object): Configuration object or namespace containing model hyperparameters.
        test_size (float, optional): Proportion of the dataset to include in the test split. Defaults to 0.33.
        use_grid_search (bool, optional): Whether to perform GridSearchCV for hyperparameter tuning. Defaults to True.
        save_model (bool, optional): Whether to save the trained model(s) to disk. Defaults to True.
        dir_path (str, optional): Directory path to save model files. Defaults to 'models/'.

    Returns:
        None
    '''
    # Select features and target variable
    X = select_features(df, train_features)
    y = df[target_features]
    logger.info(f'X shape: {X.shape}')
    logger.info(f'y shape: {y.shape}')

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    logger.info(f'X_train shape: {X_train.shape}')
    logger.info(f'X_test shape: {X_test.shape}')
    logger.info(f'y_train shape: {y_train.shape}')
    logger.info(f'y_test shape: {y_test.shape}')

    # Validate the data
    validate_data(X_train, y_train)
    validate_data(X_test, y_test)

    # ravel the target variable to ensure it is 1D
    y_train = np.ravel(y_train)
    y_test = np.ravel(y_test)
    logger.info(f'y_train shape after ravel: {y_train.shape}')
    logger.info(f'y_test shape after ravel: {y_test.shape}')
  
    # Model training
    if use_grid_search == True:  
        grid_search, regr, regr_score = run_grid_search(X_train, y_train, model_params, X_test, y_test, n_jobs=n_jobs)
        return regr, grid_search

    else:
        logger.info(f'Using RandomForestRegressor without grid search. Max_depth: {model_params.max_depth}')
        try:
            regr = RandomForestRegressor(max_depth=model_params.max_depth[0], random_state=0, n_jobs=n_jobs)
            regr.fit(X_train, y_train)
            regr_score = regr.score(X_test, y_test)
            logger.info(f'rf score on test data: {regr_score}')
            return regr, None
            
        except Exception as e:
            logger.error(f'Error during training: {e}')
            raise e
            
        
        
def run_train_pipeline():

    parser = argparse.ArgumentParser()
    parser.add_argument('--training_data', type=str, help='Path to training data')
    parser.add_argument('--model_output', type=str, help='Path of output model', default='saved_models/models/')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    # parser.add_argument('--run_id', type=str, default=str(uuid.uuid4()), help='Run ID for the training pipeline')
    args = parser.parse_args()
    logger.info('Setting up the training pipeline...')
    config = load_config(args.config)
    df_path = args.training_data 
    model_output = args.model_output
    # run_id = args.run_id

    # logger.info(f'Run ID: {run_id}')
    print('Reading file: %s ...' % df_path)
    with open(df_path, 'r') as f:
        df = load_data_from_path(file_path=Path(df_path))

    logger.info('Starting the training pipeline...')
    rf_model, grid_model = train_pipeline(df, config['model']['train_features'], config['model']['target_features'], config['model']['model_params'], test_size=config['data']['test_size'], use_grid_search=config['model']['use_grid_search'], n_jobs=config['model']['n_jobs'])
    logger.info('Training pipeline completed.')

    # Save the trained model to a file
    # see how we can use config dir path instead of parse args path
    if config['model']['save_model'] == True:
        logger.info(f'Saving the model(s) to {model_output}')
        save_model_to_path(rf_model, f'rf_model', dir_path=model_output)
        if grid_model:
            save_model_to_path(grid_model, f'grid_search_model', dir_path=model_output)

    logger.info('Training pipeline completed.')
if __name__ == '__main__':
 
    run_train_pipeline()






