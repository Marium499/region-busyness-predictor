import logging

from data_collection import load_data_from_local, preprocess_data
from train import train_pipeline
from features import run_feature_pipeline
from utils import save_model_to_path
from predict import make_predictions

logger = logging.getLogger(__name__)

def run_pipeline(config, run_id):

    # file_path, k, resolution, train_features, target_features, model_params, test_size=0.33, use_grid_search=True, save_model=True, dir_path='./models'):

    logger.info("Starting the data processing pipeline...")
    logger.info(f"Run ID: {run_id}")

    df = load_data_from_local(file_path=config['data']['file_path'])
    df_processed, restaurants_ids = preprocess_data(df)
    # df_features = add_features(df_processed, restaurant_ids, k=5, resolution=7)
    # pipeline = FeaturePipeline(k=config['features']['k'], resolution=config['features']['resolution'], restaurants_ids=restaurants_ids)
    # df_features = pipeline.transform(df_processed)
    # logger.info(f"Processed data: {df_features.head()}")

    # feature pipeline
    if config['mode'] == 'train':
        df_features, features_pipeline, preprocess = run_feature_pipeline(df_processed, restaurants_ids, config['features']['k'], config['features']['resolution'], config['mode'], config['features']['R'])
        if config['features']['save_pipeline'] == True:
            # Save the pipeline to a file
            logger.info(f"Saving the pipelines to {config['features']['dir_path']}")
            save_model_to_path(features_pipeline, f"{run_id}_feature_pipeline", dir_path=config['features']['dir_path'])
            # Save the preprocessor to a file
            save_model_to_path(preprocess, f"{run_id}_preprocessor", dir_path=config['features']['dir_path'])          
        logger.info(f"Processed data: {df_features.head()}")
        # training
        logger.info("Starting the training pipeline...")
        rf_model, grid_model = train_pipeline(df_features, config['model']['train_features'], config['model']['target_features'], config['model']['model_params'], test_size=config['data']['test_size'], use_grid_search=config['model']['use_grid_search'], run_id=run_id)
        logger.info("Training pipeline completed.")

        # Save the trained model to a file
        if config["model"]["save_model"] == True:
            logger.info(f"Saving the model(s) to {config['model']['dir_path']}")
            save_model_to_path(rf_model, f"{run_id}_rf_model", dir_path=config['model']['dir_path'])
            if grid_model:
                save_model_to_path(grid_model, f"{run_id}_grid_search_model", dir_path=config['model']['dir_path'])
            

    elif config['mode'] == 'predict':
        logger.info("Starting the prediction pipeline...")
        y_pred = make_predictions(df_processed, config['model']['model_path'], config['features']['feature_pipeline_path'], config['features']['preprocess_path'], config["model"]["train_features"], dir_path=config['dir_path'])
        logger.info(f"Predictions: {y_pred[:5]}")
        logger.info("Prediction pipeline completed.")


