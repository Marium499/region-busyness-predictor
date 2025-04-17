import logging
import logging
import argparse
import uuid

from data_collection import load_data_from_path, get_restaurant_id_mapping
from features import get_transformed_df
from utils import save_model_to_path
from utils import load_config

logging.basicConfig(filename="logs/app.log", level=logging.DEBUG, format="%(asctime)s:%(levelname)s:%(name)s:%(message)s")

logger = logging.getLogger(__name__)


def run_feature_pipeline():

    # file_path, k, resolution, train_features, target_features, model_params, test_size=0.33, use_grid_search=True, save_model=True, dir_path='./models'):
    logger.info("--------------------------------------------------")
    args = parse_args()
    config = load_config(args.config)
    run_id = str(uuid.uuid4())
    logger.info("Starting the data processing pipeline...")
    logger.info(f"Run ID: {run_id}")
    df = load_data_from_path(file_path=config['data']['file_path'])
    

    # feature pipeline
    if config['mode'] == 'train':
        restaurants_ids = get_restaurant_id_mapping(df)
        df_features, full_pipeline = get_transformed_df(df, restaurants_ids, config['features']['k'], config['features']['resolution'], config['mode'], config['features']['R'])
        if config['features']['save_pipeline'] == True:
            # Save the pipeline to a file
            logger.info(f"Saving the pipelines to {config['features']['dir_path']}")
            save_model_to_path(full_pipeline, f"{run_id}_full_feature_pipeline", dir_path=config['features']['dir_path'])

        logger.info(f"Processed data: {df_features.head()}")
        df_features.to_csv(f'data/features_dataset.csv', index=False)
    logger.info("Pipeline completed.")
        # training
        # logger.info("Starting the training pipeline...")
        # rf_model, grid_model = train_pipeline(df_features, config['model']['train_features'], config['model']['target_features'], config['model']['model_params'], test_size=config['data']['test_size'], use_grid_search=config['model']['use_grid_search'], run_id=run_id)
        # logger.info("Training pipeline completed.")

        # # Save the trained model to a file
        # if config["model"]["save_model"] == True:
        #     logger.info(f"Saving the model(s) to {config['model']['dir_path']}")
        #     save_model_to_path(rf_model, f"{run_id}_rf_model", dir_path=config['model']['dir_path'])
        #     if grid_model:
        #         save_model_to_path(grid_model, f"{run_id}_grid_search_model", dir_path=config['model']['dir_path'])
            

    # elif config['mode'] == 'predict':
    #     logger.info("Starting the prediction pipeline...")
    #     y_pred = make_predictions(df_processed, config['model']['model_path'], config['features']['feature_pipeline_path'], config['features']['preprocess_path'], config["model"]["train_features"], dir_path=config['dir_path'])
    #     logger.info(f"Predictions: {y_pred[:5]}")
    #     logger.info("Prediction pipeline completed.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    run_feature_pipeline()

    
