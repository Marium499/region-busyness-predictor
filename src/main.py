import logging

import pandas as pd
import argparse
import yaml
import uuid

from pipeline import run_pipeline


logging.basicConfig(filename="logs/app.log", level=logging.DEBUG, format="%(asctime)s:%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    return parser.parse_args()

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
    
def main():
    logger.info("--------------------------------------------------")
    args = parse_args()
    config = load_config(args.config)
    run_id = str(uuid.uuid4())
    run_pipeline(config, run_id)
    logger.info("Pipeline completed.")
        

# def main_old():

#     logger.info("Starting the data processing pipeline...")
#     df = load_data_from_local()
#     df_processed, restaurants_ids = preprocess_data(df)
#     # df_features = add_features(df_processed, restaurant_ids, k=5, resolution=7)
#     pipeline = FeaturePipeline(k=5, resolution=9, restaurants_ids=restaurants_ids, random_state=1)
#     df_features = pipeline.transform(df_processed)
#     # logger.info(f"Processed data: {df_features.head()}")
#     logger.info(f"Processed data: {df_features.head()}")

#     # feature selectors, should be in the config file


#     TRAIN_FEATURES = ['dist_to_restaurant', 'Hdist_to_restaurant', 'avg_Hdist_to_restaurants',
#                       'date_day_number', 'restaurant_id', 'clusters_embedding', 'h3_index',
#                       'date_hour_number','restaurants_per_index']
#     TARGET_FEATURES = ['orders_busyness_by_h3_hour']

#     model_params = {
#     'max_depth': [4,5],
#     'min_samples_leaf': [50,75],
#     'n_estimators': [100,150]
#     }
    
#     ###
#     # the k value in column name does not affect the code or the output so it has been removed.
#     # However, if it is required to have the k value in the column name, we could create logic to
#     # replace the clustering_embedding items in the TRAIN_FEATURES list with str('k' + str(k) + '_clusters_embedding')
#     # and str('k' + str(k) + '_clusters_embedding_error')
#     ###

#     # training
#     logger.info("Starting the training pipeline...")
#     train_pipeline(df_features, TRAIN_FEATURES, TARGET_FEATURES, model_params, test_size=0.33, use_grid_search=True, save_model=True, dir_path="./models")
#     logger.info("Training pipeline completed.")




if __name__ == "__main__":
    main()


