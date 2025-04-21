import logging
import argparse
from pathlib import Path

from data_collection import load_data_from_path, get_restaurant_id_mapping
from features import get_transformed_df
from utils import save_model_to_path
from utils import load_config

logging.basicConfig(filename='logs/app.log', level=logging.DEBUG, format='%(asctime)s:%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


def run_pretrain_pipeline():

    '''
    Main entry point into the pretraining pipeline for RBP system.
    Runs and saves the feature pipeline to process the data and save the transformed data.
    '''

    logger.info('--------------------------------------------------')
    args = parse_arguments()
    raw_data_path = Path(args.raw_data)
    config = load_config(args.config)
    model_output = Path(args.output_model_path)
    data_output = Path(args.output_data_path)

    logger.info('Starting the data processing pipeline...')
    df = load_data_from_path(file_path=raw_data_path)
    restaurants_ids = get_restaurant_id_mapping(df)

    df_features, full_pipeline = get_transformed_df(df, restaurants_ids, config['features']['k'], config['features']['resolution'], config['features']['R'])
    
    if config['features']['save_pipeline'] == True:  
        # Save the pipeline to a file
        logger.info(f'Saving the pipelines to {model_output}')
        save_model_to_path(full_pipeline, config['features']['feature_pipeline_filename'], dir_path=model_output)
    logger.info(f'Processed data: {df_features.head()}')
    df_features.to_csv(f'{data_output}/{config['features']['feature_output_filename']}', index=False)
    
    logger.info('Pretraining pipeline completed.')

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_data', type=str, help='Path to training data in blob storage')
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--output_data_path', type=str, default='data/')
    parser.add_argument('--output_model_path', type=str, default='saved_models/models/')
    return parser.parse_args()


if __name__ == '__main__':
    run_pretrain_pipeline()

    
