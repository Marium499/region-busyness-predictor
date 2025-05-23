import joblib
import logging
import yaml

import pandas as pd


logger = logging.getLogger(__name__)

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_model_to_path(model, model_name, dir_path='saved_models'):

    '''
    Save the trained model to a file.
    Args:
        model: Trained model
        model_name: Name of the model file
        dir_path: Directory path to save the model
    Returns:
        None
    '''

    try:
        joblib.dump(model, f'{dir_path}/{model_name}')
        logger.info(f'Model saved as {model_name} in {dir_path}')
    except Exception as e:
        logger.error(f'Error saving model: {e}')
        raise e
    

def load_model_from_path(model_name, dir_path=''):
    '''
    Load the trained model from a file.
    Args:
        model_name: Name of the model file
        dir_path: Directory path to load the model from
    Returns:
        model: Loaded model
    '''

    try:
        model = joblib.load(f'{dir_path}/{model_name}')
        logger.info(f'Model loaded as {model_name} from {dir_path}')
        return model
    except Exception as e:
        logger.error(f'Error loading model: {e}')
        raise e


# def numpy_to_df(numpy_array, index, pipeline):
#     '''
#     Postprocess the transformed features DataFrame.
#     Args:
#         df_transformed: Transformed features DataFrame
#         index: Original index of the DataFrame
#         pipeline: Feature pipeline
#         Returns:
#         df_transformed: Postprocessed features DataFrame
#     '''
#     df_transformed = pd.DataFrame(
#     numpy_array,
#     columns=[name.split('__', 1)[-1] for name in pipeline.get_feature_names_out()],
#     index=index
#     )

#     return df_transformed
