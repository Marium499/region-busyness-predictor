import logging
import numpy as np
import os
import pandas as pd
import mlflow
from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType

def init():
    '''
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    You can write the logic here to perform init operations like caching the model in memory
    '''

    global model
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    model_path = os.path.join(
        os.getenv('AZUREML_MODEL_DIR'), 'rbp-model-artifacts'
    )
    # Load the model, it's input types and output names
    model = mlflow.sklearn.load_model(model_path)
    logging.info('Init complete')

input_sample = pd.DataFrame(data=[{
    'courier_id': 'a98737cbhoho5012hoho4b5bhoho867fhoho8475c658546d',
    'order_number': 281289453,
    'courier_location_timestamp': '2021-04-02T04:30:42.328Z',
    'courier_lat': 50.4845203253,
    'courier_lon': -104.6188755956,
    'order_created_timestamp': '2021-04-02T04:20:42Z',
    'restaurant_lat': 50.4836962533,
    'restaurant_lon': -104.614349585
}])

output_sample = np.array([104])

@input_schema('data', PandasParameterType(input_sample))
@output_schema(NumpyParameterType(output_sample))

def run(data):
    '''
    This function is called for every invocation of the endpoint to perform the actual scoring/prediction.
    In the example we extract the data from the json input and call the scikit-learn model's predict()
    method and return the result back
    '''
    pred = model.predict(data)
    return pred.tolist()