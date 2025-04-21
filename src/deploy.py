import joblib
import logging
import yaml

import pandas as pd
from azure.ai.ml.entities import ManagedOnlineDeployment, CodeConfiguration
import argparse
from sklearn.pipeline import Pipeline, FunctionTransformer
from azure.ai.ml.entities import ManagedOnlineEndpoint
from azure.ai.ml import MLClient
from azure.identity import ClientSecretCredential, DefaultAzureCredential, AzureCliCredential
from azure.ai.ml.entities import (
    ManagedOnlineDeployment,
    Environment,
    Model,
    AmlCompute
)

logging.basicConfig(filename="logs/app.log", level=logging.DEBUG, format="%(asctime)s:%(levelname)s:%(name)s:%(message)s")

logger = logging.getLogger(__name__)

def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def deploy_model():

    logger.info('--------------------------------------------------')
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default='config.yaml', help='Path to config file')
    # parser.add_argument('--model_name', type=str, default=None, help='Name of the model to deploy')
    # parser.add_argument('--model_version', type=int, default=None, help='Version of the model to deploy')
    args = parser.parse_args()
    config = load_config(args.config_file)
    # model_name = args.model_name
    # model_version = args.model_version
    # if model_name is None:
    #     model_name = config['model']['mlflow_model_name']
    # if model_version is None:
    #     model_version = config['model']['mlflow_model_version']
    # model_output = Path(args.output_model_path)
    # data_output = Path(args.output_data_path)

    cluster_name = config['azure']['cluster_name']
    model_name = config['model']['mlflow_model_name']
    model_version = config['model']['mlflow_model_version']
    subscription_id = config['azure']['subscription_id']
    resource_group_name = config['azure']['resource_group_name']
    workspace_name = config['azure']['workspace_name']

    ml_client = MLClient(
    credential=DefaultAzureCredential(),
    subscription_id=subscription_id,
    resource_group_name=resource_group_name,
    workspace_name=workspace_name,
    )
    env_name = config['deployment']['environment']['name']
    env_version = config['deployment']['environment']['version']
    try:
        env = ml_client.environments.get(name="rbp-score-env", version=1)
        logger.info("Environment already exists")
    except Exception as e:
        logger.info("Environment does not exist, creating a new one")
        env = Environment(
            name = env_name,
            conda_file = config['deployment']['environment']['conda_file'],
            image = config['deployment']['environment']['base_image'],
        )
        ml_client.environments.create_or_update(env)

    endpoint_name = config['deployment']['endpoint_name']

    try:
        endpoint = ml_client.online_endpoints.get(name=endpoint_name)
        print(f"You already have an endpoint named {endpoint_name}, we'll reuse it as is.")
    except Exception:
        print(f"Creating a new endpoint named {endpoint_name}...")
        # create an online endpoint
        # if you run into an out of quota error, change the instance_type to a comparable VM that is available.\
        # Learn more on https://azure.microsoft.com/en-us/pricing/details/machine-learning/.
        endpoint = ManagedOnlineEndpoint(
            name=endpoint_name,
            description="this is the rbp online endpoint",
            auth_mode="key",
        )
        ml_client.online_endpoints.begin_create_or_update(endpoint).result()
        endpoint = ml_client.online_endpoints.get(name=endpoint_name)
        print(f"Endpoint {endpoint_name} created successfully.")

    model = ml_client.models.get(name=model_name, version=model_version)

    logger.info("Creating a new deployment...")
    # create the online deployment
    rbp_deployment = ManagedOnlineDeployment(
        name=config['deployment']['name'],
        endpoint_name=endpoint_name,
        model=model,
        code_configuration=CodeConfiguration(
            code=".",
            scoring_script="score.py",
        ),
        environment=env,
        instance_type=config['deployment']['instance_type'],
        instance_count=config['deployment']['instance_count'],
    )
    ml_client.online_deployments.begin_create_or_update(
        rbp_deployment
    ).result()
    logger.info(
        f"Deployment {rbp_deployment.name} created with status: {rbp_deployment.provisioning_state}"
    )

    endpoint.traffic = {rbp_deployment.name: 100}
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()



# cluster_name="spn-compute"

# ml_client = MLClient(
#     credential=AzureCliCredential(),
#     subscription_id="32daaf9e-fc90-446c-8950-13de434a7828",
#     resource_group_name="ml",
#     workspace_name="ml-ws",
# )

# try:
#     env = ml_client.environments.get(name="rbp-score-env", version=1)
#     logger.info("Environment already exists")
# except Exception:
#     logger.info("Environment does not exist, creating a new one")
#     env = Environment(
#         name="rbp-score-env",
#         conda_file="../environment/conda.yaml",
#         image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
#     )
#     ml_client.environments.create_or_update(env)

# # create an online endpoint
# endpoint_name = "rbp-endpoint"
# try:
#     endpoint = ml_client.online_endpoints.get(name=endpoint_name)
#     print(f"You already have an endpoint named {endpoint_name}, we'll reuse it as is.")
# except Exception:
#     print(f"Creating a new endpoint named {endpoint_name}...")
#     # create an online endpoint
#     # if you run into an out of quota error, change the instance_type to a comparable VM that is available.\
#     # Learn more on https://azure.microsoft.com/en-us/pricing/details/machine-learning/.
#     endpoint = ManagedOnlineEndpoint(
#         name=endpoint_name,
#         description="this is a sample endpoint",
#         auth_mode="key",
#     )
#     ml_client.online_endpoints.begin_create_or_update(endpoint).result()
#     endpoint = ml_client.online_endpoints.get(name=endpoint_name)
#     print(f"Endpoint {endpoint_name} created successfully.")



# model = ml_client.models.get(name='rbp_model', version=2)


# logger.info("Creating a new deployment...")
# # create the online deployment
# rbp_deployment = ManagedOnlineDeployment(
#     name="rbp-deployment",
#     endpoint_name=endpoint_name,
#     model=model,
#     code_configuration=CodeConfiguration(
#         code=".",
#         scoring_script="score.py",
#     ),
#     environment=env,
#     instance_type="Standard_D2a_v4",
#     instance_count=1,
# )
# ml_client.online_deployments.begin_create_or_update(
#     rbp_deployment
# ).result()
# logger.info(
#     f"Deployment {rbp_deployment.name} created with status: {rbp_deployment.provisioning_state}"
# )

# endpoint.traffic = {"rbp-deployment": 100}
# ml_client.online_endpoints.begin_create_or_update(endpoint).result()

# # blue_deployment = ManagedOnlineDeployment(
# #     name="rbp-deployment",
# #     endpoint_name='rbp-endpoint-bca685ff',
# #     model='azureml:rbp_model:2',
# #     code_configuration=CodeConfiguration(
# #         code="scoring/",
# #         scoring_script="score.py"
# #     ),
# #     environment='azureml:rbp-env1:9',
# #     instance_type="Standard_DS3_v2",
# #     instance_count=1,
# # )



# # blue deployment takes 100% traffic
# # expect the deployment to take approximately 8 to 10 minutes.

if __name__ == '__main__':
    deploy_model()