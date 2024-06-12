import json
import pathlib

from azure.ai.ml.entities import ManagedOnlineDeployment, CodeConfiguration
from azure.identity import AzureCliCredential
from azure.ai.ml import MLClient

deployment_name = "claims-inference-deployment"

if __name__ == "__main__":
    with open("config.json") as f:
        config = json.load(f)

    ml_client = MLClient(AzureCliCredential(),
                         config["subscription_id"],
                         config["resource_group"],
                         config["workspace_name"])

    model = ml_client.models.get(
        name="claims_model",
        version="8"
    )

    claims_model_deployment = ManagedOnlineDeployment(
        name=deployment_name,
        endpoint_name="claims-inference-endpoint",
        model=model,
        environment="Claims_env:2",
        code_configuration=CodeConfiguration(
            code=str(pathlib.Path.cwd().parent),
            scoring_script="predict_claim_cost.py")
    )

    ml_client.online_deployments.begin_create_or_update(
        deployment=claims_model_deployment
    ).result()

    print("Deployment created successfully")
