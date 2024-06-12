import json
import pathlib

from azure.ai.ml.entities import ManagedOnlineDeployment, CodeConfiguration
from azure.identity import AzureCliCredential
from azure.ai.ml import MLClient

deployment_name = "claim-loss-deployment"

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

    apr_model_deployment = ManagedOnlineDeployment(
        name=deployment_name,
        endpoint_name="no-loss-endpoint",
        model=model,
        environment="Claims_env:2",
        code_configuration=CodeConfiguration(
            code=str(pathlib.Path.cwd().parent / "claim_cost"),
            scoring_script="predict_claim_cost.py")
        ,
        environment_variables={
            "AZURE_SUBSCRIPTION_ID": config["subscription_id"],
            "AZURE_RESOURCE_GROUP": config["resource_group"],
            "AZURE_MACHINE_LEARNING_WORKSPACE": config["workspace_name"],
        }
    )
    ml_client.online_deployments.begin_create_or_update(
        deployment=apr_model_deployment
    ).result()

    print("Deployment created successfully")
