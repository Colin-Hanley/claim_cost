import json
from azure.ai.ml.entities import ManagedOnlineEndpoint
from azure.ai.ml import MLClient
from azure.identity import AzureCliCredential

if __name__ == "__main__":
    endpoint_name = "no-loss-endpoint"

    with open("config.json") as f:
        config = json.load(f)

    ml_client = MLClient(AzureCliCredential(),
                         config["subscription_id"],
                         config["resource_group"],
                         config["workspace_name"])

    endpoint = ManagedOnlineEndpoint(
        name=endpoint_name,
    )

    ml_client.begin_create_or_update(endpoint).result()
    print(f"Endpoint {endpoint_name} created successfully")


