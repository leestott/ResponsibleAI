import sys
from azureml.core import Workspace
from azureml.core.authentication import ServicePrincipalAuthentication


def get_workspace(
        workspace_name: str,
        resource_group: str,
        subscription_id: str):

    try:
        aml_workspace = Workspace.get(
            name=workspace_name,
            subscription_id=subscription_id,
            resource_group=resource_group)

        return aml_workspace
    except Exception as caught_exception:
        print("Error while retrieving Workspace...")
        print(str(caught_exception))
        sys.exit(1)
