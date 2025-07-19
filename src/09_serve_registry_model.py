import mlflow
import argparse
import sys
import subprocess
import requests

def list_model_versions(model_name):
    """
    List all versions of a registered model.

    Args:
        model_name: Name of the registered model
    Returns:
        list: List of model versions
    """
    client = mlflow.tracking.MlflowClient()

    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        if not versions:
            raise Exception(f"No versions found for model '{model_name}'")

        print("\nAvailable model versions:")
        for idx, version in enumerate(versions, 1):
            status = version.current_stage
            tags = version.tags if version.tags else {}
            print(f"{idx}. Version: {version.version}, Stage: {status}")
            if tags:
                print("   Tags:")
                for key, value in tags.items():
                    print(f"   - {key}: {value}")
            print(f"   Run ID: {version.run_id}")

        return versions

    except Exception as e:
        print(f"Error listing model versions: {str(e)}")
        raise
    
def get_versions_list(model_name: str):
    """
    Get the list of registered versions of a specified model.
    If non exists, an error is raised.

    Args:
        model_name: name of model for which the version is to be selected.
    Returns:
        versions_list: a list of the versions of the specified model
    """
    
    try:
        versions_list = client.search_model_versions(f"name='{model_name}'")

    except:
        print(f'Error while trying to get versions of model "{model_name}". Was it registered, yet?')
        sys.exit(1)
        
    if not versions_list :
            raise Exception(f'Model "{model_name}" has no versions, yet.')
        
    return versions_list


def list_model_versions_a(model_name: str):
    """
    TO DO
    """
    
    client = mlflow.tracking.MlflowClient
    
    versions_list = get_versions_list(model_name)
    

    for id, version in enumerate(versions_list, 1):
        status = version.current_stage
        tags = versions.tags if version.tags else {}
        print(f"{id}. Version: {version.version}, Stage: {status}")
        if tags:
            print("   Tags:")
            for key, value in tags.items():
                print(f"   - {key}: {value}")
        print(f"   Run ID: {version.run_id}")

    return versions_list
    

def select_model_version(versions_list: list):
    """
    Prompts user to select one of the versions listed by the function "list_model_versions" if more than one found.
    If only one version present, it is selected automatically.
    
    Args: 
        versions_list: a list of model versions from which to prompt a selection.
    
    """
    
    if len(versions_list) == 1:
        print(f'Deploying version "{versions_list[0]}" as model has only that one version.')
        return versions_list[0]
    
    count = 0
    
    while count < 3 :
        count += 1
        index = int(input(f'\nPlease choose from the above versions available for that model by entering the index of the chosen version from the list above. This is try {count} of 3.'))
        if 1 <= index <= len(versions_list):
            return versions_list[index+1]
    
    raise ValueError('You used all 3 tries but did not enter a valid index. Stop wasting my time.')


def serve_model(model_uri, port):
    """
    Serve the model using MLflow's CLI command

    Args:
        model_uri: URI of the model to serve
        port: Port number to serve on
    """
    print(f"\nServing model from: {model_uri}")
    print(f"The model will be served on port {port}")

    try:
        # Use mlflow models serve command with --env-manager local
        cmd = [
            "mlflow", "models", "serve",
            "--model-uri", model_uri,
            "--port", str(port),
            "--host", "0.0.0.0",
            "--env-manager", "local"  # Use current environment
        ]

        print(f"Executing command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    except subprocess.CalledProcessError as e:
        print(f"Failed to serve model from {model_uri}")
        print(f"Error: {str(e)}")
        raise
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise

def main():
    # get arguments of calling instruction
    parser = argparse.ArgumentParser(description='Deploy model and version selected by user from versions list in MLflow Model Registry.')
    parser.add_argument('--tracking_uri', type=str, required=True, help='MLflow tracking URI.')
    parser.add_argument('--model_name', type=str, required=True, help='Name of registered model selected.')
    parser.add_argument('--version', type=str, help='Selected version of model selected (automatic if unique).')
    parser.add_argument('--port', type=str, help='Port on which to deploy the selected model.')
    
    args = parser.parse_args()

    # start tracking server
    mlflow.set_tracking_uri(args.tracking_uri)
    
    # check health of tracking server
    response = requests.get(f'{args.tracking_uri}/health')  
    if response.status_code != 200:
        raise Exception(f'MLflow server at {args.tracking_uri} not responding')
    else:
        print('Tracking server okay.')
        
        
    versions_list = list_model_versions(args.model_name)
    

    # determine version

    if args.version:
        # Find specified version
        selected_version = next((v for v in versions_list if v.version == str(args.version)), None)
        if selected_version is None:
            raise Exception(f"Version {args.version} not found for model '{args.model_name}'")
        
    else:
        selected_version = select_model_version(versions_list)

    # Construct model URI
    model_uri = f"models:/{args.model_name}/{selected_version.version}"

    # Serve model
    serve_model(model_uri, args.port)

if __name__ == "__main__":
    main()