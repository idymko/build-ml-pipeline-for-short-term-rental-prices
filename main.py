import json

import mlflow
import tempfile
import os
import wandb
import hydra
from omegaconf import DictConfig

_steps = [
    "download",
    "basic_cleaning",
    "data_check",
    "data_split",
    "train_random_forest",
    # NOTE: We do not include this in the steps so it is not run by mistake.
    # You first need to promote a model export to "prod" before you can run this,
    # then you need to run this step explicitly
#    "test_regression_model"
]


# This automatically reads in the configuration
# Added config_path to avoid error: Cannot find primary config 'config'.
@hydra.main(version_base=None, config_path=".", config_name='config')  # Adding version_base for Python 3.13 compatibility
def go(config: DictConfig):
    
    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # Steps to execute
    steps_par = config['main']['steps']
    active_steps = steps_par.split(",") if steps_par != "all" else _steps
    
    # You can get the path at the root of the MLflow project with this:
    root_path = hydra.utils.get_original_cwd()
    
    # Move to a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:

        if "download" in active_steps:
            # Download file and load in W&B
            _ = mlflow.run(
                ## run form the repository (initial)
                # f"{config['main']['components_repository']}/get_data",
                # os.path.join(root_path, "components/get_data"),
                # "main",
                # version='main',
                # env_manager="conda",
                
                ## run from current project (preffered)
                #os.path.join(root_path, "components/get_data"),
                os.path.join(hydra.utils.get_original_cwd(), "components", "get_data"),
                "main",
                parameters={
                    "sample": config["etl"]["sample"],
                    "artifact_name": "sample.csv",
                    "artifact_type": "raw_data",
                    "artifact_description": "Raw file as downloaded"
                },
            )

        if "basic_cleaning" in active_steps:
            """
                accept the parameters:
                    input_artifact (the input artifact), 
                    output_artifact (the name for the output artifact), 
                    output_type (the type for the output artifact), 
                    output_description (a description for the output artifact), 
                    min_price (the minimum price to consider) and 
                    max_price (the maximum price to consider)
                    
                NOTE: Remember that when you refer to an artifact stored on W&B, 
                    you MUST specify a version or a tag. 
                    For example, here the input_artifact should be sample.csv:latest 
                    and NOT just sample.csv. If you forget to do this, you will see 
                    a message like Attempted to fetch artifact without alias 
                    (e.g. "<artifact_name>:v3" or "<artifact_name>:latest")
                    
            """
            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "src", "basic_cleaning"),
                "main",
                parameters={
                    "input_artifact": "sample.csv:latest", 
                    "output_artifact": "clean_sample.csv", 
                    "output_type": "clean_sample", 
                    "output_description": "Data with outliers and null values removed",
                    "min_price": config["etl"]["min_price"],
                    "max_price": config["etl"]["max_price"]
                }
            )

        if "data_check" in active_steps:
            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "src", "data_check"),
                "main",
                # command: "pytest . -vv 
                # --csv {csv}
                # --ref {ref}
                # --kl_threshold {kl_threshold} 
                # --min_price {min_price} 
                # --max_price {max_price}"
                parameters = {
                    "csv": "clean_sample.csv:latest",
                    "ref": "clean_sample.csv:reference",
                    "kl_threshold": config["data_check"]["kl_threshold"],
                    "min_price": config["etl"]["min_price"],
                    "max_price": config["etl"]["max_price"]
                }
            )

        if "data_split" in active_steps:
            ##################
            # Implement here #
            ##################
            pass

        if "train_random_forest" in active_steps:

            # NOTE: we need to serialize the random forest configuration into JSON
            rf_config = os.path.abspath("rf_config.json")
            with open(rf_config, "w+") as fp:
                json.dump(dict(config["modeling"]["random_forest"].items()), fp)  # DO NOT TOUCH

            # NOTE: use the rf_config we just created as the rf_config parameter for the train_random_forest
            # step

            ##################
            # Implement here #
            ##################

            pass

        if "test_regression_model" in active_steps:

            ##################
            # Implement here #
            ##################

            pass


if __name__ == "__main__":
    
    # run only one step
    # mlflow run . -P steps=download
    # mlflow run src/eda
    # mlflow run . -P steps=basic_cleaning
    # mlflow run . -P steps=data_check
    
    go()
