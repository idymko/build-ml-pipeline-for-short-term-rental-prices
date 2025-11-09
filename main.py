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
                parameters = {
                    "csv": "clean_sample.csv:latest",
                    "ref": "clean_sample.csv:reference",
                    "kl_threshold": config["data_check"]["kl_threshold"],
                    "min_price": config["etl"]["min_price"],
                    "max_price": config["etl"]["max_price"]
                }
            )

        if "data_split" in active_steps:
            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "components", "train_val_test_split"),
                "main",
                parameters = {
                    "input": "clean_sample.csv:latest",
                    "test_size": config["modeling"]["test_size"],
                    "random_seed": config["modeling"]["random_seed"],
                    "stratify_by": config["modeling"]["stratify_by"]
                }
            )

        if "train_random_forest" in active_steps:

            # NOTE: we need to serialize the random forest configuration into JSON
            rf_config = os.path.abspath("rf_config.json")
            with open(rf_config, "w+") as fp:
                json.dump(dict(config["modeling"]["random_forest"].items()), fp)  # DO NOT TOUCH

            # NOTE: use the rf_config we just created as the rf_config parameter for the train_random_forest
            # step

            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "src", "train_random_forest"),
                "main",
                parameters={
                    "trainval_artifact": "trainval_data.csv:latest",
                    "val_size": config["modeling"]["val_size"],
                    "random_seed": config["modeling"]["random_seed"],
                    "stratify_by": config["modeling"]["stratify_by"],
                    "rf_config": rf_config,
                    "max_tfidf_features": config["modeling"]["max_tfidf_features"],
                    "output_artifact": "random_forest_export"
                }
            )

        if "test_regression_model" in active_steps:

            ##################
            # Implement here #
            ##################

            pass


if __name__ == "__main__":
    
    # run step-by-step:
        # mlflow run . -P steps=download
        # mlflow run src/eda
        # mlflow run . -P steps=basic_cleaning
        # mlflow run . -P steps=data_check
        # mlflow run . -P steps=data_split
        # mlflow run . -P steps=train_random_forest
        
    # hyperparameter tuning 
    """
        mlflow run . \
        -P steps=train_random_forest \
        -P hydra_options="modeling.random_forest.max_depth=10,50,100 modeling.random_forest.n_estimators=100,200,500 -m"
        
        mlflow run . \
        -P steps=train_random_forest \
        -P hydra_options="modeling.max_tfidf_features=10,15,30 modeling.random_forest.max_features=0.1,0.33,0.5,0.75,1. -m"
    """
    go()
