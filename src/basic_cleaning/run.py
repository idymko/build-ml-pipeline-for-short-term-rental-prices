#!/usr/bin/env python
"""
Performs basic cleaning on the data and saves the results in W&B
"""
import argparse
import logging
import wandb
import pandas as pd
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    logger.info(f"Downloading artifact {args.input_artifact}")
    artifact_local_path = run.use_artifact(args.input_artifact).file()
    df = pd.read_csv(artifact_local_path)

    logger.info(f"Drop price outliers setting them between {args.min_price} and {args.max_price}")
    idx = df['price'].between(args.min_price, args.max_price)
    df = df[idx].copy()
    
    logger.info(f"Convert last_review to dataframe")
    df['last_review'] = pd.to_datetime(df['last_review'])
    
    logger.info(f"Save the results to a CSV file")
    #       NOTE: Remember to use index=False when saving to CSV,
    #       otherwise the data checks in the next step might fail
    #       because there will be an extra index column
    
    filename = "clean_sample.csv"
    df.to_csv(filename, index=False)
    
    artifact = wandb.Artifact(
        name = args.output_artifact,
        type = args.output_type,
        description = args.output_description
    )
    artifact.add_file(filename)
    
    logger.info(f"Logging artifact to {args.output_artifact}")
    run.log_artifact(artifact)
    
    os.remove(filename)

if __name__ == "__main__":
    
    """
    TIP: Create "src/basic_cleaning" automatically by using cookiecutter
    > cookiecutter cookie-mlflow-step -o src
        step_name [step_name]: basic_cleaning
        script_name [run.py]: run.py
        job_type [my_step]: basic_cleaning
        short_description [My step]: A very basic data cleaning
        long_description [An example of a step using MLflow and Weights & Biases]: Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
        parameters [parameter1,parameter2]: input_artifact,output_artifact,output_type,output_description,min_price,max_price
    """

    parser = argparse.ArgumentParser(description="This step cleans the data")

    parser.add_argument(
        "--input_artifact", 
        type=str,
        help="Name of input input datafile",
        required=True
    )
    parser.add_argument(
        "--output_artifact", 
        type=str,
        help="Name of output cleaned file",
        required=True
    )
    parser.add_argument(
        "--output_type", 
        type=str,
        help="Type of output file",
        required=True
    )
    parser.add_argument(
        "--output_description", 
        type=str,
        help="Decsription of output file",
        required=True
    )
    parser.add_argument(
        "--min_price", 
        type=float,
        help="Minimum pirce in the cleaned dataset",
        required=True
    )
    parser.add_argument(
        "--max_price", 
        type=float,
        help="Maximum pirce in the cleaned dataset",
        required=True
    )

    args = parser.parse_args()

    go(args)
