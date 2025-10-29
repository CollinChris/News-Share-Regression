# Standard library imports
import logging

# Third-party imports
import pandas as pd
import yaml
from sklearn.utils._testing import ignore_warnings

# Local application/library specific imports
from src.data_preparation import DataPrep
from src.model_training import ModelTraining

logging.basicConfig(level=logging.INFO)

@ignore_warnings(category=Warning)
def main():

    # Configuration file path
    config_path = "./src/config.yaml"
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    
    # Load CSV file
    df = pd.read_csv(config["file_path"])
    # Run data preparation
    data_prep = DataPrep(config)
    cleaned_df = data_prep.clean_data(df)

    # Initialize model training 
    model_training = ModelTraining(config, data_prep.preprocessor)
    # Split the data
    X_train, X_val, X_test, y_train, y_val, y_test = model_training.split_data(cleaned_df)
    # Train and evaluate baseline models with default hyperparameters
    baseline_models, baseline_metrics = model_training.train_and_evaluate_baseline_models(X_train, y_train, X_val, y_val)

    # Train and evaluate models with hyperparameter tuning
    tuned_models, tuned_metrics = model_training.train_and_evaluate_tuned_models(X_train, y_train, X_val, y_val)

    # Combine all models and their metrics into dictionaries
    all_models = {**baseline_models, **tuned_models}
    all_metrics = {**tuned_models, **tuned_metrics}

    # Find the best model based on R² score
    best_model_name = max(all_metrics, key= lambda k: all_metrics[k]["r2"])
    best_model = all_models[best_model_name]
    logging.info(f"Best model found: {best_model_name}")

    # re-fit the best pipeline on train + val
    X_train_full = pd.concat([X_train, X_val], axis=0)
    y_train_full = pd.concat([y_train, y_val], axis=0)
    best_model.fit(X_train_full, y_train_full)

    # Evaluate best model on test set
    final_metrics = model_training.evaluate_final_model(best_model, X_test, y_test, best_model_name)
    logging.info(f"Final evalation metrics: {final_metrics}")

if __name__ == "__main__":
    main()
