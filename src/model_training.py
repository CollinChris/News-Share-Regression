# Standard library imports
import logging
from typing import Any, Dict, Tuple

# Related third-party imports
import pandas as pd
import numpy as np
import importlib
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

class ModelTraining:
    """
    A class to train and evaluate ML models on news article shares dataset.

    Attributes:
    ------------
    config: (Dict[str, Any])
        Configuration dictionary containing parameters for model training & evauluation.
    preprocessor: sklearn.compose.ColumnTransformer
        A preprocessor pipeline for log transforming numerical and nominal features usning RobustScaler.
    """

    def __init__(self, config: Dict[str, Any], preprocessor: ColumnTransformer):
        """
        Initialize model_training class with config & preprocessor.

        Args:
        ------
            config(Dict[str, Any]): Configuration dictionary containing parameters for model training & evaluation.
            preprocessor(sklearn.compose.ColumnTransformer): A preprocessor pipeline for transforming numerical, nominal, and ordinal features.
        """
        self.config = config
        self.preprocessor = preprocessor

    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Splits data into training, validation, and test sets.

        Args:
        -----
        df(pd.DataFrame): The input DataFrame containing the cleaned data.

        Returns:
        --------
        Tuple[pd.DataFrame..., pd.Series...]: A tuple containing the training, validation, and test features(pd.DataFrame, X) and target variables(Series, y).
        """
        logging.info("Starting data splitting.")
        # Separate data into features and target variable
        X = df.drop(columns= self.config["target_column"])
        # Log-transform 'shares' to reduce skewness
        y = np.log1p(df[self.config["target_column"]])
        # Split the data into training (80%) and test-validation (20%) sets
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size= self.config["val_test_size"], random_state=42)
        # Split the test-validation set (20%) into validation (10%) and test (10%) sets
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size= self.config["val_size"], random_state=42)
        logging.info("Data splitting completed.")
        return X_train, X_val, X_test, y_train, y_val, y_test

    def train_and_evaluate_baseline_models(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series,
                                           ) -> Tuple[Dict[str, Pipeline], Dict[str, Dict[str, float]]]:
        """
        Train the *baseline* versions (the `params` block in config.yaml) and
        return the fitted pipelines + validation metrics.

        Args:
        -----
        X_train(pd.DataFrame): Training features.
        y_train(pd.Series): Training target variable.
        X_val(pd.DataFrame): Validation features.
        y_val(pd.Series): Validation target variable.

        Returns:
        --------
        Tuple[Dict[str, Pipeline], Dict[str, Dict[str, float]]]: A tuple containing: pipeles and metrics.
        """
        logging.info("Starting baseline model training & evaluation.")
        pipelines = {}
        metrics = {}

        for model_name, cfg in self.config["baseline_models"].items():
            # Load class
            module_name, class_name = cfg["model"].rsplit(".", 1)
            model_class = getattr(importlib.import_module(module_name), class_name)

            # Instantiate with baseline params
            model = model_class(**cfg["params"])

            # Build pipeline
            pipeline = Pipeline(
                steps=[
                    ("preprocessor", self.preprocessor),
                    ("model", model),
                ]
            )

            pipeline.fit(X_train, y_train)
            pipelines[model_name] = pipeline

            metrics[model_name] = self._evaluate_model(
                pipeline, X_val, y_val, model_name
            )
            logging.info(f"{model_name} baseline trained & evaluated.")

        return pipelines, metrics


    def train_and_evaluate_tuned_models(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series,
                                        ) -> Tuple[Dict[str, Pipeline], Dict[str, Dict[str, float]]]:
        """
        Perform GridSearchCV **per model** using the `param_grid` defined in
        `config.yaml`.  Models without a `param_grid` (or `null`) are skipped.

        Args:
        -----
        X_train(pd.DataFrame): Training features.
        y_train(pd.Series): Training target variable.
        X_val(pd.DataFrame): Validation features.
        y_val(pd.Series): Validation target variable.
        Returns:
        --------
        Tuple[Dict[str, Pipeline], Dict[str, Dict[str, float]]]: A tuple containing: tuned pipeles and metrics.

        """
        logging.info("Starting hyper-parameter tuning & evaluation.")
        tuned_models = {}
        tuned_metrics = {}

        # Optional global CV / scoring (you can keep them in config or hard-code)
        cv = self.config.get("cv", 5)
        scoring = self.config.get("scoring", "r2")

        for model_name, cfg in self.config["baseline_models"].items():
            # Load class
            module_name, class_name = cfg["model"].rsplit(".", 1)
            model_class = getattr(importlib.import_module(module_name), class_name)

            # Use *baseline* params
            base_model = model_class(**cfg["params"])

            pipeline = Pipeline(
                steps=[
                    ("preprocessor", self.preprocessor),
                    ("model", base_model),
                ]
            )

            # Get param_grid
            param_grid = cfg.get("param_grid")
            if not param_grid:
                logging.info(f"No param_grid for {model_name} – skipping tuning.")
                continue

            # Run GridSearchCV
            grid = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
                verbose=1,
            )
            grid.fit(X_train, y_train)

            # Store best estimator & evaluate
            best_pipe = grid.best_estimator_
            tuned_models[model_name] = best_pipe
            tuned_metrics[model_name] = self._evaluate_model(
                best_pipe, X_val, y_val, f"{model_name} (Tuned)"
            )
            logging.info(
                f"{model_name} tuning finished – best CV {scoring}: {grid.best_score_:.4f}"
            )

        return tuned_models, tuned_metrics

    def evaluate_final_model(self, model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, model_name: str
                             ) -> Dict[str, float]:
        """
        Evaluate final model on test set and log metrics.

        Args:
        -----
        model(sklearn.pipeline.Pipeline): The trained model pipeline.
        X_test(pd.DataFrame): Test features.
        y_test(pd.Series): Test target variable.(HDB resale price)
        model_name(str): Name of the model.

        Returns:
        --------
        Dict[str, float]: A dictionary containing evaluation metrics.
        """
        y_test_pred = model.predict(X_test)
        metrics = {"MAE": mean_absolute_error(y_test, y_test_pred),
                   "MSE": mean_squared_error(y_test, y_test_pred),
                   "RMSE": root_mean_squared_error(y_test, y_test_pred),
                   "r2": r2_score(y_test, y_test_pred),
                   "mae_orig" : mean_absolute_error(np.expm1(y_test), np.expm1(y_test_pred))}
        logging.info(f"Final test metrics for {model_name}:")
        for metric_name, metric_value in metrics.items():
            logging.info(f"{metric_name}: {metric_value}")
        return metrics
    
    def _evaluate_model(self, model: Pipeline, X_val: pd.DataFrame, y_val: pd.Series, model_name: str
                        ) -> Dict[str, float]:
        """
        Evaluate model on validation set and return metrics.

        Args:
        -----
        model(sklearn.pipeline.Pipeline): The trained model pipeline.
        X_val(pd.DataFrame): Validation features.
        y_val(pd.Series): Validation target variable.
        model_name(str): Name of the model.

        Returns:
        --------
        Dict[str, float]: A dictionary containing evaluation metrics.
        """
        y_val_pred = model.predict(X_val)
        metrics = {"MAE": mean_absolute_error(y_val, y_val_pred),
                   "MSE": mean_squared_error(y_val, y_val_pred),
                   "RMSE": root_mean_squared_error(y_val, y_val_pred),
                   "r2": r2_score(y_val, y_val_pred),
                   "mae_orig" : mean_absolute_error(np.expm1(y_val), np.expm1(y_val_pred))}  # Original scale
        logging.info(f"Validation metrics for {model_name}:")
        for metric_name, metric_value in metrics.items():
            logging.info(f"{metric_name}: {metric_value}")
        return metrics