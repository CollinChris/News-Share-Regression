# Import necessary libraries
from typing import Dict, Any
import re
import logging

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, FunctionTransformer

class DataPrep:
    """
    A class used to clean and preprocess data(In this case news article data).

    Attributes:
    -----------
    config (Dict[str, Any]): Configuration dictionary containing parameters for data cleaning and preprocessing.
    preprocessor (ColumnTransformer): A preprocessor pipeline for transforming numerical, nominal, and ordinal features.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize DataPrep class with config.

        Args:
        -----
        config (Dict[str, Any]): Configuration dictionary containing parameters for data cleaning and preprocessing.
        """
        self.config = config
        self.preprocessor = self._create_preprocessor()

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the input DataFrame by performing several preprocessing steps.
        
        Args:
        -----
        df (pd.DataFrame): The input raw data.
            
        Returns:
        --------
        pd.DataFrame: Cleaned DataFrame.
        """
        # Function body
        logging.info("Starting data cleaning.")
        # Replace negative values
        df = self._replace_negative_values(df)
        # Fill missing values
        df = self._fill_missing_features(df)
        # Create new features
        df = self._create_new_features(df)
        # Drop columns from config
        df = df.drop(columns=self.config["columns_to_drop"])
        # Cap outliers at 99th percentile for shares and n_comments
        for col in ['shares', 'n_comments']:
            p99 = np.percentile(df[col], 99)
            df[col] = df[col].clip(upper=p99)
            logging.info(f"Capped {col} at 99th percentile: {p99}")
        logging.info("Data cleaning completed.")
        return df

    def _create_preprocessor(self) -> ColumnTransformer:
        """
        Creates preprocessor pipeline to log transform numerical and nominal features.

        Returns:
        --------
        ColumnTransformer: A preprocessor pipeline.
        """
        # Transformer pipelines
        numerical_transformer = Pipeline(steps=[('log_transform', FunctionTransformer(np.log1p, validate=True)),
                                                ('scaler', RobustScaler())])
        nominal_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
        # Combine transformers into a ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[('num', numerical_transformer, self.config['numerical_features']),
                          ('nom', nominal_transformer, self.config['nominal_features']),
            ],
            remainder="passthrough",
            n_jobs=-1)
        return preprocessor        

    @staticmethod
    def _replace_negative_values(df: pd.DataFrame) -> pd.DataFrame:
        """
        Replace negative values in 'kw_min_min', 'kw_avg_min', 'kw_min_avg'.

        Args:
        -----
        df(pd.DataFrame): The Dataframe containing columns to be filled.

        Returns:
        --------
        pd.DataFrame: DataFrame with negative values replaced.
        """
        keyword_cols = ['kw_min_min', 'kw_avg_min', 'kw_min_avg']
        for col in keyword_cols:
            if col in df.columns:
                num_neg = (df[col] < 0).sum()
                if num_neg > 0:
                    logging.info(f"Replaced {num_neg} negative values in '{col}'")
                df.loc[df[col] < 0, col] = 0
        return df
    
    @staticmethod
    def _fill_missing_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Fills missing values in 'num_videos', 'data_channel', 'num_imgs', 'num_self_hrefs', 'self_reference_max_shares', 
                    'self_reference_avg_shares', 'self_reference_min_shares', 'num_hrefs'.

        Args:
        -----
        df(pd.DataFrame): The Dataframe containing columns to be filled.

        Returns:
        --------
        pd.DataFrame: DataFrame with missing values in 'num_videos', 'data_channel', 'num_imgs', 'num_self_hrefs', 
                    'self_reference_max_shares', 'self_reference_avg_shares', 'self_reference_min_shares', 'num_hrefs' filled.
        """
        # Fill data_channel missing values with 'unknown'
        df['data_channel'] = df['data_channel'].fillna('unknown')
        # Create missing feature for num_videos and fill missing with 0
        df['is_missing_videos'] = df['num_videos'].isnull().astype(int)
        df['num_videos'] = df['num_videos'].fillna(0)
        # Fill other features with 0
        df['num_imgs'] = df['num_imgs'].fillna(0)
        link_cols = ['num_hrefs', 'num_self_hrefs', 
             'self_reference_min_shares', 'self_reference_max_shares', 'self_reference_avg_shares']
        df[link_cols] = df[link_cols].fillna(0)
        logging.info('Missing values filled.')
        return df
    
    @staticmethod
    def _create_new_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates new features (data_channel x comments), keyword_ratio, time_comments,comments_selfref_avg, comments_non_stop_words  .

        Args:
        -----
        df(pd.DataFrame): The Dataframe containing columns to be filled.

        Returns:
        --------
        pd.DataFrame: DataFrame with new features
        """
        df['time_comments'] = df['timedelta'] * df['n_comments'] 
        df['comments_selfref_avg'] = df['n_comments'] * df['self_reference_avg_shares']
        df['comments_non_stop_words'] = df['n_comments'] * df['n_non_stop_words']
        # Create one-hot-encoded data_channel columns
        data_channel_dummies = pd.get_dummies(df['data_channel'], prefix='data_channel')
        # Create interaction terms: n_comments * each data_channel
        for col in data_channel_dummies.columns:
            df[f'n_comments_{col}'] = df['n_comments'] * data_channel_dummies[col]
        logging.info('New features created.')
        return df