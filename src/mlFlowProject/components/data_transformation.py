import os

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import joblib

from mlFlowProject import logger
from mlFlowProject.entity.config_entity import DataTransformationConfig



class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        """
        Initialize the DataTransformation class with a given configuration.
        """
        self.config = config

    def preprocess_data(self, data: pd.DataFrame) -> tuple:
        """
        Preprocess the data by handling missing values, converting data types, and applying transformations.

        Parameters:
        data (pd.DataFrame): The raw input data.

        Returns:
        tuple: A tuple containing the preprocessed features and target.
        """
        data = self._handle_missing_values(data)
        data = self._drop_irrelevant_columns(data)
        data = self._convert_column_types(data)

        X, y = self._separate_features_and_target(data)
        X = self._map_gender_column(X)

        num_cols, cat_cols = self._select_columns_by_type(X)

        transformer = self._create_transformer(num_cols, cat_cols)
        transformer.fit(X)
        
        X_preprocessed = transformer.transform(X)
        feature_names = transformer.get_feature_names_out()
        
        X_preprocessed = pd.DataFrame(X_preprocessed, columns=feature_names)

        self._save_transformer(transformer)

        return X_preprocessed, y

    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Drop rows with missing values.

        Parameters:
        data (pd.DataFrame): The raw input data.

        Returns:
        pd.DataFrame: The data without missing values.
        """
        return data.dropna()

    def _drop_irrelevant_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Drop irrelevant columns from the data.

        Parameters:
        data (pd.DataFrame): The input data.

        Returns:
        pd.DataFrame: The data without the dropped columns.
        """
        return data.drop(columns=['Surname'])

    def _convert_column_types(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Convert specified columns to appropriate data types.

        Parameters:
        data (pd.DataFrame): The input data.

        Returns:
        pd.DataFrame: The data with converted column types.
        """
        columns_to_convert = ['HasCrCard', 'IsActiveMember', 'Tenure', 'NumOfProducts']
        for column in columns_to_convert:
            data[column] = data[column].astype('int').astype('object')
        return data

    def _separate_features_and_target(self, data: pd.DataFrame) -> tuple:
        """
        Separate features and target variable from the data.

        Parameters:
        data (pd.DataFrame): The input data.

        Returns:
        tuple: Features (X) and target (y).
        """
        X = data.drop(columns=self.config.target_column)
        y = data[self.config.target_column]
        return X, y

    def _map_gender_column(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Map the 'Gender' column to numeric values.

        Parameters:
        X (pd.DataFrame): The feature data.

        Returns:
        pd.DataFrame: The feature data with 'Gender' column mapped.
        """
        mapping = {'Male': 0, 'Female': 1}
        X['Gender'] = X['Gender'].map(mapping)
        return X

    def _select_columns_by_type(self, X: pd.DataFrame) -> tuple:
        """
        Select numerical and categorical columns from the feature data.

        Parameters:
        X (pd.DataFrame): The feature data.

        Returns:
        tuple: Lists of numerical and categorical column names.
        """
        num_cols = X.select_dtypes(include=np.number).columns.to_list()
        cat_cols = X.select_dtypes(exclude=np.number).columns.to_list()
        return num_cols, cat_cols

    def _create_transformer(self, num_cols: list, cat_cols: list) -> ColumnTransformer:
        """
        Create a column transformer for preprocessing.

        Parameters:
        num_cols (list): List of numerical column names.
        cat_cols (list): List of categorical column names.

        Returns:
        ColumnTransformer: The column transformer.
        """
        num_pipeline = Pipeline(steps=[
            ('scaler', MinMaxScaler())
        ])

        cat_pipeline = Pipeline(steps=[
            ('one_hot_enc', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        transformer = ColumnTransformer(transformers=[
            ('num_pipeline', num_pipeline, num_cols),
            ('cat_pipeline', cat_pipeline, cat_cols),
        ], remainder='drop', n_jobs=-1)

        return transformer

    def _save_transformer(self, transformer: ColumnTransformer) -> None:
        """
        Save the fitted transformer to a file.

        Parameters:
        transformer (ColumnTransformer): The fitted column transformer.
        """
        joblib.dump(transformer, os.path.join(self.config.root_dir, self.config.preprocessor_name))

    def train_test_splitting(self) -> None:
        """
        Load data, preprocess it, and split into training and test sets.
        """
        try:
            data = pd.read_csv(self.config.data_path)
            X, y = self.preprocess_data(data)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

            X_train = X_train.reset_index(drop=True)
            X_test = X_test.reset_index(drop=True)

            y_train_df = y_train.to_frame().reset_index(drop=True)
            y_test_df = y_test.to_frame().reset_index(drop=True)

            train_processed = pd.concat([X_train, y_train_df], axis=1)
            test_processed = pd.concat([X_test, y_test_df], axis=1)

            train_processed.to_csv(os.path.join(self.config.root_dir, "train.csv"), index=False)
            test_processed.to_csv(os.path.join(self.config.root_dir, "test.csv"), index=False)

            logger.info("Data split into training and test sets")
            logger.info(f"Shape of preprocessed training data: {train_processed.shape}")
            logger.info(f"Shape of preprocessed test data: {test_processed.shape}")

        except Exception as e:
            logger.error("An error occurred during train-test splitting", exc_info=True)
            raise
