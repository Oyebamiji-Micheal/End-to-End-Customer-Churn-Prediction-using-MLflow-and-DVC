import os

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

from mlFlowProject import logger
from mlFlowProject.entity.config_entity import DataTransformationConfig



class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        
    def preprocess_data(self, data):
        data.dropna(inplace=True) # Drop missing values

        X = data.drop(columns=self.config.target_column)
        y = data[self.config.target_column]

        # Map the 'Gender' column
        mapping = {'Male': 0, 'Female': 1}
        X['Gender'] = X['Gender'].map(mapping)

        num_cols = X.select_dtypes(include=np.number).columns.to_list()
        cat_cols = X.select_dtypes(exclude=np.number).columns.to_list()

        num_pipeline = Pipeline(steps=[
            ('scaler', MinMaxScaler())
        ])

        cat_pipeline = Pipeline(steps=[
            ('one_hot_enc', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        col_trans = ColumnTransformer(transformers=[
            ('num_pipeline', num_pipeline, num_cols),
            ('cat_pipeline', cat_pipeline, cat_cols),
        ], remainder='drop', n_jobs=-1)

        X_preprocessed = col_trans.fit_transform(X)

        # Get the list of output feature names from the column transformer
        feature_names = col_trans.get_feature_names_out()

        # Create a new DataFrame from the numpy array and assign the column names to it
        X_preprocessed = pd.DataFrame(X_preprocessed, columns=feature_names)

        return X_preprocessed, y

    
    def train_test_splitting(self):
        data = pd.read_csv(self.config.data_path)

        X, y = self.preprocess_data(data)
        
        # Split the data into training and test sets. (0.75, 0.25) split.
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

        # Reset feature index 
        X_train = X_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        
        # Convert target to dataframe and reset index
        y_train_df = y_train.to_frame().reset_index(drop=True)
        y_test_df = y_test.to_frame().reset_index(drop=True)
        
        # Concatenate along columns
        train_processed = pd.concat([X_train, y_train_df], axis=1)
        test_processed = pd.concat([X_test, y_test_df], axis=1)

        # Save the processed data
        train_processed.to_csv(os.path.join(self.config.root_dir, "train.csv"), index=False)
        test_processed.to_csv(os.path.join(self.config.root_dir, "test.csv"), index=False)

        logger.info("Data splitted into training and test sets")
        logger.info(f"Shape of preprocessed training data: {train_processed.shape}")
        logger.info(f"Shape of preprocessed test data: {test_processed.shape}")
