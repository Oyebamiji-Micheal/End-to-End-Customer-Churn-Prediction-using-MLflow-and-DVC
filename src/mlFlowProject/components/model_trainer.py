import os

import pandas as pd
from lightgbm import LGBMClassifier
import joblib

from mlFlowProject import logger
from mlFlowProject.entity.config_entity import ModelTrainerConfig


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        
    
    def train(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)

        train_x = train_data.drop([self.config.target_column], axis=1)
        test_x = test_data.drop([self.config.target_column], axis=1)
        train_y = train_data[[self.config.target_column]]
        test_y = test_data[[self.config.target_column]]

        lgbm_clf = LGBMClassifier(n_estimators=self.config.n_estimators, num_leaves=self.config.num_leaves, learning_rate=self.config.learning_rate, max_depth= self.config.max_depth)
        lgbm_clf.fit(train_x, train_y)

        joblib.dump(lgbm_clf, os.path.join(self.config.root_dir, self.config.model_name))