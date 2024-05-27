import joblib 
import numpy as np
import pandas as pd
from pathlib import Path


class PredictionPipeline:
    def __init__(self):
        self.preprocessor = joblib.load('artifacts/data_transformation/preprocessor.joblib')
        self.model = joblib.load(Path('artifacts/model_trainer/model.joblib'))
    
    def predict(self, data):
        data = self.preprocessor.transform(data)
        prediction = self.model.predict(data)

        return prediction
