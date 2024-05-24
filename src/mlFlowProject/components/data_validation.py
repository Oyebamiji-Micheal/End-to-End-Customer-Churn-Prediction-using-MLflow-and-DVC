import os
import pandas as pd

from mlFlowProject import logger
from mlFlowProject.entity.config_entity import DataValidationConfig


class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config
    
    def validate_all_columns(self)-> bool:
        try:
            validation_status = None

            data = pd.read_csv(self.config.unzip_data_dir)
            data_columns = list(data.columns)
            data_dtypes =  data.dtypes.astype(str).tolist()

            schema = self.config.all_schema
            schema_cols = self.config.all_schema.keys()

            for col, dtype in zip(data_columns, data_dtypes):
                if (col not in schema_cols) or (schema.get(col) != dtype):
                    validation_status = False
                    with open(self.config.STATUS_FILE, 'w') as f:
                        f.write(f"Validation status: {validation_status}")
                        break
                else:
                    validation_status = True
                    with open(self.config.STATUS_FILE, 'w') as f:
                        f.write(f"Validation status: {validation_status}")

            return validation_status
        
        except Exception as e:
            raise e