import os
import sys

from src.logger import logger
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

class TrainingPipeline:
    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()

        def run_pipeline(self):
            try:
                logger.info("starting training pipeline")
                X_train, X_test, y_train, y_test = (self.data_ingestion.initiate_data_ingestion())

                #data transformation
                X_train , X_test , y_train , y_test = self.data_transformation.inititate_data_transformation(X_train , X_test , y_train , y_test)

                #model training
                X_train , X_test , y_train , y_test = self.model_trainer.initiate_model_trainer(X_train , X_test , y_train , y_test)

                logger.info("training pipeline completed")
            except Exception as e:
                raise CustomException(e , sys)

if __name__ == "__main__":

    pipeline = TrainingPipeline()

    accuracy = pipeline.run_pipeline()

    print("Model Accuracy:", accuracy)