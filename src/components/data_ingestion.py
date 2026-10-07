from tensorflow.keras.datasets import mnist
from src.logger import logger
from src.exception import CustomException


class DataIngestion:

    def initiate_data_ingestion(self):
        try:
            logger.info("Starting Data Ingestion")
            (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
            logger.info("MNIST data loaded successfully")
            logger.info(f"Training data shape: {X_train.shape}")
            logger.info(f"Testing data shape: {X_test.shape}")

            return X_train, X_test, Y_train, Y_test
        except Exception as e:
            raise CustomException(e)