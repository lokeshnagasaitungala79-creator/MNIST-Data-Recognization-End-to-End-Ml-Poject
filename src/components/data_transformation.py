import numpy as np

from src.logger import logger
from src.exception import CustomException


class DataTransformation:

    def initiate_data_transformation(
        self,
        X_train,
        X_test,
        y_train,
        y_test
    ):

        try:

            logger.info("Starting data transformation")

            X_train = X_train.astype("float32") / 255.0
            X_test = X_test.astype("float32") / 255.0

            X_train = np.expand_dims(X_train, axis=-1)
            X_test = np.expand_dims(X_test, axis=-1)

            logger.info("Data transformation completed")

            return X_train, X_test, y_train, y_test

        except Exception as e:

            raise CustomException(e, None)