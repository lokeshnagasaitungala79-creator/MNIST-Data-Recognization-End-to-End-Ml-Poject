import os
import numpy as np
import tensorflow as tf
from PIL import Image


class PredictionPipeLine:

    def __init__(self):

        project_root = os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.abspath(__file__)
                )
            )
        )

        model_path = os.path.join(
            project_root,
            "artifacts",
            "mnist_model.keras"
        )

        self.model = tf.keras.models.load_model(model_path)

    def predict(self, image):

        if isinstance(image, Image.Image):
            image = image.convert("L")
            image = image.resize((28, 28))
            image = np.array(image)

        image = image.astype("float32") / 255.0
        image = np.expand_dims(image, axis=0)
        image = np.expand_dims(image, axis=-1)

        prediction = self.model.predict(image, verbose=0)

        return int(np.argmax(prediction))