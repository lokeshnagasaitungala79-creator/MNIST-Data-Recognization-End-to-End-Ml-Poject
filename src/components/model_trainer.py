import os
import tensorflow as tf


class ModelTrainer:

    def initiate_model_trainer(self):

        print("Loading MNIST dataset...")

        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

        print("Dataset loaded successfully")

        x_train = x_train.astype("float32") / 255.0
        x_test = x_test.astype("float32") / 255.0

        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)

        print("Building CNN model...")

        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(28, 28, 1)),

            tf.keras.layers.Conv2D(
                32,
                (3, 3),
                activation="relu"
            ),

            tf.keras.layers.MaxPooling2D(
                (2, 2)
            ),

            tf.keras.layers.Conv2D(
                64,
                (3, 3),
                activation="relu"
            ),

            tf.keras.layers.MaxPooling2D(
                (2, 2)
            ),

            tf.keras.layers.Flatten(),

            tf.keras.layers.Dense(
                128,
                activation="relu"
            ),

            tf.keras.layers.Dense(
                10,
                activation="softmax"
            )
        ])

        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        print("Training model...")

        model.fit(
            x_train,
            y_train,
            validation_data=(x_test, y_test),
            epochs=5,
            batch_size=32
        )

        print("Creating artifacts folder...")

        os.makedirs("artifacts", exist_ok=True)

        model_path = "artifacts/mnist_model.keras"

        print("Saving model...")

        model.save(model_path)

        print(f"Model saved successfully: {model_path}")


if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.initiate_model_trainer()