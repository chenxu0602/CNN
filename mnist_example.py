import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Conv2D, Input, Dense, MaxPool2D, BatchNormalization, GlobalAvgPool2D
from tensorflow.python.ops.gen_array_ops import batch_to_space
from dl_models import functional_model
from my_utils import display_some_examples

seq_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.GlobalAvgPool2D(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax"),
])

if __name__ == "__main__":

    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

    print("\nX_train.shape = ", X_train.shape)
    print("y_train.shape = ", y_train.shape)
    print("X_test.shape = ", X_test.shape)
    print("y_test.shape = ", y_test.shape)

    # display_some_examples(X_train, y_train)

    X_train = X_train.astype("float32") / 255.
    X_test = X_test.astype("float32") / 255.

    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)

    print("\nX_train.shape = ", X_train.shape)
    print("y_train.shape = ", y_train.shape)
    print("X_test.shape = ", X_test.shape)
    print("y_test.shape = ", y_test.shape)


    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    print("\nX_train.shape = ", X_train.shape)
    print("y_train.shape = ", y_train.shape)
    print("X_test.shape = ", X_test.shape)
    print("y_test.shape = ", y_test.shape)


    # model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics="accuracy")
    model = functional_model()
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics="accuracy")

    batch_size = 64
    model.fit(X_train, y_train, batch_size=batch_size, epochs=3, validation_split=0.2)

    model.evaluate(X_test, y_test, batch_size=batch_size)