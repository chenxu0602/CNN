import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Conv2D, Input, Dense, MaxPool2D, BatchNormalization, GlobalAvgPool2D

def display_some_examples(examples, labels):
    plt.figure(figsize=(10, 10))

    for i in range(25):
        idx = np.random.randint(0, examples.shape[0] - 1)
        img = examples[idx]
        lbl = labels[idx]

        plt.subplot(5, 5, i + 1)
        plt.title(str(lbl))
        plt.tight_layout()
        plt.imshow(img, cmap="gray")

    plt.show()

model = tf.keras.Sequential([
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

    model.compile(optimizer="adam", loss="SparseCategoricalCrossentropy", metrics="accuracy")

    model.fit(X_train, y_train, batch_size=64)