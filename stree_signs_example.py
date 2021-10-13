import os, sys
import glob, shutil
from sklearn.model_selection import train_test_split
from my_utils import order_test_set, split_data, order_test_set, create_generator
from dl_models import streesigns_model
import tensorflow as tf


if __name__ == "__main__":
    # path_to_data = "GTSRB/Train"
    # path_to_save_train = "GTSRB/data/train"
    # path_to_save_val = "GTSRB/data/val"
    # path_to_images = "GTSRB/Test"
    # path_to_csv = "GTSRB/Test.csv"

    # order_test_set(path_to_images, path_to_csv)

    # split_data(path_to_data, path_to_save_train, path_to_save_val)

    TRAIN = False

    path_to_train = "GTSRB/data/train"
    path_to_val = "GTSRB/data/val"
    path_to_test = "GTSRB/Test"

    batch_size = 64
    epochs = 15

    lr = 1e-4

    train_generator, val_generator, test_generator = create_generator(batch_size, path_to_train, path_to_val, path_to_test)
    nbr_classes = train_generator.num_classes

    path_to_save_model = "./Models"
    ckpt_saver = tf.keras.callbacks.ModelCheckpoint(
        path_to_save_model,
        monitor="val_accuracy",
        mode="max",
        # monitor="val_loss",
        # mode="min"
        save_best_only=True,
        save_freq="epoch",
        verbose=1,
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=10)

    model = streesigns_model(nbr_classes)
    # Note in my_utils.create_geneorator the label is categorical
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr, amsgrad=True)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics="accuracy")

    if TRAIN:
        model.fit(train_generator, 
                epochs=epochs,
                batch_size=batch_size,
                validation_data=val_generator,
                callbacks=[ckpt_saver, early_stop])


    model = tf.keras.models.load_model("./Models")
    model.summary()

    print("Evaluating validation set: \n")
    model.evaluate(val_generator)

    print("Evaluating test set: \n")
    model.evaluate(test_generator)

