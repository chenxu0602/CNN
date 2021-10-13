import numpy as np
import tensorflow as tf

def predict_with_model(model, imgpath):
    image = tf.io.read_file(imgpath)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.resize(image, [60, 60]) # (60, 60, 3)
    image = tf.expand_dims(image, axis=0) # (1, 60, 60, 3)

    predictions = model.predict(image)
    predictions = np.argmax(predictions)

    return predictions


if __name__ == "__main__":
    img_path = "GTSRB/Test/0/00807.png"

    model = tf.keras.models.load_model("./Models")
    prediction = predict_with_model(model, img_path)

    print(f"prediction = {prediction}")