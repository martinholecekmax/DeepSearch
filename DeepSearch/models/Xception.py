import os

from keras.models import load_model
from keras.models import Model
from keras.applications.xception import Xception, preprocess_input
from keras.layers import Input


def get_model():
    package_directory = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(package_directory, "weights/Xception.h5")
    if not os.path.exists(weights_path):
        print("Downloading Xception model...")
        base_model = Xception(weights="imagenet", input_tensor=Input(shape=(224, 224, 3)))
        model = Model(inputs=base_model.input, outputs=base_model.get_layer("avg_pool").output)
        model.save(weights_path)
    else:
        model = load_model(weights_path, compile=False)
    return model


def preprocess_image(image):
    return preprocess_input(image)
