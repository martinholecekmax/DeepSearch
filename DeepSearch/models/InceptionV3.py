import os

from keras.models import load_model
from keras.models import Model
from keras.layers import Input
from keras.applications.inception_v3 import InceptionV3, preprocess_input


def get_model():
    package_directory = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(package_directory, "weights/InceptionV3.h5")
    if not os.path.exists(weights_path):
        print("Downloading InceptionV3 model...")
        base_model = InceptionV3(weights="imagenet", input_tensor=Input(shape=(224, 224, 3)))
        model = Model(inputs=base_model.input, outputs=base_model.get_layer("avg_pool").output)
        model.save(weights_path)
    else:
        model = load_model(weights_path, compile=False)
    return model


def preprocess_image(image):
    return preprocess_input(image)
