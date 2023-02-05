from keras.models import load_model
from keras.models import Model
from keras.applications.vgg16 import VGG16, preprocess_input
import os


def get_model():
    package_directory = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(package_directory, "weights/VGG16.h5")
    if not os.path.exists(weights_path):
        print("Downloading VGG16 model...")
        base_model = VGG16(weights="imagenet", include_top=True)
        model = Model(inputs=base_model.input, outputs=base_model.get_layer("fc1").output)
        model.save(weights_path)
    else:
        model = load_model(weights_path, compile=False)
    return model


def preprocess_image(image):
    return preprocess_input(image)
