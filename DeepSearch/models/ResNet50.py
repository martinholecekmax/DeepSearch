import os

from keras.models import load_model
from keras.models import Model
from keras.applications.resnet import ResNet50, preprocess_input


def get_model():
    package_directory = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(package_directory, "weights/ResNet50.h5")
    if not os.path.exists(weights_path):
        print("Downloading ResNet50 model...")
        base_model = ResNet50(weights="imagenet")
        model = Model(inputs=base_model.input, outputs=base_model.get_layer("avg_pool").output)
        model.save(weights_path)
    else:
        model = load_model(weights_path, compile=False)
    return model


def preprocess_image(image):
    return preprocess_input(image)
