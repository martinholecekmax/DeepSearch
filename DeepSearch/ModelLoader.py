import keras.utils as image_utils
from keras.models import Model
from keras.layers import Input
from keras.applications.vgg16 import VGG16, preprocess_input as vgg_preprocess_input
from keras.applications.inception_v3 import (
    InceptionV3,
    preprocess_input as inception_preprocess_input,
)
from keras.applications.resnet import ResNet50, preprocess_input as resnet_preprocess_input
from keras.applications.xception import Xception, preprocess_input as xception_preprocess_input

models = {
    "VGG16": {
        "preprocess_input": vgg_preprocess_input,
        "base_model": VGG16(weights="imagenet"),
        "layer": "fc1",
    },
    "ResNet50": {
        "preprocess_input": resnet_preprocess_input,
        "base_model": ResNet50(weights="imagenet"),
        "layer": "avg_pool",
    },
    "InceptionV3": {
        "preprocess_input": inception_preprocess_input,
        "base_model": InceptionV3(weights="imagenet", input_tensor=Input(shape=(224, 224, 3))),
        "layer": "avg_pool",
    },
    "Xception": {
        "preprocess_input": xception_preprocess_input,
        "base_model": Xception(weights="imagenet", input_tensor=Input(shape=(224, 224, 3))),
        "layer": "avg_pool",
    },
}


class ModelLoader:
    def __init__(self, model_name):
        self.model_name = model_name
        self.load_model(model_name)

    def load_model(self, model_name=None):
        # Model name is not in models dictionary
        if model_name not in models:
            raise Exception(f"Model must be one of {models}")

        base_model = models[model_name]["base_model"]
        layer = models[model_name]["layer"]
        self.model = Model(inputs=base_model.input, outputs=base_model.get_layer(layer).output)

    def get_model_name(self):
        return self.model_name

    def get_model(self):
        return self.model

    def predict(self, x, verbose=False):
        return self.model.predict(x, verbose)[0]

    def preprocess_input(self, image):
        return models[self.model_name]["preprocess_input"](image)
