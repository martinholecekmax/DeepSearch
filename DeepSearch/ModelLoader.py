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


models = ["VGG16", "InceptionV3", "ResNet50", "Xception"]


class ModelLoader:
    def __init__(self, model_name):
        self.model_name = model_name
        self.load_model(model_name)

    def load_model(self, model_name=None):
        if model_name not in models:
            raise Exception(f"Model must be one of {models}")

        if model_name == "VGG16":
            self.preprocess_input = vgg_preprocess_input
            base_model = VGG16(weights="imagenet")
            self.model = Model(inputs=base_model.input, outputs=base_model.get_layer("fc1").output)
        elif model_name == "ResNet50":
            self.preprocess_input = resnet_preprocess_input
            base_model = ResNet50(weights="imagenet")
            self.model = Model(
                inputs=base_model.input, outputs=base_model.get_layer("avg_pool").output
            )
        elif model_name == "InceptionV3":
            self.preprocess_input = inception_preprocess_input
            input_tensor = Input(shape=(224, 224, 3))
            base_model = InceptionV3(weights="imagenet", input_tensor=input_tensor)
            self.model = Model(
                inputs=base_model.input, outputs=base_model.get_layer("avg_pool").output
            )
        elif model_name == "Xception":
            self.preprocess_input = xception_preprocess_input
            input_tensor = Input(shape=(224, 224, 3))
            base_model = Xception(weights="imagenet", input_tensor=input_tensor)
            self.model = Model(
                inputs=base_model.input, outputs=base_model.get_layer("avg_pool").output
            )

    def get_model(self):
        return self.model

    def get_preprocess_input(self):
        return self.preprocess_input

    def preprocess(self, image):
        return self.preprocess_input(image)

    def get_model_name(self):
        return self.model_name
