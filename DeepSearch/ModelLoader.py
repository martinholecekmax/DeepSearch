from keras.models import Model
from keras.layers import Input
from keras.applications.vgg16 import VGG16, preprocess_input as vgg_preprocess_input
from keras.applications.inception_v3 import (
    InceptionV3,
    preprocess_input as inception_preprocess_input,
)
from keras.applications.resnet import ResNet50, preprocess_input as resnet_preprocess_input
from keras.applications.xception import Xception, preprocess_input as xception_preprocess_input
import numpy as np

"""
Models dictionary from Keras documentation:
- https://keras.io/api/applications/#usage-examples-for-image-classification-models
"""

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
    def __init__(self, model_name: str):
        """Initialize the model loader class with the model name.

        Args:
            model_name (str):
                The model name to load. Must be one of the keys in the models
                dictionary (VGG16, ResNet50, InceptionV3, Xception, etc.)
        """
        self.model_name = model_name
        self.load_model(model_name)

    def load_model(self, model_name: str = None):
        """Load the model from the models dictionary and set the model attribute.

        Args:
            model_name (str, optional):
                The model name to load. Defaults to None.

        Raises:
            Exception:
                Model must be one of models from models dictionary
                (VGG16, ResNet50, InceptionV3, Xception, etc.)
        """

        # Model name is not in models dictionary
        if model_name not in models:
            raise Exception(f"Model must be one of {models}")

        base_model = models[model_name]["base_model"]
        layer = models[model_name]["layer"]
        self.model = Model(inputs=base_model.input, outputs=base_model.get_layer(layer).output)

    def get_model_name(self) -> str:
        """Get the model name.

        Returns:
            str: The model name
        """
        return self.model_name

    def get_model(self) -> Model:
        """Get the model.

        Returns:
            Model: The keras model
        """
        return self.model

    def predict(self, x: np.ndarray, verbose: bool = False) -> np.ndarray:
        """Predict the output of the model.

        Args:
            x (np.ndarray): Input to the model (preprocessed image stored as a numpy array)
            verbose (bool, optional): Whether to print the progress. Defaults to False.

        Returns:
            np.ndarray: The output (prediction) of the model (the feature vector)
        """
        return self.model.predict(x, verbose)[0]

    def preprocess_input(self, image: np.ndarray) -> np.ndarray:
        """Preprocess the input image.

        Args:
            image (np.ndarray): The image to preprocess (stored as a numpy array)

        Returns:
            np.ndarray: The preprocessed image
        """
        return models[self.model_name]["preprocess_input"](image)
