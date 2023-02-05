from keras.models import Model
import numpy as np

from DeepSearch.models import VGG16, ResNet50, InceptionV3, Xception

"""
Models dictionary from Keras documentation:
- https://keras.io/api/applications/#usage-examples-for-image-classification-models
"""

models = {
    "VGG16": VGG16,
    "ResNet50": ResNet50,
    "InceptionV3": InceptionV3,
    "Xception": Xception,
}


class ModelLoader:
    def __init__(self, model_name: str):
        """Initialize the model loader class with the model name.

        Args:
            model_name (str):
                The model name to load. Must be one of the keys in the models
                dictionary (VGG16, ResNet50, InceptionV3, Xception, etc.)
        """
        # Model name is not in models dictionary
        if model_name not in models:
            raise Exception(f"Model must be one of {models}")

        self.model_name = model_name
        self.model = models[model_name].get_model()

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
        return models[self.model_name].preprocess_image(image)
