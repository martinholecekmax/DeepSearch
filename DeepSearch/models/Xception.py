import os

from keras.models import load_model
from keras.models import Model
from keras.applications.xception import Xception, preprocess_input
from keras.layers import Input


def get_model():
    if not os.path.exists("DeepSearch/weights/Xception.h5"):
        print("Downloading Xception model...")
        base_model = Xception(weights="imagenet", input_tensor=Input(shape=(224, 224, 3)))
        model = Model(inputs=base_model.input, outputs=base_model.get_layer("avg_pool").output)
        model.save("DeepSearch/weights/Xception.h5")
    else:
        model = load_model("DeepSearch/weights/Xception.h5", compile=False)
    return model


def preprocess_image(image):
    return preprocess_input(image)
