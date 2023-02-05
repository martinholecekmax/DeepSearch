import os

from keras.models import load_model
from keras.models import Model
from keras.applications.resnet import ResNet50, preprocess_input


def get_model():
    if not os.path.exists("DeepSearch/weights/ResNet50.h5"):
        print("Downloading ResNet50 model...")
        base_model = ResNet50(weights="imagenet")
        model = Model(inputs=base_model.input, outputs=base_model.get_layer("avg_pool").output)
        model.save("DeepSearch/weights/ResNet50.h5")
    else:
        model = load_model("DeepSearch/weights/ResNet50.h5", compile=False)
    return model


def preprocess_image(image):
    return preprocess_input(image)
