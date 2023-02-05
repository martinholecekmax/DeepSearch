from keras.models import load_model
from keras.models import Model
from keras.applications.vgg16 import VGG16, preprocess_input
import os


def get_model():
    if not os.path.exists("DeepSearch/weights/VGG16.h5"):
        print("Downloading VGG16 model...")
        base_model = VGG16(weights="imagenet", include_top=True)
        model = Model(inputs=base_model.input, outputs=base_model.get_layer("fc1").output)
        model.save("DeepSearch/weights/VGG16.h5")
    else:
        model = load_model("DeepSearch/weights/VGG16.h5", compile=False)
    return model


def preprocess_image(image):
    return preprocess_input(image)
