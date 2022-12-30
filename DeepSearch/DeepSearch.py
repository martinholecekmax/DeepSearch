import tensorflow as tf
from tensorflow import keras
import keras.utils as image_utils
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
from annoy import AnnoyIndex
from tqdm import tqdm

import os
import pandas as pd
import numpy as np
from PIL import Image

metrics = ["angular", "euclidean", "manhattan", "hamming", "dot"]


class DeepSearch:
    def __init__(self, verbose=False, metric="angular", n_trees=100):
        base_model = VGG16(weights="imagenet")
        self.model = Model(inputs=base_model.input, outputs=base_model.get_layer("fc1").output)
        self.verbose = verbose
        self.n_trees = n_trees

        if metric not in metrics:
            raise Exception(f"Metric must be one of {metrics}")

        self.metric = metric

    def load_images(self, db_path):
        images = []
        for file in os.listdir(db_path):
            if file.endswith(".jpg") or file.endswith(".png"):
                image_path = os.path.join(db_path, file)
                image_path = image_path.replace("\\", "/")
                images.append(image_path)
        return images

    def extract(self, image):
        # Load the image
        image = Image.open(image)
        # Resize the image
        image = image.resize((224, 224))
        # Convert the image color space
        image = image.convert("RGB")
        # Reformat the image
        x = image_utils.img_to_array(image)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        # Extract Features
        feature = self.model.predict(x, verbose=self.verbose)[0]
        return feature / np.linalg.norm(feature)

    def get_features(self, images):
        features = []
        for image in tqdm(images):
            try:
                features.append(self.extract(image))
            except Exception as e:
                print(e)
                print(f"Error extracting features from {image}")
                continue
        return features

    def start_feature_extraction(self, images, representations_path="representations.pkl"):
        images_data = pd.DataFrame()
        images_data["images_path"] = images
        images_data["features"] = self.get_features(images)
        images_data = images_data.dropna().reset_index(drop=True)
        images_data.to_pickle(representations_path)
        print(f"Features extracted and saved to {representations_path}")
        return images_data

    def start_indexing(self, images_data, annoy_index_path="annoy_index.ann"):
        dim = len(images_data["features"][0])
        annoy_index = AnnoyIndex(dim, self.metric)
        for i, feature in tqdm(zip(images_data.index, images_data["features"])):
            annoy_index.add_item(i, feature)
        annoy_index.build(self.n_trees)
        annoy_index.save(annoy_index_path)
        print(f"Annoy index built and saved to {annoy_index_path}")

    def Start(self, db_path):
        if os.path.exists(db_path):
            # Load images
            images = self.load_images(db_path)

            representations_path = os.path.join(db_path, "representations.pkl")
            representations_path = representations_path.replace("\\", "/")
            annoy_index_path = os.path.join(db_path, "annoy_index.ann")
            annoy_index_path = annoy_index_path.replace("\\", "/")
            self.representations_path = representations_path
            self.annoy_index_path = annoy_index_path

            # Check if representations and annoy index already exist
            if os.path.exists(representations_path) and os.path.exists(annoy_index_path):
                # Ask if user wants to use existing representations and annoy index
                print("Found existing representations and annoy index")
                print("Do you want to use them? (y/n)")
                flag = str(input())
                if flag == "y" or flag == "Y":
                    print("Using existing representations and annoy index")
                    return True

            print("Extracting features")
            image_data = self.start_feature_extraction(images, representations_path)
            self.start_indexing(image_data, annoy_index_path)
            print("Done. Please apply search now.")
            return True
        else:
            print("Please Enter the Valid Folder Path")
            return False

    def get_similar_images(self, image_path, num_results=10, with_distance=False):
        query_vector = self.extract(image_path)
        annoy_index_path = self.annoy_index_path
        representations_path = self.representations_path
        images_data = pd.read_pickle(representations_path)
        dim = len(images_data["features"][0])
        annoy_index = AnnoyIndex(dim, self.metric)
        annoy_index.load(annoy_index_path)
        if with_distance:
            similar_images, distances = annoy_index.get_nns_by_vector(
                query_vector, num_results, include_distances=True
            )
            return list(
                zip(
                    similar_images,
                    distances,
                    images_data.iloc[similar_images]["images_path"].to_list(),
                )
            )
            # return dict(zip(similar_images, images_data.iloc[similar_images]["images_path"].to_list())), distances
        else:
            similar_images = annoy_index.get_nns_by_vector(query_vector, num_results)
            return dict(
                zip(similar_images, images_data.iloc[similar_images]["images_path"].to_list())
            )
