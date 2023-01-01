import os
from tqdm import tqdm
from annoy import AnnoyIndex
import pandas as pd
import numpy as np
from PIL import Image

import keras.utils as image_utils

from DeepSearch.ModelLoader import ModelLoader, models


metrics = ["angular", "euclidean", "manhattan", "hamming", "dot"]


class DeepSearch:
    """
    DeepSearch is a class that allows you to search for similar images using deep learning models and Annoy Indexes.
    """

    def __init__(self, verbose=False, metric="angular", n_trees=100, model_name="VGG16"):
        """
        Initialize the DeepSearch object
        :param verbose: Print the progress of the model
        :param metric: Metric to use for the index
        :param n_trees: Number of trees to use for the index
        :param model_name: Model to use for the feature extraction
        defaults:
            verbose: False
            metric: angular
            n_trees: 100
            model_name: VGG16
        """
        self.verbose = verbose
        self.set_metric(metric)
        self.set_n_trees(n_trees)
        self.model = ModelLoader(model_name)

    def get_available_models(self):
        return list(models.keys())

    def set_metric(self, metric):
        if metric not in metrics:
            raise Exception(f"Metric must be one of {metrics}")
        self.metric = metric

    def set_n_trees(self, n_trees):
        self.n_trees = n_trees

    def set_paths(self, db_path):
        model_name = self.model.get_model_name()
        representations_path = os.path.join(
            db_path, f"{model_name}_{self.metric}_{self.n_trees}_representations.pkl"
        )
        representations_path = representations_path.replace("\\", "/")
        annoy_index_path = os.path.join(
            db_path, f"{model_name}_{self.metric}_{self.n_trees}_annoy_index.ann"
        )
        annoy_index_path = annoy_index_path.replace("\\", "/")
        self.representations_path = representations_path
        self.annoy_index_path = annoy_index_path

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

        # Preprocess the image
        x = self.model.preprocess_input(x)

        # Extract Features
        feature = self.model.predict(x, verbose=self.verbose)
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

    def start_feature_extraction(self, images, representations_path):
        images_data = pd.DataFrame()
        images_data["images_path"] = images
        images_data["features"] = self.get_features(images)
        images_data = images_data.dropna().reset_index(drop=True)
        images_data.to_pickle(representations_path)
        print(f"Features extracted and saved to {representations_path}")
        return images_data

    def start_indexing(self, images_data, annoy_index_path):
        dim = len(images_data["features"][0])
        annoy_index = AnnoyIndex(dim, self.metric)
        for i, feature in tqdm(zip(images_data.index, images_data["features"])):
            annoy_index.add_item(i, feature)
        annoy_index.build(self.n_trees)
        annoy_index.save(annoy_index_path)
        print(f"Annoy index built and saved to {annoy_index_path}")

    def rebuild(self, db_path):
        # Load images
        images = self.load_images(db_path)

        # Set paths
        self.set_paths(db_path)

        # Extract features
        images_data = self.start_feature_extraction(images, self.representations_path)

        # Build Annoy index
        self.start_indexing(images_data, self.annoy_index_path)

    def build(self, db_path, metric=None, n_trees=None, model_name=None):

        # Set Metric if different and not null
        if metric and metric != self.metric:
            self.set_metric(metric)

        # Set n_trees if different and not null
        if n_trees and n_trees != self.n_trees:
            self.set_n_trees(n_trees)

        # Set model_name if different and not null
        if model_name and model_name != self.model.get_model_name():
            self.model.load_model(model_name)

        if os.path.exists(db_path):
            # Load images
            images = self.load_images(db_path)

            # Set paths
            self.set_paths(db_path)

            image_data = None
            update = False

            if os.path.exists(self.representations_path):
                print("Found existing representations")
                image_data = pd.read_pickle(self.representations_path)

                # Remove images that are no longer in the database
                for image in image_data["images_path"]:
                    if image not in images:
                        print(f"Image {image} removed from database")
                        # Remove image from representations pandas dataframe
                        image_data = image_data[image_data["images_path"] != image]
                        update = True

                # Extract features for new images
                new_images = []
                for image in images:

                    # Check if image is not in representations
                    if image not in image_data["images_path"].values:
                        # If image is not in representations
                        # extract features and add to representations (concatenate)
                        print(f"Image {image} added to database")
                        new_images.append(
                            {
                                "images_path": image,
                                "features": self.extract(image),
                            }
                        )

                if len(new_images) > 0:
                    update = True
                    new_images = pd.DataFrame(new_images)
                    image_data = pd.concat([image_data, new_images], ignore_index=True)

                if update:
                    image_data = image_data.dropna().reset_index(drop=True)

                    # Save updated representations
                    image_data.to_pickle(self.representations_path)
                    print(f"Updated representations saved to {self.representations_path}")
                else:
                    print("No changes detected. No update required.")

            else:
                print("Extracting features")
                image_data = self.start_feature_extraction(images, self.representations_path)
                update = True

            if os.path.exists(self.annoy_index_path) and not update:
                print("Found existing annoy index")
            else:
                print("Building annoy index")
                self.start_indexing(image_data, self.annoy_index_path)

            print("Done. Please apply search now.")
            return True
        else:
            print("Please Enter the Valid Folder Path")
            return False

    def get_similar_images(self, image_path, num_results=10, with_distance=True):
        query_vector = self.extract(image_path)
        annoy_index_path = self.annoy_index_path
        representations_path = self.representations_path
        images_data = pd.read_pickle(representations_path)
        dim = len(images_data["features"][0])
        annoy_index = AnnoyIndex(dim, self.metric)
        annoy_index.load(annoy_index_path)

        similar_images, distances = annoy_index.get_nns_by_vector(
            query_vector, num_results, include_distances=True
        )

        # Similar images, distances, images path in pandas DataFrame
        df = pd.DataFrame(
            {
                "index": similar_images,
                "distance": distances,
                "image_path": images_data.iloc[similar_images]["images_path"].to_list(),
            }
        )

        if not with_distance:
            df = df.drop(columns=["distance"])

        return df.to_dict(orient="records")
