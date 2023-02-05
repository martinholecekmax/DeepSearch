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

        verbose: bool, optional, default=False
            Print the progress of the model
        metric: str, optional, default=angular
            Metric to use for the index
        n_trees: int, optional, default=100
            Number of trees to use for the index
        model_name: str, optional, default=VGG16
            Model name to use for the feature extraction
        """
        self.verbose = verbose
        self.set_metric(metric)
        self.set_n_trees(n_trees)
        self.model = ModelLoader(model_name)

    @staticmethod
    def get_available_models() -> list:
        """
        Get the available models

        Returns:
            list: List of available models
        """
        return list(models.keys())

    @staticmethod
    def get_available_metrics() -> list:
        """
        Get the available metrics

        Returns:
            list: List of available metrics
        """
        return metrics

    def set_metric(self, metric: str):
        """Set the metric to use for the index"""
        if metric not in metrics:
            raise Exception(f"Metric must be one of {metrics}")
        self.metric = metric

    def set_n_trees(self, n_trees: int):
        """
        Set the number of trees to use for the index

        Args:
            n_trees (int): Number of trees
        """
        self.n_trees = n_trees

    def set_paths(self, db_path: str):
        """
        Set the paths for the representations and annoy index

        Args:
            db_path (str): Path to the database
        """
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

    def load_images(self, db_path: str) -> list:
        """
        Load the images from the database path

        Args:
            db_path (str): Path to the database

        Returns:
            list: List of images
        """
        images = []
        for file in os.listdir(db_path):
            if file.endswith(".jpg") or file.endswith(".png"):
                image_path = os.path.join(db_path, file)
                image_path = image_path.replace("\\", "/")
                images.append(image_path)
        return images

    def extract(self, image: str) -> np.array:
        """
        Extract the features from the image

        Args:
            image (str): Path to the image

        Returns:
            np.array: Array of features
        """
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

    def get_features(self, images: list) -> list:
        """
        Get the features from the images

        Args:
            images (list): List of images

        Returns:
            list: List of features
        """
        features = []
        for image in tqdm(images):
            try:
                features.append(self.extract(image))
            except Exception as e:
                print(e)
                print(f"Error extracting features from {image}")
                continue
        return features

    def start_feature_extraction(self, images: list, representations_path: str) -> pd.DataFrame:
        """
        Start the feature extraction

        Args:
            images (list): List of images
            representations_path (str): Path to save the representations

        Returns:
            pd.DataFrame: Pandas Dataframe with the images and features
        """
        images_data = pd.DataFrame()
        images_data["images_path"] = images
        images_data["features"] = self.get_features(images)
        images_data = images_data.dropna().reset_index(drop=True)
        images_data.to_pickle(representations_path)
        print(f"Features extracted and saved to {representations_path}")
        return images_data

    def start_indexing(self, images_data: pd.DataFrame, annoy_index_path: str):
        """
        Start the indexing process and save the annoy index

        Args:
            images_data (pd.DataFrame): Pandas Dataframe with the images and features
            annoy_index_path (str): Path to save the annoy index
        """
        dim = len(images_data["features"][0])
        annoy_index = AnnoyIndex(dim, self.metric)
        for i, feature in tqdm(zip(images_data.index, images_data["features"])):
            annoy_index.add_item(i, feature)
        annoy_index.build(self.n_trees)
        annoy_index.save(annoy_index_path)
        print(f"Annoy index built and saved to {annoy_index_path}")

    def rebuild(self, db_path: str):
        """
        Rebuild the index and representations from the database path

        Args:
            db_path (str): Path to the database
        """
        # Load images
        images = self.load_images(db_path)

        # Set paths
        self.set_paths(db_path)

        # Extract features
        images_data = self.start_feature_extraction(images, self.representations_path)

        # Build Annoy index
        self.start_indexing(images_data, self.annoy_index_path)

    def build(
        self, db_path: str, metric: str = None, n_trees: int = None, model_name: str = None
    ) -> bool:
        """
        Build the index and representations from the database path and save them to the database path.

        Args:
            db_path (str, required):
                Path to the database
            metric (str, optional):
                Metric to use for the index (euclidean, manhattan, angular, etc.). Defaults to None.
            n_trees (int, optional):
                Number of trees to use for the index. Defaults to None.
            model_name (str, optional):
                Name of the model to use for the feature extraction. Defaults to None.

        Note:
            If metric, n_trees or model_name are not provided,
            the default values will be used, which were set
            when the class was instantiated.

        Returns:
            bool: True if the index and representations were built successfully, False otherwise
        """

        # Set Metric if different and not null
        if metric and metric != self.metric:
            self.set_metric(metric)

        # Set n_trees if different and not null
        if n_trees and n_trees != self.n_trees:
            self.set_n_trees(n_trees)

        # Set model_name if different and not null
        if model_name and model_name != self.model.get_model_name():
            self.model = ModelLoader(model_name)

        if os.path.exists(db_path):
            # Load images
            images = self.load_images(db_path)

            # No images found
            if len(images) == 0:
                print("No images found in database")
                return False

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
                        # Remove image from representations pandas DataFrame
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

    def get_similar_images(
        self, image_path: str, num_results: int = 10, with_distance: bool = True
    ) -> dict:
        """
        Get similar images to the query image

        Args:
            image_path (str):
                Path to the query image
            num_results (int, optional):
                Number of similar images to return. Defaults to 10.
            with_distance (bool, optional):
                Whether to return the calculated distances. Defaults to True.

        Returns:
            dict: Dictionary with the similar images and distances. Lower distance means more similar.
                {
                    "index": [0, 1, 2, ...] - index of the image in the database,
                    "distance": [0.0, 0.1, 0.2, ...] - distance to the query image (depends on the metric),
                    "image_path": ["path/to/image1", "path/to/image2", ...] - path to the image in the database
                }
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image {image_path} not found")

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
