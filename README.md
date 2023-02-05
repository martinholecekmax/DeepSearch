# DeepSearch

<p align="center"><img src="https://raw.githubusercontent.com/martinholecekmax/DeepSearch/main/assets/logo.png" width="200" height="200"></p>

DeepSearch is a sophisticated AI-powered search engine designed to enhance image searching. It utilizes deep learning algorithms to efficiently search a vast collection of images and find the most similar matches.

The DeepSearch engine is built on top of the [Annoy](https://github.com/spotify/annoy) library, which is a fast, memory-efficient, and easy-to-use library for approximate nearest neighbor search.

The engine uses a pre-trained models from [Keras](https://keras.io/api/applications/) to extract features from images and then stores them in an Annoy index. The index is then used to find the most similar images to a given query image.

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation from PyPI](#installation-from-pypi)
- [Installation from GitHub Repository](#installation-from-github-repository)
- [Usage](#usage)
  - [Importing the DeepSearch class](#importing-the-deepsearch-class)
  - [Initializing the DeepSearch class](#initializing-the-deepsearch-class)
  - [Building the index](#building-the-index)
  - [Saving the index](#saving-the-index)
  - [Searching for similar images](#searching-for-similar-images)
- [Full Implementation Example](#full-implementation-example)
- [CLI Usage](#cli-usage)
- [Supported Models](#supported-models)
- [Supported Metrics](#supported-metrics)
- [Impact of Image Quantity on Processing Time](#impact-of-image-quantity-on-processing-time)
- [Contributing](#contributing)

## Features

- **Fast**: DeepSearch is built on top of the Annoy library, which is a fast, memory-efficient, and easy-to-use library for approximate nearest neighbor search.
- **Easy to use**: DeepSearch is designed to be easy to use and integrate into your existing applications.
- **High Accuracy**: DeepSearch uses a pre-trained model from Keras to extract features from images and then stores them in an Annoy index. The index is then used to find the most similar images to a given query image.

## Prerequisites

Python 3.10.6+ is required to install DeepSearch. You can download the latest version of Python from [here](https://www.python.org/downloads/).

You also need to install TensorFlow at least 2.10.1 which can be downloaded from [here](https://www.tensorflow.org/install).

## Installation from PyPI

You can install DeepSearch from PyPI package repository found [here](https://pypi.org/project/deep-search-engine/)

To install DeepSearch, run the following command:

```bash
pip install deep-search-engine
```

## Installation from GitHub Repository

You can also install DeepSearch from the GitHub repository found [here](https://github.com/martinholecekmax/DeepSearch) by cloning the repository and installing the requirements.

**Important Note**: In order to use DeepSearch CLI you need to install it from the GitHub repository.

It is recommended to install DeepSearch in a virtual environment. You can use [virtualenv](https://virtualenv.pypa.io/en/latest/) or [venv](https://docs.python.org/3/library/venv.html) to create a virtual environment.

To create a virtual environment using `venv`, run the following command:

```bash
python -m venv env
```

To activate the virtual environment, run the following command:

```bash
# Windows
source env/Scripts/activate

# Linux
source env/bin/activate
```

You will need to install the requirements before you can use DeepSearch. To install the requirements, run the following command:

```bash
pip install -r requirements.txt
```

With everything installed, you can now start utilizing DeepSearch.

## Usage

There are two options for using DeepSearch in your application. You can either use the DeepSearch class and its methods in your code or you can use the DeepSearch CLI.

### Importing the DeepSearch class

First, you need to import the DeepSearch class from the DeepSearch module as follows:

```python
from DeepSearch import DeepSearch
```

### Initializing the DeepSearch class

Then, you need to create an instance of the DeepSearch class. You can optionally pass the model name, the number of trees to the constructor, metric, and verbose parameters. The default values are as follows:

```python
deepSearch = DeepSearch(model_name='VGG16', n_trees=100, metric='angular', verbose=True)
```

The `model_name` parameter specifies the name of the model to use for extracting features from images. More information about the supported models can be found [here](#supported-models).

The `n_trees` parameter specifies the number of trees to use in the Annoy index. The default value is `100`. More trees will give you better accuracy but will also increase the memory usage and search time.

The `metric` parameter specifies the distance metric to use in the Annoy index. More information about the supported metrics can be found [here](#supported-metrics).

The `verbose` parameter specifies whether to print the progress of the indexing process. The default value is `False`.

### Building the index

Now, you can build the index and representations by calling the `build()` method. This method requires the path to the `dataset` directory which contains the images to index as a string.

```python
deepSearch.build('dataset')
```

This function will go through all the images in the `dataset` directory and extract features from them. It will use those features to build the Annoy index and store the indexes and representations in the same directory.

You can optionally pass `metric`, `n_trees` and `model_name` parameters to the `build()` method. The default values are the same as the ones you passed to the constructor.

This can be useful if you want to try different values for the parameters without creating a new instance of the DeepSearch class.

### Saving the index

The `build()` method will save the index and representations in the same directory as the images. If you use different values for the parameters, the `build()` method will save the index and representations as a separate file.

For example, if you use the `VGG16` model with the `angular` metric and `100` trees, the index and representations will be saved in the `VGG16_angular_100_annoy_index.ann` and `VGG16_angular_100_representations.pkl` files respectively.

The saving format is as follows:

```python
# Annoy index file
f'{model_name}_{metric}_{n_trees}_annoy_index.ann'

# Representations file
f'{model_name}_{metric}_{n_trees}_representations.pkl'
```

The `pickle` module is used to save the representations.

### Searching for similar images

Finally, you can search for similar images by calling the `get_similar_images()` method. This method will extract features from the query image and then use them to find the most similar images in the index. You have to specify the path to the query image as a string.

You can optionally pass the number of similar images to return as an integer. The default value is 10. You can specify the optional parameter `with_distances` as True to return the distances of the similar images as well. The default value of this parameter is `False`.

```python
similar_images = deepSearch.get_similar_images('query.jpg', num_results=20, with_distance=True)
print(similar_images)
```

The output of the `get_similar_images()` method is a python list of dictionaries. Each dictionary contains the image index from the index file, the path to the similar image and the distance between the query image and the similar image. The list is sorted by the distance in ascending order (the first image is the most similar).

```python
[
    {
        'index': 0,
        'path': 'images/0.jpg',
        'distance': 0.0
    },
    {
        'index': 1,
        'path': 'images/1.jpg',
        'distance': 0.6206140518188477
    },
    {
        'index': 2,
        'path': 'images/2.jpg',
        'distance': 0.7063581943511963
    },
    ...
]
```

## Full Implementation Example

The following example shows how to use DeepSearch in your code. It will index all the images in the `dataset` directory and then find the most similar images to the query image.

```python
from DeepSearch import DeepSearch

# Initialize the DeepSearch class
deepSearch = DeepSearch(model_name='VGG16', n_trees=100, metric='angular', verbose=True)

# Build the index and representations
deepSearch.build('dataset')

# Search for similar images
similar_images = deepSearch.get_similar_images('lookup/query.jpg', num_results=20, with_distance=True)

# Print the similar images
print(similar_images)
```

The full implementation of the example can be found in the DeepSearchDemo.py file.

To run the demo, you need to copy the images you want to index to the dataset directory, copy the query image to the lookup directory, and then run the DeepSearchDemo.py file as follows:

```bash
python DeepSearchDemo.py
```

## CLI Usage

In order to use DeepSearch from the command line, you need to install the DeepSearch CLI from GitHub explained in the [Installation from GitHub Repository](#installation-from-github-repository) section.

The another option for using DeepSearch is to use the DeepSearch CLI. The DeepSearch CLI allows you to use DeepSearch from the command line without writing any code.

Running the DeepSearch CLI will build the index and search for similar images. The similar images will then be saved in a directory which can be specified using the `--output` option or will be saved in the `output` by default. The output directory will be created if it doesn't exist.

There are several options you can pass to the DeepSearch CLI. The options are as follows:

- `--folder`: The path to the folder containing the images to index. This option is required.
- `--output`: The path to the output directory where the similar images will be saved. The default value is `output`.
- `--image`: The path to the query image. This option is required.
- `--num-results`: The number of similar images to return. The default value is 10.
- `--metric`: The distance metric to use in the Annoy index. The default value is `angular`.
- `--n-trees`: The number of trees to use in the Annoy index. The default value is 100.
- `--model`: The name of the model to use for extracting features from images. The default value is `VGG16`.
- `--verbose`: Whether to print the progress of the indexing process. The default value is `False`.

To run the DeepSearch CLI, you need to run the DeepSearchCLI.py file as follows:

```bash
# Example with required options only
python DeepSearchCLI.py --folder dataset --image lookup/query.jpg

# Example with several options
python DeepSearchCLI.py --folder dataset --image lookup/query.jpg --output output --num_results 20 --metric euclidean --n_trees 20 --model ResNet50 --verbose True
```

## Supported Models

The following models are supported:

- **[VGG16](https://keras.io/api/applications/vgg/#vgg16-function)** (default)
- **[ResNet50](https://keras.io/api/applications/resnet/#resnet50-function)**
- **[InceptionV3](https://keras.io/api/applications/inceptionv3/)**
- **[Xception](https://keras.io/api/applications/xception/)**

The default value is `VGG16`. You can get a list of available models by calling the static `get_available_models()` method of the DeepSearch class as follows:

```python
# Get a list of available models
models = DeepSearch.get_available_models()
print(models) # ['VGG16', 'ResNet50', 'InceptionV3', 'Xception']
```

The models are case sensitive and must be specified exactly as shown above.

You can easily add support for other models from the [Keras](https://keras.io/api/applications/) Applications library by adding a new model class to the `models` dictionary in the `ModelLoader` class.

## Supported Metrics

The following metrics are supported:

- **[angular](https://en.wikipedia.org/wiki/Cosine_similarity)** (default) - The cosine similarity metric.
- **[euclidean](https://en.wikipedia.org/wiki/Euclidean_distance)** - The Euclidean distance metric.
- **[manhattan](https://en.wikipedia.org/wiki/Taxicab_geometry)** - The Manhattan distance metric.
- **[hamming](https://en.wikipedia.org/wiki/Hamming_distance)** - The Hamming distance metric.
- **[dot](https://en.wikipedia.org/wiki/Dot_product)** - The dot product metric.

The default value is `angular` which is the cosine distance.

You can get a list of available metrics by calling the static `get_available_metrics()` method of the DeepSearch class as follows:

```python
# Get a list of available metrics
metrics = DeepSearch.get_available_metrics()
print(metrics) # ['angular', 'euclidean', 'manhattan', 'hamming', 'dot']
```

The metrics are case sensitive and must be specified exactly as shown above.

## Impact of Image Quantity on Processing Time

When processing a large number of images, it may take longer for the algorithm to generate representations. This is due to the increased computational demands of processing more data (more memory and CPU usage).

Once the representations are generated, the search process is very fast. The search process is limited by the number of trees in the Annoy index. The more trees you use, the more accurate the search results will be, but the longer it will take to search.

When you run the algorithm for the first time, it will generate the representations and save them to a file. The next time you run the algorithm, it will load the representations from the file instead of generating them again. This will significantly reduce the processing time.

For example, I have run the algorithm on a dataset of 100,000 images and the generation of the representations took approximately 12 minutes. Each subsequent run took couple of seconds which depends on the size of the dataset and the number of trees in the Annoy index.

One of the great features is that you can add more images to the dataset and run the algorithm again. The algorithm will only generate representations for the new images and will load the representations for the existing images from the file. This will significantly reduce the processing time.

When the image is deleted from the dataset, the algorithm will remove the representation for the image from the file. This will avoid any issues when searching for similar images.

Any of this operations will force the algorithm to remove the annoy index file and generate it again. This will ensure that the annoy index file is up to date. However, this is relatively fast operation depending on the size of the dataset and the number of trees in the Annoy index. For the previous example of 100,000 images, the generation of the annoy index file took approximately 3 seconds.

You can force the algorithm to remove the representations file and annoy index file by passing the `--clear` option to the DeepSearch CLI as follows:

```bash
python DeepSearchCLI.py --folder dataset --image lookup/query.jpg --clear True
```

Or you can call the `rebuild()` method of the DeepSearch class if you are using the DeepSearch API as follows:

```python
# Rebuild the index
deep_search.rebuild()
```

## Contributing

If you would like to contribute to this project, please feel free to submit a pull request. If you have any questions, please feel free to open an issue.

