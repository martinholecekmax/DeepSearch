from DeepSearch import DeepSearch

"""
DeepSearchDemo.py: Demo of the DeepSearch library.

Before running this script, you need to add images 
to the dataset folder and a lookup image to the 
lookup folder.
"""

# Get the available models
models = DeepSearch.get_available_models()
print(models)

# Get the available metrics
metrics = DeepSearch.get_available_metrics()
print(metrics)

# Create a DeepSearch object with the InceptionV3 model and the angular metric
deepSearch = DeepSearch(model_name="InceptionV3", metric="angular", n_trees=10)

# Start processing the images (make sure to add images to the dataset folder)
deepSearch.build("dataset")

# Search for similar images (make sure to add a lookup image called test.jpg to the lookup folder)
similar = deepSearch.get_similar_images("lookup/test.jpg", num_results=10, with_distance=True)

# Print the results
print(similar)
