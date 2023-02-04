from DeepSearch import DeepSearch


models = DeepSearch().get_available_models()
print(models)

metrics = DeepSearch().get_available_metrics()
print(metrics)

# Create a DeepSearch object
deepSearch = DeepSearch(model_name="InceptionV3", metric="angular", n_trees=10)

# Start processing the images
deepSearch.build("dataset")

# Search for similar images
similar = deepSearch.get_similar_images("lookup/test.jpg", num_results=10, with_distance=True)

# Print the results
print(similar)
