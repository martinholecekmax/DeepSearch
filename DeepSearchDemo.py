from DeepSearch import DeepSearch


models = DeepSearch().get_available_models()
output = []

for model_name in models:
    deepSearch = DeepSearch(model_name=model_name, metric="euclidean", n_trees=10)
    deepSearch.build("input/test")
    similar = deepSearch.get_similar_images("lookup/test.jpg", num_results=2, with_distance=True)
    output.append({"model": model_name, "results": similar})

for i in output:
    print(i["model"])
    print(i["results"])

# # Create a DeepSearch object
# deepSearch = DeepSearch(model_name="InceptionV3", metric="angular", n_trees=10)

# # Start processing the images
# deepSearch.build("input/test")

# # Search for similar images
# similar = deepSearch.get_similar_images("lookup/test.jpg", num_results=2, with_distance=True)

# # Print the results
# print(similar)

# # for i in similar:
# #     print(i["image_path"])

# # Rebuild the index
# deepSearch.build("input/test", metric="euclidean", n_trees=10, model_name="Xception")

# # Search for similar images
# similar = deepSearch.get_similar_images("lookup/test.jpg", num_results=2, with_distance=True)

# # Print the results
# print(similar)
