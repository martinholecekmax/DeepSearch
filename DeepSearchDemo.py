from DeepSearch import DeepSearch

# Create a DeepSearch object
deepSearch = DeepSearch()

# Start processing the images
deepSearch.Start("input/test")

# Search for similar images
similar = deepSearch.get_similar_images("lookup/test.jpg", num_results=2, with_distance=True)

# Print the results
print(similar)
