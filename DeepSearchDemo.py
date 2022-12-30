from DeepSearch import DeepSearch

# Create a DeepSearch object
deepSearch = DeepSearch()

# Start processing the images
deepSearch.Start("input/images_0")

# Search for similar images
similar = deepSearch.get_similar_images("lookup/eli.jpg", num_results=50, with_distance=True)

# Print the results
print(similar)
