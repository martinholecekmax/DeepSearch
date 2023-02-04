import click
import shutil
from DeepSearch import DeepSearch
import os


@click.command()
@click.option(
    "--folder",
    prompt="Database path of images",
    help="The folder that contains the images to be searched.",
)
@click.option("--image", prompt="Query image", help="The image that you want to search for.")
@click.option(
    "--output",
    prompt="Output path",
    help="The folder that will contain the results.",
    default="output",
)
@click.option(
    "--model",
    prompt="Model name",
    help="The model to use for the feature extraction.",
    default="VGG16",
)
@click.option(
    "--metric",
    prompt="Metric",
    help="The metric to use for the index.",
    default="angular",
)
@click.option(
    "--n_trees",
    prompt="Number of trees",
    help="The number of trees to use for the index.",
    default=100,
)
@click.option(
    "--num_results",
    prompt="Number of results",
    help="The number of results to return.",
    default=10,
)
@click.option(
    "--verbose",
    prompt="Verbose",
    help="Print the progress.",
    default=False,
)
def main(folder, image, output, model, metric, n_trees, num_results, verbose):
    print(f"Folder: {folder}")

    # Create a DeepSearch object
    deepSearch = DeepSearch(verbose=verbose, model_name=model, metric=metric, n_trees=n_trees)

    # Start processing the images
    # "input/images"
    deepSearch.build(folder)

    # Search for similar images
    # "input/images/1.jpg"
    similar = deepSearch.get_similar_images(image, num_results=num_results)

    # Print the results
    print(similar)

    if not os.path.exists(output):
        os.makedirs(output)

    # Copy the results to the output folder (dictionary)
    counter = 0
    for image in similar:
        image_path = image["image_path"]
        shutil.copy(image_path, f"{output}/{counter}.jpg")
        counter += 1


if __name__ == "__main__":
    main()
