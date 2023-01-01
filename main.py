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
)
def main(folder, image, output):
    print(f"Folder: {folder}")

    # Create a DeepSearch object
    deepSearch = DeepSearch(verbose=True)

    # Start processing the images
    # "input/images"
    deepSearch.build(folder)

    # Search for similar images
    # "input/images/1.jpg"
    similar = deepSearch.get_similar_images(image, num_results=50)

    # Print the results
    print(similar)

    if not os.path.exists(output):
        os.makedirs(output)

    # Copy the results to the output folder (dictionary)
    counter = 0
    for key, image in similar.items():
        shutil.copy(image, f"{output}/{counter}.jpg")
        counter += 1


if __name__ == "__main__":
    main()
