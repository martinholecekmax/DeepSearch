from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

# Get the long description from the relevant file
with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = fh.read()

VERSION = "0.0.1"
DESCRIPTION = "DeepSearch image search engine"

# Setting up
setup(
    name="deep-search-engine",
    version=VERSION,
    author="Martin Holecek",
    author_email="<martin.holecek.max@gmail.com>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    license="MIT",
    url="https://github.com/martinholecekmax/DeepSearch",
    install_requires=[
        "annoy",
        "keras",
        "Keras-Preprocessing",
        "numpy",
        "pandas",
        "Pillow",
        "tqdm",
        "tensorflow",
        "typing-extensions",
    ],
    keywords=[
        "DeepSearch image search",
        "AI image search",
        "image search engine",
        "deep image search",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
    ],
)
