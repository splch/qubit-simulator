from setuptools import setup, find_packages
from os import path

# Get the directory where this script is located
here = path.abspath(path.dirname(__file__))

# Read the requirements from the requirements.txt file
with open(path.join(here, "requirements.txt")) as f:
    requirements = f.readlines()

setup(
    name="qubit_simulator",
    version="0.0.6",
    description="A simple qubit simulator",
    long_description=open(path.join(here, "README.md")).read(),
    long_description_content_type="text/markdown",
    author="Spencer Churchill",
    author_email="churchill@ionq.com",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
)
