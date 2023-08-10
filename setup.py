from setuptools import setup, find_packages

setup(
    name="qubit_simulator",
    version="0.0.1",
    description="A simple qubit simulator",
    author="Spencer Churchill",
    author_email="churchill@ionq.com",
    packages=find_packages(),
    install_requires=["numpy"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
)
