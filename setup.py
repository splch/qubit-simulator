from setuptools import setup, find_packages

setup(
    name="qubit-simulator",
    version="0.1.1",
    description="A simple quantum circuit simulator.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Spencer Churchill",
    author_email="churchill@ionq.com",
    packages=find_packages(),
    install_requires=["numpy"],
    extras_require={"visualization": ["matplotlib"]},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
)
