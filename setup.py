import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mll",
    version="2.1.0",
    author="Bernardi Riccardo",
    author_email="riccardo.bernardi@rocketmail.com",
    description="A machine learning language that have a specific grammar made to simplify large and complex machine learning and deep learning tasks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/riccardobernardi/MLL-machine-learning-language",
    packages=setuptools.find_packages()+['mll'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'lark-parser',
        'numpy',
        'tensorflow',
        'keras',
        'termcolor',
        'sklearn',
        'mlxtend'
    ],
)