#!/usr/bin/env python
from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

setup(
    name="classical_exps",
    version="0.0",
    description="classical experiments to study neurons digital twins",
    author="lucabaroni",
    packages=find_packages(exclude=[]),
    install_requires=[
        "deeplake",
        'torch==1.13.1',
        'torchvision==0.14.1',
        'torchaudio==0.13.1',
        'transformers==4.35.2',
        'deeplake[enterprise]',
        'wandb',
        'moviepy',
        'imageio',
        'tqdm',
        'statsmodels',
        'param==1.5.1',
    ],
    dependency_links=[
        'https://github.com/sinzlab/nnvision.git#egg=nnvision',
        'https://github.com/MathysHub/featurevis.git#egg=featurevis',
        'https://github.com/CSNG-MFF/imagen.git#egg=imagen',
    ],
    url="https://github.com/lucabaroni/classical_exps.git",
)
