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
        'torch',
        'torchvision',
        'torchaudio',
        'deeplake[enterprise]',
        'wandb',
        'moviepy',
        'imageio',
        'tqdm',
        'statsmodels',
        'param==1.5.1',
        # 'featurevis @ git+https://github.com/lucabaroni/featurevis_mod.git#egg=featurevis',
        'imagen @ git+https://github.com/CSNG-MFF/imagen.git#egg=imagen',
    ],
    url="https://github.com/lucabaroni/classical_exps.git",
)
