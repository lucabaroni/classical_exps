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
    ],
    url="https://github.com/lucabaroni/classical_exps.git",
)