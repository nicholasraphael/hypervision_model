#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='project',
    version='0.0.0',
    description='Hyperspectral image classification prototype,
    author='',
    author_email='',
    # REPLACE WITH YOUR OWN GITHUB PROJECT LINK
    url='https://github.com/nicholasraphael/hypervision_model',
    install_requires=['pytorch-lightning'],
    packages=find_packages(),
)

