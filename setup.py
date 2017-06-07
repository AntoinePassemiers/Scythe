# -*- coding: utf-8 -*-
# setup.py
# author : Antoine Passemiers

from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding = 'utf-8') as f:
    long_description = f.read()

packages = find_packages(exclude = [])
print(packages)

setup(
    name = 'Scythe',
    version = '1.3.0',
    description = 'A gradient boosting library',
    long_description = long_description,
    url = 'https://github.com/AntoinePassemiers/Scythe',
    author = 'Antoine Passemiers',
    author_email = 'apassemi@ulb.ac.be',
    license = 'GPL-3.0',
    classifiers = [
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Machine Learning :: Gradient boosting',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
    keywords = 'machine-learning boosting trees',
    packages = packages,
)