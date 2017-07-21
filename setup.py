# -*- coding: utf-8 -*-
# setup.py
# author : Antoine Passemiers

from setuptools import setup, Extension, find_packages
import os, platform, site
from shutil import copyfile

if platform.system() == "Windows":
    LIB_NAME = "scythe.lib"
elif platform.system() == "Linux":
    LIB_NAME = "scythe.so"
else:
    raise OSError("Your OS is not supported")

site_packages = site.getsitepackages()[-1]

REAL_PATH = os.path.realpath(__file__)
BASE_PATH = os.path.abspath(os.path.join(REAL_PATH, os.pardir))

copyfile(
    os.path.join(BASE_PATH, "src/scythe.lib"), 
    os.path.join(site_packages, "scythe.lib"))

setup(
    name = "scythe",
    version = "1.0",
    packages = ["scythe"],
    description = "Deep learning library based on random forests",
    long_description = open("README.md").read(),
    author = "Antoine Passemiers",
    author_email = "apassemi@ulb.ac.be",
    install_requires = ["numpy"],
    include_package_data = True,
    url = "https://github.com/AntoinePassemiers/Scythe",
    classifiers = [
        "Programming Language :: Python :: 2.7",
        "Development Status :: 1 - Planning",
        "License :: OSI Approved",
        "Natural Language :: French",
        "Operating System :: Microsoft :: Windows",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
   ]
)