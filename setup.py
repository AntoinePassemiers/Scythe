# -*- coding: utf-8 -*-
# setup.py
# author : Antoine Passemiers

from setuptools import setup, Extension
import os, platform


if platform.system == "Windows":
    LIB_NAME = "scythe.lib"
elif platform.system == "Linux":
    LIB_NAME = "scythe.so"
else:
    raise OSError("Your OS is not supported")

BASE_PATH = os.path.join(os.path.realpath(__file__), "..")

setup(
   name = "scythe",
   version = "1.0",
   description = "Deep learning library based on random forests",
   author = "Antoine Passemiers",
   author_email = "apassemi@ulb.ac.be",
   install_requires = ["numpy"]
)