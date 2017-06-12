# -*- coding: utf-8 -*-
# setup.py
# author : Antoine Passemiers

from setuptools import setup, Extension


setup(
   name = "scythe",
   version = "1.0",
   description = "Deep learning library based on random forests",
   author = "Antoine Passemiers",
   author_email = "apassemi@ulb.ac.be",
   install_requires = ["numpy"]
)