# -*- coding: utf-8 -*-
# setup.py
# author : Antoine Passemiers

from setuptools import setup, find_packages


setup(
   name = "scythe",
   version = "1.0",
   description = 'A useful module',
   author = "Antoine Passemiers",
   author_email = 'foomail@foo.com',
   # packages = ["scythe"],
   packages = find_packages(exclude = []),
   install_requires = ["numpy"],
   data_files = [("scythe", ["scythe/scythe.lib"])],
)