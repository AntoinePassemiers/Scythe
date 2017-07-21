# -*- coding: utf-8 -*-
# setup.py
# author : Antoine Passemiers

from setuptools import setup, Extension, find_packages
import distutils.sysconfig
import os, platform


if platform.system() == "Windows":
    LIB_NAME = "scythe.lib"
elif platform.system() == "Linux":
    LIB_NAME = "scythe.so"
else:
    raise OSError("Your OS is not supported")


REAL_PATH = os.path.realpath(__file__)
BASE_PATH = os.path.abspath(os.path.join(REAL_PATH, os.pardir))

LIB_PATH = os.path.join(BASE_PATH, "../src/%s" % LIB_NAME)

os.environ['PATH'] = os.path.dirname(LIB_PATH) + ';' + os.environ['PATH']

setup(
    name = "scythe",
    version = "1.0",
    packages = ["scythe"],
    description = "Deep learning library based on random forests",
    long_description = open("../README.md").read(),
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
   ],
   data_files = [(distutils.sysconfig.get_python_lib(), [LIB_PATH])],
)