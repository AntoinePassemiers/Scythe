# -*- coding: utf-8 -*-
# setup.py
# author : Antoine Passemiers

import os, sys
from distutils.extension import Extension

import numpy as np
from numpy.distutils.misc_util import Configuration
from numpy.distutils.core import setup as np_setup


source_folder = ""
sub_packages = [
    "python"
]
source_files = [
    (["python/core.py"], "core"),
    (["python/structures.py"], "structures"),
    (["python/example.py"], "example.py")
]

libraries = ["m"] if os.name == "posix" else list()
include_dirs = [np.get_include()]

config = Configuration(source_folder, "", "")
for sub_package in sub_packages:
    config.add_subpackage(sub_package)
for sources, extension_name in source_files:
    sources = [os.path.join(source_folder, source) for source in sources]
    print(extension_name, sources)
    config.add_extension(
        extension_name, 
        sources = sources,
        libraries = libraries
    )

np_setup(**config.todict())