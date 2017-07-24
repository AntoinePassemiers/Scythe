# -*- coding: utf-8 -*-

import os, sys
from distutils.extension import Extension

import numpy as np
from numpy.distutils.misc_util import Configuration
from numpy.distutils.core import setup as np_setup


source_folder = "scythe"
sub_packages = []
source_files = [
    (["core.cpp"], "core"),
]

extra_compile_args = [
    "-std=c++14", 
    "-ftree-loop-optimize",
    "-ftree-vectorize",
    "-funroll-loops",
    "-ftree-vectorizer-verbose=1",
    "-g",
    # "-fopenmp",
    # "-fopenmp-simd"
    "-Iinclude",
    "-O3"
]
extra_link_args = [
    # "-fopenmp",
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
        language = "c++",
        sources = sources,
        include_dirs = include_dirs + [os.curdir],
        libraries = libraries,
        extra_compile_args = extra_compile_args,
        extra_link_args = extra_link_args
    )

np_setup(**config.todict())