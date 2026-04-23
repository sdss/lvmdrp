# encoding: utf-8
#
# setup.py
#

import os
import sys
from setuptools import Extension, setup
from distutils import sysconfig


FAST_MEDIAN_PATH = 'cextern/fast_median/src'

cpp_flags = []
link_flags = []

if sys.platform == "linux":
    cpp_flags += ["-fPIC", "-shared", "-O3", "-march=native"]
elif sys.platform == "darwin":
    cpp_flags += ["-O3", "-march=native", '-stdlib=libc++', '-mmacosx-version-min=10.9']
    link_flags += ["-v", '-mmacosx-version-min=10.9']
    cvars = sysconfig.get_config_vars()
    cvars['LDSHARED'] = cvars['LDSHARED'].replace('-bundle', '-dynamiclib')


setup(
    ext_modules=[
        Extension(
            name="fast_median",
            sources=[os.path.join(FAST_MEDIAN_PATH, "fast_median.cpp")],
            define_macros=[],
            extra_compile_args=cpp_flags,
            extra_link_args=link_flags,
            language='c++',
            optional=False
        ),
    ]
)
