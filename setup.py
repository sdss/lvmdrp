# encoding: utf-8
#
# setup.py
#

import sys
from setuptools import Extension, setup


FAST_MEDIAN_PATH = 'python/cextern/fast_median/src'

if sys.platform == "linux":
    cpp_flags = ["-fPIC", "-shared", "-O3", "-march=native"]
elif sys.platform == "darwin":
    cpp_flags = ["-dynamiclib", "-current_version", "1.0", "-compatibility_version", "1.0", "-O3", "-march=native"]
else:
    cpp_flags = []


if sys.platform == 'darwin':
    cpp_flags += ['-stdlib=libc++', '-mmacosx-version-min=10.9']
    extra_link_args = ["-v", '-mmacosx-version-min=10.9']
    from distutils import sysconfig
    vars = sysconfig.get_config_vars()
    vars['LDSHARED'] = vars['LDSHARED'].replace('-bundle', '-dynamiclib')


setup(
    ext_modules=[
        Extension(
            name="fast_median",
            sources=[FAST_MEDIAN_PATH + "/fast_median.cpp"],
            include_dirs=[FAST_MEDIAN_PATH],
            libraries=[],
            define_macros=[],
            extra_compile_args=cpp_flags,
            extra_link_args=extra_link_args,
            language='c++',
            optional=False
        ),
    ]
)
