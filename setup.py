# encoding: utf-8
#
# setup.py
#
# BUG: this script should take an optional master configuration template, other wise use the one shipped with the package

from __future__ import absolute_import, division, print_function

import os
import sys

from setuptools import find_packages, setup


# The NAME variable should be of the format "sdss-drp".
# Please check your NAME adheres to that format.
NAME = "lvmdrp"
VERSION = "0.1.1dev"


def run(packages, install_requires):
    setup(
        name=NAME,
        version=VERSION,
        license="BSD3",
        description="SDSSV-LVM Data Reduction Pipeline",
        long_description=open("README.rst").read(),
        author="Eric Pellegrini",
        author_email="ericpellegrini@outlook.com",
        keywords="astronomy software",
        url="https://github.com/sdss/lvmdrp",
        include_package_data=True,
        python_requires=">=3.8",
        packages=packages,
        install_requires=install_requires,
        package_dir={"": "python"},
        scripts=["bin/drp", "bin/pix2wave",
            "bin/build_super_pixmask",
            "bin/build_super_arc",
            "bin/build_super_waves",
            "bin/build_super_trace"
        ],
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: BSD License",
            "Natural Language :: English",
            "Operating System :: OS Independent",
            "Programming Language :: Python",
            "Programming Language :: Python :: 3.8",
            "Topic :: Documentation :: Sphinx",
            "Topic :: Software Development :: Libraries :: Python Modules",
        ],
    )


def parse_requirements(reqfile_path):
    """
    Returns the parsed requirements from a requirements .txt file
    """
    install_requires = []
    with open(reqfile_path, "r") as r:
        for requirement in r.readlines():
            requirement = requirement.strip()
            if requirement.startswith("-r"):
                install_requires.extend(
                    parse_requirements(requirement.replace("-r ", ""))
                )
            else:
                install_requires.append(requirement)
    return install_requires


def get_requirements():
    """Get the proper requirements file based on the optional argument"""

    requirements_file = os.path.join(os.path.dirname(__file__), "requirements_all.txt")
    install_requires = parse_requirements(requirements_file)
    return install_requires


def remove_args(parser):
    """Remove custom arguments from the parser"""

    arguments = []
    for action in list(parser._get_optional_actions()):
        if "--help" not in action.option_strings:
            arguments += action.option_strings

    for arg in arguments:
        if arg in sys.argv:
            sys.argv.remove(arg)


if __name__ == "__main__":
    # Get the proper requirements file
    install_requires = get_requirements()

    # Have distutils find the packages
    packages = find_packages(where="python")

    # Runs distutils
    run(packages, install_requires)
