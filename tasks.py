# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date: 2017-09-27
# @Filename: tasks.py
# @License: BSD 3-Clause
# @Copyright: Brian Cherinka


from __future__ import absolute_import, division, print_function, unicode_literals

import os

from invoke import Collection, task


# This file contains tasks that can be easily run from the shell terminal using the Invoke
# python package. If you do not have invoke, install it with pip install
# To list the tasks available, type invoke --list from the top-level repo directory


@task
def clean_docs(ctx):
    """Cleans up the docs"""
    print("Cleaning the docs")
    ctx.run("rm -rf docs/sphinx/_build")


@task
def build_docs(ctx, clean=False):
    """Builds the Sphinx docs"""

    if clean:
        print("Cleaning the docs")
        ctx.run("rm -rf docs/sphinx/_build")

    print("Building the docs")
    os.chdir("docs/sphinx")
    ctx.run("make html", pty=True)


@task
def show_docs(ctx):
    """Shows the Sphinx docs"""
    print("Showing the docs")
    os.chdir("docs/sphinx/_build/html")
    ctx.run("open ./index.html")


@task
def clean(ctx):
    """Cleans up the crap before a Pip build"""

    print("Cleaning")
    ctx.run("rm -rf htmlcov")
    ctx.run("rm -rf build")
    ctx.run("rm -rf dist")
    ctx.run("rm -rf *.egg-info")
    ctx.run("rm -rf python/*.egg-info")


@task(clean)
def deploy(ctx):
    """Deploy the project to pypi"""
    print("Deploying to Pypi!")
    ctx.run("python setup.py sdist bdist_wheel --universal")
    ctx.run("twine upload dist/*")


os.chdir(os.path.dirname(__file__))

# create a collection of tasks
ns = Collection(clean, deploy)

# create a sub-collection for the doc tasks
docs = Collection("docs")
docs.add_task(build_docs, "build")
docs.add_task(clean_docs, "clean")
docs.add_task(show_docs, "show")
ns.add_collection(docs)
