#!/usr/bin/env python
# encoding: utf-8

from sdsstools import get_logger, get_package_version


NAME = 'lvmdrp'


# init the logger
log = get_logger(NAME)


__version__ = get_package_version(path=__file__, package_name=NAME)
