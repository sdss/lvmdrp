# !usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2017-12-05 12:01:21
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2017-12-05 12:19:32

from __future__ import absolute_import, division, print_function


class DrpError(Exception):
    """A custom core Drp exception"""

    def __init__(self, message=None):

        message = 'There has been an error' \
            if not message else message

        super(DrpError, self).__init__(message)


class DrpNotImplemented(DrpError):
    """A custom exception for not yet implemented features."""

    def __init__(self, message=None):

        message = 'This feature is not implemented yet.' \
            if not message else message

        super(DrpNotImplemented, self).__init__(message)


class DrpAPIError(DrpError):
    """A custom exception for API errors"""

    def __init__(self, message=None):
        if not message:
            message = 'Error with Http Response from Drp API'
        else:
            message = 'Http response error from Drp API. {0}'.format(message)

        super(DrpAPIError, self).__init__(message)


class DrpApiAuthError(DrpAPIError):
    """A custom exception for API authentication errors"""
    pass


class DrpMissingDependency(DrpError):
    """A custom exception for missing dependencies."""
    pass


class DrpWarning(Warning):
    """Base warning for Drp."""


class DrpUserWarning(UserWarning, DrpWarning):
    """The primary warning class."""
    pass


class DrpSkippedTestWarning(DrpUserWarning):
    """A warning for when a test is skipped."""
    pass


class DrpDeprecationWarning(DrpUserWarning):
    """A warning for deprecated features."""
    pass
