#!/usr/bin/env python
# encoding: utf-8
#

import os

from lvmdrp import log


REQUIRED_VARS = ["LVMCORE_DIR", "SAS_BASE_DIR", "LVM_MASTER_DIR"]


def check_environment_variables():
    """Test that all required environment variables are set.
    """
    missing_vars = [var for var in REQUIRED_VARS if var not in os.environ]
    
    if missing_vars:
        missing_str = ", ".join(missing_vars)
        log.error(f"the following required environment variables are not set: {missing_str}")
        log.info("see the README for more information about how to set these variables")
        exit(1)
    else:
        for var in REQUIRED_VARS:
            log.info(f"{var}='{os.environ[var]}'")
    
    log.info("all required environment variables for the LVM DRP are set")

if __name__ == "__main__":
    check_environment_variables()