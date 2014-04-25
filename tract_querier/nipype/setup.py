#!/usr/bin/env python


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('nipype', parent_package, top_path)
    config.set_options(quiet=True)
    return config
