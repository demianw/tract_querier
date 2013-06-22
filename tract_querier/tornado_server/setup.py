#!/usr/bin/env python


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('tornado_server', parent_package, top_path)
    config.set_options(quiet=True)

    config.add_data_files((('.'), ['index.html', 'FreeSurferColorLUT.txt']))
    config.add_data_dir(('./js', 'js'))
    config.add_data_dir(('./css', 'css'))

    return config
