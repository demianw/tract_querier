#!/usr/bin/env python


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('tract_querier', parent_package, top_path)
    config.add_subpackage('tractography')
    config.add_subpackage('tract_math')
    config.add_subpackage('code_util')
    config.add_subpackage('tensor')
    config.add_subpackage('nipype')
    config.add_data_files(('queries', [
        'data/FreeSurfer.qry',
        'data/JHU_MNI_SS_WMPM_Type_I.qry',
        'data/JHU_MNI_SS_WMPM_Type_II.qry',
        'data/freesurfer_queries.qry',
        'data/mori_queries.qry',
    ]))
    return config

if __name__ == '__main__':
    from distutils.core import setup
    setup(**configuration(top_path='').todict())
