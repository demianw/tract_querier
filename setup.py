#!/usr/bin/env python
from setuptools import setup, find_packages

DISTNAME = 'tract_querier'
DESCRIPTION = \
    'WMQL: Query language for automatic tract extraction from '\
    'full-brain tractographies with '\
    'a registered template on top of them'
LONG_DESCRIPTION = open('README.md').read()
MAINTAINER = 'Demian Wassermann'
MAINTAINER_EMAIL = 'demian@bwh.harvard.edu'
URL = 'http://demianw.github.io/tract_querier'
LICENSE = open('license.rst').read()
DOWNLOAD_URL = 'https://github.com/demianw/tract_querier'
VERSION = '0.1'


if __name__ == "__main__":
    setup(
        name=DISTNAME,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        license=LICENSE,
        url=URL,
        version=VERSION,
        download_url=DOWNLOAD_URL,
        long_description=LONG_DESCRIPTION,
        requires=[
            'numpy(>=1.6)',
            'nibabel(>=1.3)'
        ],
        classifiers=[
            'Intended Audience :: Science/Research',
            'Programming Language :: Python',
            'Topic :: Scientific/Engineering',
            'Operating System :: Microsoft :: Windows',
            'Operating System :: POSIX',
            'Operating System :: Unix',
            'Operating System :: MacOS'
        ],
        scripts=[
            'scripts/tract_querier',
            'scripts/tract_math'
        ],
        test_suite='nose.collector',
        data_files=[
            ('data',
             [
                 'data/FreeSurfer.qry',
                 'data/JHU_MNI_SS_WMPM_Type_I.qry',
                 'data/JHU_MNI_SS_WMPM_Type_II.qry',
                 'data/freesurfer_queries.qry',
                 'data/mori_queries.qry',
             ]
             )
        ],
        include_package_data=True,
        packages=find_packages(),
    )
