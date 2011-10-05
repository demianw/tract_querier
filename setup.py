#!/usr/bin/env python
from distutils.core import setup

DISTNAME = 'tract_querier'
DESCRIPTION = 'Complex queries for full brain tractographies with a registered template on top of them'
LONG_DESCRIPTION = '' #open('README.rst').read()
MAINTAINER = 'Demian Wassermann'
MAINTAINER_EMAIL = 'demian@bwh.harvard.edu'
URL = ''
LICENSE = ''
DOWNLOAD_URL = ''
VERSION = '0.1'

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration(None, parent_package, top_path)
    config.set_options(quiet=True)
    config.add_subpackage('tract_querier')
    config.add_data_files(
        ('queries',[
        'data/FreeSurfer.qry',
        'data/JHU_MNI_SS_WMPM_Type-II.qry',
        'data/freesurfer_queries.qry',
        'data/mori_queries.qry',
        ])
    )
    return config

if __name__ == "__main__":
    setup(
          name=DISTNAME,
          maintainer=MAINTAINER,
          include_package_data=False,
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          license=LICENSE,
          url=URL,
          version=VERSION,
          download_url=DOWNLOAD_URL,
          long_description=LONG_DESCRIPTION,
          zip_safe=False,  # the package can run out of an .egg file
          classifiers=[
              'Intended Audience :: Science/Research',
              'Intended Audience :: Developers',
              'Programming Language :: Python',
              'Topic :: Software Development',
              'Topic :: Scientific/Engineering',
              'Operating System :: Microsoft :: Windows',
              'Operating System :: POSIX',
              'Operating System :: Unix',
              'Operating System :: MacOS'
             ],
            scripts=[
                'scripts/tract_querier',
              ],
            **(configuration().todict())
    )
