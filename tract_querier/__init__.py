import os
import sysconfig

from .query_processor import *
from .tract_label_indices import *
from .shell import *
from . import tractography


def find_queries_path():
    possible_paths = []
    # Try all possible schemes where python expects data to stay.
    for scheme in sysconfig.get_scheme_names():
        default_path = sysconfig.get_path(name='data', scheme=scheme)
        possible_paths.append(os.path.join(default_path, 'tract_querier', 'queries'))

    # Try to manage Virtual Environments on some OSes,
    # where data is not put the 'local' subdirectory,
    # but at the root of the virtual environment.
    if default_path.endswith('local'):
        possible_paths.append(os.path.join(default_path.rsplit('local', 1)[0],
                                           'tract_querier', 'queries'))

    # Case where the Tract_querier is cloned from git and simply
    # added to the python path, without installation.
    possible_paths.append(os.path.abspath(os.path.join(
                                          os.path.dirname(__file__), 'data')))

    paths_found = [path for path in possible_paths if os.path.exists(path)]

    if not paths_found:
        raise Exception('Default path for queries not found')

    return paths_found[0]


default_queries_folder = find_queries_path()

try:
    # Written by setuptools_scm when the package is built or installed.
    from ._version import version as __version__
except ImportError:
    from importlib.metadata import PackageNotFoundError, version
    try:
        __version__ = version('tract_querier')
    except PackageNotFoundError:
        __version__ = '0+unknown'
    del PackageNotFoundError, version
