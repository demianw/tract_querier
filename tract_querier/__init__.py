import os
import sysconfig

from query_processor import *
from tract_label_indices import *
from shell import *
import tractography


def find_queries_path():
    default_path = sysconfig.get_path(name='data')
    possible_paths = [os.path.join(default_path, 'tract_querier', 'queries')]

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

#import tract_metrics

__version__ = 0.1
