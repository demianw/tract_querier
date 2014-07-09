import os
import sysconfig

from query_processor import *
from tract_label_indices import *
from shell import *
import tractography


def find_queries_path():
    default_data_path = sysconfig.get_path(name='data')
    queries_path = os.path.join(default_data_path, 'tract_querier', 'queries')

    if not os.path.exists(queries_path):
        # Try to manage Virtual Environments on some OSes,
        # where data is not put the 'local' subdirectory,
        # but at the root of the virtual environment.
        if default_data_path.endswith('local'):
            queries_path = os.path.join(default_data_path.rsplit('local', 1)[0],
                                        'tract_querier', 'queries')

            if not os.path.exists(queries_path):
                raise Exception('Default path for queries not found')

    return queries_path


default_queries_folder = find_queries_path()

#import tract_metrics

__version__ = 0.1
