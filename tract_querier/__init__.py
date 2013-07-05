import os

from query_processor import *
from tract_label_indices import *
from shell import *
import tractography


default_queries_folder = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    '..', '..', '..', '..',
    'tract_querier', 'queries'
))

#import tract_metrics

__version__ = 0.1
