import importlib.resources

from .query_processor import *
from .tract_label_indices import *
from .shell import *
from . import tractography


default_queries_folder = str(importlib.resources.files(__name__) / 'data')

__version__ = 0.1
