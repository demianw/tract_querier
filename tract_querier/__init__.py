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
        possible_paths.append(os.path.join(default_path, 'tract_querier',
                                           'queries'))

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


def search_file_and_create_query_body(queries_string):
    # Search order precidence for .qry files
    # 1. Current directory
    # 2. Command line options specified are respected second
    # 3. Default query location thrid
    # 4. Source Tree 4th
    qry_search_folders = []
    qry_search_folders.extend([os.getcwd()])
    default_queries_folder = find_queries_path()

    if os.path.exists(default_queries_folder):
        qry_search_folders.extend([default_queries_folder])

    # Source Tree Data
    source_tree_data_path = os.path.abspath(
        os.path.join(os.path.dirname(os.path.dirname(__file__)),
                     'tract_querier', 'data')
    )
    if os.path.exists(source_tree_data_path):
        qry_search_folders.extend([source_tree_data_path])
    
    found = False
    try:
        if os.path.exists(queries_string):
            query_script = open(queries_string).read()
            query_filename = queries_string
        else:
            found = False
            for folder in qry_search_folders:
                file_ = os.path.join(folder, queries_string)
                if os.path.exists(file_):
                    found = True
                    break
            if found:
                query_script = open(file_).read()
                query_filename = file_
            else:
                query_script = queries_string
                query_filename = '<script>'

        query_file_body = queries_preprocess(
            query_script,
            filename=query_filename,
            include_folders=qry_search_folders
        )

        queries_syntax_check(query_file_body)
    except TractQuerierSyntaxError or TractographySpatialIndexing as e:
        if not found:
            raise ValueError(f"The file {queries_string} does not exist")

        raise ValueError("Error parsing the query file")
    # except tract_querier.TractographySpatialIndexing as e:
    #     parser.error(e.value)

    return query_file_body, query_script, qry_search_folders


default_queries_folder = find_queries_path()

__version__ = 0.2
