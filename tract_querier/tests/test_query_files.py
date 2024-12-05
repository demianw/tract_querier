from .. import queries_preprocess, queries_syntax_check

import os
import fnmatch
import pytest


@pytest.fixture
def data_folder():
    return os.path.join(os.path.dirname(__file__), '..', 'data')

@pytest.mark.parametrize("filename", [
    pytest.param(os.path.join(os.path.join(os.path.dirname(__file__), '..', 'data'), f), id=f)
    for f in fnmatch.filter(os.listdir(os.path.join(os.path.dirname(__file__), '..', 'data')), '*qry')
])
def test_query_files(data_folder, filename):
    query_file_test(filename, [data_folder])


def query_file_test(filename, include_folders):
    buf = open(filename).read()
    query_body = queries_preprocess(
        buf, filename=filename, include_folders=include_folders
    )

    queries_syntax_check(query_body)
