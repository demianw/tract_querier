from .. import queries_preprocess, queries_syntax_check
from nose.tools import nottest

import os
import fnmatch


def test_query_files(
    folder=os.path.join(os.path.dirname(__file__), '..', 'data')
):
    files = fnmatch.filter(os.listdir(folder), '*qry')
    for f in files:
        yield query_file_test, os.path.join(folder, f), [folder]


@nottest
def query_file_test(filename, include_folders):
    buf = open(filename).read()
    query_body = queries_preprocess(
        buf, filename=filename, include_folders=include_folders
    )

    queries_syntax_check(query_body)
