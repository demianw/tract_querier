from os import path

import tract_querier


def test_default_queries_folder_has_bundled_queries():
    folder = tract_querier.default_queries_folder
    assert isinstance(folder, str)
    assert path.isfile(path.join(folder, 'FreeSurfer.qry'))
