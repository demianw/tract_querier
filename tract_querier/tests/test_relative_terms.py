import numpy as np
import pytest

import tract_querier


# A 20 mm cube atlas (identity affine, RAS+) with label 1 in the centre
# (voxels 8..11 along each axis), and one straight tract on each side of it.
IMAGE = np.zeros((20, 20, 20), dtype=int)
IMAGE[8:12, 8:12, 8:12] = 1

TRACTS = {
    'above': [[10., 10., 15.], [10., 10., 18.]],
    'below': [[10., 10., 1.], [10., 10., 4.]],
    'front': [[10., 15., 10.], [10., 18., 10.]],
    'behind': [[10., 1., 10.], [10., 4., 10.]],
}
NAMES = list(TRACTS)


@pytest.mark.parametrize('term, expected', [
    ('superior_of', {'above'}),
    ('inferior_of', {'below'}),
    ('anterior_of', {'front'}),
    ('posterior_of', {'behind'}),
])
def test_relative_terms_direction(term, expected):
    spatial_indexing = tract_querier.TractographySpatialIndexing(
        [np.array(TRACTS[name]) for name in NAMES],
        IMAGE, np.eye(4), 0, 2.
    )
    queries = tract_querier.queries_preprocess(
        "region = 1\nresult = %s(region)\n" % term
    )
    result = tract_querier.eval_queries(queries, spatial_indexing)['result']

    assert {NAMES[i] for i in result} == expected
