import numpy as np

from .. import query_processor
from ..tract_label_indices import TractographySpatialIndexing


# Atlas with three slabs along the anterior-posterior axis:
# label 1 for y < 5, label 3 for 5 <= y < 15 and label 2 for y >= 15
image = np.ones((10, 20, 10), dtype=int)
image[:, 5:15] = 3
image[:, 15:] = 2
affine = np.eye(4)


def tract(y_start, y_end, x=5, z=5):
    y = np.arange(y_start, y_end + 1, dtype=float)
    return np.c_[np.full_like(y, x), y, np.full_like(y, z)]


tracts = [
    tract(1, 18),         # 1 -> 2, long
    tract(1, 3),          # within 1, short
    tract(6, 17, x=3),    # 3 -> 2, long
    tract(16, 18, z=3),   # within 2, short
    tract(2, 18, x=7),    # 1 -> 2, long
    tract(12, 16, z=7),   # 3 -> 2, short
    tract(3, 7, x=2),     # 1 -> 3, short
]
lengths = np.array([np.linalg.norm(np.diff(t, axis=0), axis=1).sum() for t in tracts])

queries = '''
one |= 1
two |= 2
a = one
b = endpoints_in(one)
c = endpoints_in(one) and endpoints_in(two)
d = both_endpoints_in(two)
e = only(one or 3)
f = not 3
g = anterior_of(one)
h = endpoints_in(posterior_of(two))
i = 3 and posterior_of(two)
'''


def evaluate(length_threshold):
    spatial_indexing = TractographySpatialIndexing(
        tracts, image, affine, length_threshold, 2.
    )
    return query_processor.eval_queries(
        query_processor.queries_preprocess(queries), spatial_indexing
    )


def test_length_threshold():
    length_threshold = 5
    results = evaluate(0)
    results_threshold = evaluate(length_threshold)
    assert set(results) == set(results_threshold)
    for name, result in results.items():
        assert len(result) > 0, name
        assert results_threshold[name] == set(
            i for i in result if lengths[i] >= length_threshold
        ), name
