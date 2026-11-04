import numpy as np
import pytest

from .. import query_processor
from ..aabb import BoundingBox
from ..tract_label_indices import compute_tract_bounding_boxes


class SpatialIndexing:
    # Two tracts: tract 0 crosses label 1, tract 1 crosses label 2
    def __init__(self):
        tracts = [
            np.array([[0., 0, 0], [1, 1, 1]]),
            np.array([[5., 5, 5], [6, 6, 6]])
        ]
        self.crossing_tracts_labels = {0: {1}, 1: {2}}
        self.crossing_labels_tracts = query_processor.labels_for_tracts(
            self.crossing_tracts_labels
        )
        self.ending_tracts_labels = ({0: 1, 1: 2}, {0: 1, 1: 2})
        self.ending_labels_tracts = ({1: {0}, 2: {1}}, {1: {0}, 2: {1}})
        self.label_bounding_boxes = {
            1: BoundingBox(tracts[0]), 2: BoundingBox(tracts[1])
        }
        self.tract_bounding_boxes = compute_tract_bounding_boxes(tracts)
        self.tract_endpoints_pos = np.array([[t[0], t[-1]] for t in tracts])


@pytest.mark.parametrize("query,error,message", [
    ("a = 1\nb = a.x.y", query_processor.TractQuerierSyntaxError,
     "line 2: a.x.y"),
    ("a = 1\nfor x in 3:\n    b = x", query_processor.TractQuerierSyntaxError,
     "FOR statement in line 2"),
    ("import broken.qry", query_processor.TractQuerierSyntaxError,
     "broken.qry line 2"),
    ("a = 1\nb = medial_of(a)", query_processor.TractQuerierSyntaxError,
     "medial_of needs a query ending in .left or .right. Line 2"),
    ("a = 1\nb = lateral_of(a)", query_processor.TractQuerierSyntaxError,
     "lateral_of needs a query ending in .left or .right. Line 2"),
    ("a.foo = 1\nb = anterior_of(a.foo)",
     query_processor.TractQuerierSyntaxError,
     "Attribute not recognized for relative specification.Line 2"),
    ("a = 1 - 1\nb = anterior_of(a)", query_processor.TractQuerierLabelNotFound,
     "anterior_of applied to a query with no labels. Line 2"),
    ("a = 'nothing*'\nb = posterior_of(a)",
     query_processor.TractQuerierLabelNotFound,
     "posterior_of applied to a query with no labels. Line 2"),
])
def test_invalid_query_error(tmp_path, query, error, message):
    (tmp_path / "broken.qry").write_text("a = 1\nb = (a or\n")
    with pytest.raises(error) as excinfo:
        body = query_processor.queries_preprocess(
            query, filename='test.qry', include_folders=[str(tmp_path)]
        )
        query_processor.eval_queries(body, SpatialIndexing())
    assert message in excinfo.value.value


@pytest.mark.parametrize("query", [
    "a = 1\nb = a.x.y",
    "a = 1\nfor x in 3:\n    b = x",
    "a = 1\nb = medial_of(a)",
])
def test_invalid_query_syntax_check(query):
    body = query_processor.queries_preprocess(query, filename='test.qry')
    with pytest.raises(query_processor.TractQuerierSyntaxError):
        query_processor.queries_syntax_check(body)


def test_relative_terms():
    body = query_processor.queries_preprocess(
        "a.left = 1\nb = medial_of(a.left)\nc = anterior_of(a.left)",
        filename='test.qry'
    )
    assert query_processor.eval_queries(body, SpatialIndexing()) == {
        'a.left': {0}, 'b': {1}, 'c': {1}
    }
