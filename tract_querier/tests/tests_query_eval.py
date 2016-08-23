from .. import query_processor
from nose.tools import assert_true, assert_equal

from numpy import random
import ast

# Ten tracts traversing random labels
another_set = True
while (another_set):
    tracts_labels = dict([(i, set(random.randint(100, size=2))) for i in range(100)])
    labels_tracts = query_processor.labels_for_tracts(tracts_labels)
    another_set = 0 not in labels_tracts.keys() or 1 not in labels_tracts.keys()


tracts_in_0 = set().union(*[labels_tracts[label] for label in labels_tracts if label == 0])
tracts_in_all_but_0 = set().union(*[labels_tracts[label] for label in labels_tracts if label != 0])
tract_in_label_0_uniquely = labels_tracts[0].difference(tracts_in_all_but_0)


class DummySpatialIndexing:

    def __init__(
        self,
        crossing_tracts_labels, crossing_labels_tracts,
        ending_tracts_labels, ending_labels_tracts,
        label_bounding_boxes, tract_bounding_boxes
    ):
        self.crossing_tracts_labels = crossing_tracts_labels
        self.crossing_labels_tracts = crossing_labels_tracts
        self.ending_tracts_labels = ending_tracts_labels
        self.ending_labels_tracts = ending_labels_tracts
        self.label_bounding_boxes = label_bounding_boxes
        self.tract_bounding_boxes = tract_bounding_boxes

dummy_spatial_indexing = DummySpatialIndexing(tracts_labels, labels_tracts, ({}, {}), ({}, {}), {}, {})
empty_spatial_indexing = DummySpatialIndexing({}, {}, ({}, {}), ({}, {}), {}, {})


def test_assign():
    query_evaluator = query_processor.EvaluateQueries(dummy_spatial_indexing)
    query_evaluator.visit(ast.parse("A=0"))
    assert_true((
        'A' in query_evaluator.queries_to_save and
        query_evaluator.evaluated_queries_info['A'].tracts == labels_tracts[0] and
        query_evaluator.evaluated_queries_info['A'].labels == set((0,))
    ))


def test_assign_attr():
    query_evaluator = query_processor.EvaluateQueries(dummy_spatial_indexing)
    query_evaluator.visit(ast.parse("a.left=0"))
    assert_true((
        'a.left' in query_evaluator.queries_to_save and
        query_evaluator.evaluated_queries_info['a.left'].tracts == labels_tracts[0] and
        query_evaluator.evaluated_queries_info['a.left'].labels == set((0,))
    ))


def test_assign_side():
    query_evaluator = query_processor.EvaluateQueries(empty_spatial_indexing)

    queries_labels = {
        'a.left': set([3, 6]),
        'a.right': set([4, 5]),
        'b.left': set([3]),
        'b.right': set([4]),
        'c.left': set([5]),
        'c.right': set([6])
    }

    queries_tracts = {
        'a.left': set([]),
        'a.right': set([]),
        'b.left': set([]),
        'b.right': set([]),
        'c.left': set([]),
        'c.right': set([])
    }

    query = r"""
b.left=3 ;
b.right = 4;
c.left = 5;
c.right = 6;
a.side = b.side or c.opposite
    """

    query_evaluator.visit(ast.parse(query))

    assert_equal({k: v.labels for k, v in query_evaluator.evaluated_queries_info.items()}, queries_labels)
    assert_equal({k: v.tracts for k, v in query_evaluator.evaluated_queries_info.items()}, queries_tracts)


def test_assign_str():
    query_evaluator = query_processor.EvaluateQueries(empty_spatial_indexing)

    queries_labels = {
        'b.left': set([3]),
        'b.right': set([4]),
        'c.left': set([5]),
        'c.right': set([6]),
        'h': set([3, 5])
    }

    queries_tracts = {
        'b.left': set([]),
        'b.right': set([]),
        'c.left': set([]),
        'c.right': set([]),
        'h': set([])
    }

    query = """
b.left=3
b.right = 4
c.left = 5
c.right = 6
h = '*left'
    """

    query_evaluator.visit(ast.parse(query))

    assert_equal({k: v.labels for k, v in query_evaluator.evaluated_queries_info.items()}, queries_labels)
    assert_equal({k: v.tracts for k, v in query_evaluator.evaluated_queries_info.items()}, queries_tracts)


def test_for_list():
    query_evaluator = query_processor.EvaluateQueries(empty_spatial_indexing)

    queries_tracts = {
        'a.left': set([]),
        'a.right': set([]),
        'b.left': set([]),
        'b.right': set([]),
        'c.left': set([]),
        'c.right': set([]),
        'd.left': set([]),
        'd.right': set([]),
        'e.left': set([]),
        'e.right': set([])
    }

    query = """
a.left= 0
b.left= 1
c.left= 2
d.left= 3
e.left= 4
for i in [a,b,c,d,e]: i.right = i.left
    """

    query_evaluator.visit(ast.parse(query))

    assert_equal({k: v.tracts for k, v in query_evaluator.evaluated_queries_info.items()}, queries_tracts)


def test_for_str():
    query_evaluator = query_processor.EvaluateQueries(empty_spatial_indexing)

    queries_tracts = {
        'a.left': set([]),
        'a.left.right': set([]),
        'b.left': set([]),
        'b.left.right': set([]),
        'c.left': set([]),
        'c.left.right': set([]),
        'd.left': set([]),
        'd.left.right': set([]),
        'e.left': set([]),
        'e.left.right': set([])
    }

    query = """
a.left= 0
b.left= 1
c.left= 2
d.left= 3
e.left= 4
for i in '*left': i.right = i
    """

    query_evaluator.visit(ast.parse(query))

    assert_equal({k: v.tracts for k, v in query_evaluator.evaluated_queries_info.items()}, queries_tracts)


def test_add():
    query_evaluator = query_processor.EvaluateQueries(dummy_spatial_indexing)
    query_evaluator.visit(ast.parse("A=0+1"))
    assert_true((
        'A' in query_evaluator.queries_to_save and
        query_evaluator.evaluated_queries_info['A'].tracts == labels_tracts[0].union(labels_tracts[1]) and
        query_evaluator.evaluated_queries_info['A'].labels == set((0, 1))
    ))


def test_mult():
    query_evaluator = query_processor.EvaluateQueries(dummy_spatial_indexing)
    query_evaluator.visit(ast.parse("A=0 * 1"))
    assert_true((
        'A' in query_evaluator.queries_to_save and
        query_evaluator.evaluated_queries_info['A'].tracts == labels_tracts[0].intersection(labels_tracts[1]) and
        query_evaluator.evaluated_queries_info['A'].labels == set((0, 1))
    ))


def test_sub():
    query_evaluator = query_processor.EvaluateQueries(dummy_spatial_indexing)
    query_evaluator.visit(ast.parse("A=(0 + 1) - 1"))
    assert_true((
        'A' in query_evaluator.queries_to_save and
        query_evaluator.evaluated_queries_info['A'].tracts == labels_tracts[0].difference(labels_tracts[1]) and
        query_evaluator.evaluated_queries_info['A'].labels == set((0,))
    ))


def test_or():
    query_evaluator = query_processor.EvaluateQueries(dummy_spatial_indexing)
    query_evaluator.visit(ast.parse("A=0 or 1"))
    assert_true((
        'A' in query_evaluator.queries_to_save and
        query_evaluator.evaluated_queries_info['A'].tracts == labels_tracts[0].union(labels_tracts[1]) and
        query_evaluator.evaluated_queries_info['A'].labels == set((0, 1))
    ))


def test_and():
    query_evaluator = query_processor.EvaluateQueries(dummy_spatial_indexing)
    query_evaluator.visit(ast.parse("A=0 and 1"))
    assert_true((
        'A' in query_evaluator.queries_to_save and
        query_evaluator.evaluated_queries_info['A'].tracts == labels_tracts[0].intersection(labels_tracts[1]) and
        query_evaluator.evaluated_queries_info['A'].labels == set((0, 1))
    ))


def test_not_in():
    query_evaluator = query_processor.EvaluateQueries(dummy_spatial_indexing)
    query_evaluator.visit(ast.parse("A=0 or 1 not in 1"))
    assert_true((
        'A' in query_evaluator.queries_to_save and
        query_evaluator.evaluated_queries_info['A'].tracts == labels_tracts[0].difference(labels_tracts[1]) and
        query_evaluator.evaluated_queries_info['A'].labels == set((0,))
    ))


def test_only_sign():
    query_evaluator = query_processor.EvaluateQueries(dummy_spatial_indexing)
    query_evaluator.visit(ast.parse("A=~0"))
    assert_true((
        'A' in query_evaluator.queries_to_save and
        query_evaluator.evaluated_queries_info['A'].tracts == tract_in_label_0_uniquely and
        query_evaluator.evaluated_queries_info['A'].labels == set((0,))
    ))


def test_only():
    query_evaluator = query_processor.EvaluateQueries(dummy_spatial_indexing)
    query_evaluator.visit(ast.parse("A=only(0)"))
    assert_true((
        'A' in query_evaluator.queries_to_save and
        query_evaluator.evaluated_queries_info['A'].tracts == tract_in_label_0_uniquely and
        query_evaluator.evaluated_queries_info['A'].labels == set((0,))
    ))


def test_unsaved_query():
    query_evaluator = query_processor.EvaluateQueries(dummy_spatial_indexing)
    query_evaluator.visit(ast.parse("A|=0"))
    assert_true((
        'A' not in query_evaluator.queries_to_save and
        query_evaluator.evaluated_queries_info['A'].tracts == labels_tracts[0] and
        query_evaluator.evaluated_queries_info['A'].labels == set((0,))
    ))


def test_symbolic_assignment():
    query_evaluator = query_processor.EvaluateQueries(dummy_spatial_indexing)
    query_evaluator.visit(ast.parse("A=0; B=A"))
    assert_true((
        'B' in query_evaluator.queries_to_save and
        query_evaluator.evaluated_queries_info['B'].tracts == labels_tracts[0] and
        query_evaluator.evaluated_queries_info['B'].labels == set((0,))
    ))


def test_unarySub():
    query_evaluator = query_processor.EvaluateQueries(dummy_spatial_indexing)
    query_evaluator.visit(ast.parse("B=0; A=-B"))
    assert_true((
        'A' in query_evaluator.queries_to_save and
        query_evaluator.evaluated_queries_info['A'].tracts == tracts_in_all_but_0 and
        query_evaluator.evaluated_queries_info['A'].labels == set(labels_tracts.keys()).difference((0,))
    ))


def test_not():
    query_evaluator = query_processor.EvaluateQueries(dummy_spatial_indexing)
    query_evaluator.visit(ast.parse("A= not 0"))
    assert_true((
        'A' in query_evaluator.queries_to_save and
        query_evaluator.evaluated_queries_info['A'].tracts == tracts_in_all_but_0 and
        query_evaluator.evaluated_queries_info['A'].labels == set(labels_tracts.keys()).difference((0,))
    ))
