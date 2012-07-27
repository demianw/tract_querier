from .. import query_processor
from numpy import random
import ast

#Ten fibers traversing random labels
another_set = True
while (another_set):
    fibers_labels = dict([(i, set(random.randint(100, size=2))) for i in xrange(100)])
    labels_fibers = query_processor.labels_for_fibers(fibers_labels)
    another_set = 0 not in labels_fibers.keys() or 1 not in labels_fibers.keys()


fibers_in_all_but_0 = set().union(*[labels_fibers[label] for label in labels_fibers if label!=0])
fiber_in_label_0_uniquely = labels_fibers[0].difference(fibers_in_all_but_0)


def test_assign():
    query_evaluator = query_processor.EvaluateQueries(fibers_labels, labels_fibers)
    query_evaluator.visit(ast.parse("A=0"))
    assert((
        'A' in query_evaluator.queries_to_save and
        query_evaluator.evaluated_queries_fibers['A'] == labels_fibers[0] and
        query_evaluator.evaluated_queries_labels['A'] == set((0,))
    ))

def test_assign_attr():
    query_evaluator = query_processor.EvaluateQueries(fibers_labels, labels_fibers)
    query_evaluator.visit(ast.parse("a.left=0"))
    assert((
        'a.left' in query_evaluator.queries_to_save and
        query_evaluator.evaluated_queries_fibers['a.left'] == labels_fibers[0] and
        query_evaluator.evaluated_queries_labels['a.left'] == set((0,))
    ))


def test_assign_side():
    query_evaluator = query_processor.EvaluateQueries({}, {})

    queries_labels = {
        'a.left': set([3, 6]),
        'a.right': set([4, 5]),
        'b.left': set([3]),
        'b.right': set([4]),
        'c.left': set([5]),
        'c.right': set([6])
    }

    queries_fibers = {
        'a.left': set([]),
        'a.right': set([]),
        'b.left': set([]),
        'b.right': set([]),
        'c.left': set([]),
        'c.right': set([])
    }

    query =  r"""
b.left=3 ;
b.right = 4;
c.left = 5;
c.right = 6;
a.side = b.side or c.opposite
    """

    query_evaluator.visit(ast.parse(query))

    assert(query_evaluator.evaluated_queries_labels == queries_labels)
    assert(query_evaluator.evaluated_queries_fibers == queries_fibers)


def test_assign_str():
    query_evaluator = query_processor.EvaluateQueries({}, {})

    queries_labels = {
        'b.left': set([3]),
        'b.right': set([4]),
        'c.left': set([5]),
        'c.right': set([6]),
        'h': set([3, 5])
    }

    queries_fibers = {
        'b.left': set([]),
        'b.right': set([]),
        'c.left': set([]),
        'c.right': set([]),
        'h': set([])
    }

    query =  """
b.left=3
b.right = 4
c.left = 5
c.right = 6
h = '*left'
    """

    query_evaluator.visit(ast.parse(query))

    assert(query_evaluator.evaluated_queries_labels == queries_labels)
    assert(query_evaluator.evaluated_queries_fibers == queries_fibers)


def test_for_list():
    query_evaluator = query_processor.EvaluateQueries({}, {})

    queries_fibers = {
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

    query ="""
a.left= 0
b.left= 1
c.left= 2
d.left= 3
e.left= 4
for i in [a,b,c,d,e]: i.right = i.left
    """

    query_evaluator.visit(ast.parse(query))

    assert(query_evaluator.evaluated_queries_fibers == queries_fibers)

def test_for_str():
    query_evaluator = query_processor.EvaluateQueries({}, {})

    queries_fibers = {
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

    query ="""
a.left= 0
b.left= 1
c.left= 2
d.left= 3
e.left= 4
for i in '*left': i.right = i
    """

    query_evaluator.visit(ast.parse(query))

    assert(query_evaluator.evaluated_queries_fibers == queries_fibers)


def test_add():
    query_evaluator = query_processor.EvaluateQueries(fibers_labels, labels_fibers)
    query_evaluator.visit(ast.parse("A=0+1"))
    assert((
        'A' in query_evaluator.queries_to_save and
        query_evaluator.evaluated_queries_fibers['A'] == labels_fibers[0].union(labels_fibers[1]) and
        query_evaluator.evaluated_queries_labels['A'] == set((0, 1))
    ))

def test_mult():
    query_evaluator = query_processor.EvaluateQueries(fibers_labels, labels_fibers)
    query_evaluator.visit(ast.parse("A=0 * 1"))
    assert((
        'A' in query_evaluator.queries_to_save and
        query_evaluator.evaluated_queries_fibers['A'] == labels_fibers[0].intersection(labels_fibers[1]) and
        query_evaluator.evaluated_queries_labels['A'] == set((0, 1))
    ))

def test_sub():
    query_evaluator = query_processor.EvaluateQueries(fibers_labels, labels_fibers)
    query_evaluator.visit(ast.parse("A=(0 + 1) - 1"))
    assert((
        'A' in query_evaluator.queries_to_save and
        query_evaluator.evaluated_queries_fibers['A'] == labels_fibers[0].difference(labels_fibers[1]) and
        query_evaluator.evaluated_queries_labels['A'] == set((0,))
    ))

def test_or():
    query_evaluator = query_processor.EvaluateQueries(fibers_labels, labels_fibers)
    query_evaluator.visit(ast.parse("A=0 or 1"))
    assert((
        'A' in query_evaluator.queries_to_save and
        query_evaluator.evaluated_queries_fibers['A'] == labels_fibers[0].union(labels_fibers[1]) and
        query_evaluator.evaluated_queries_labels['A'] == set((0, 1))
    ))

def test_and():
    query_evaluator = query_processor.EvaluateQueries(fibers_labels, labels_fibers)
    query_evaluator.visit(ast.parse("A=0 and 1"))
    assert((
        'A' in query_evaluator.queries_to_save and
        query_evaluator.evaluated_queries_fibers['A'] == labels_fibers[0].intersection(labels_fibers[1]) and
        query_evaluator.evaluated_queries_labels['A'] == set((0, 1))
    ))

def test_not_in():
    query_evaluator = query_processor.EvaluateQueries(fibers_labels, labels_fibers)
    query_evaluator.visit(ast.parse("A=0 or 1 not in 1"))
    assert((
        'A' in query_evaluator.queries_to_save and
        query_evaluator.evaluated_queries_fibers['A'] == labels_fibers[0].difference(labels_fibers[1]) and
        query_evaluator.evaluated_queries_labels['A'] == set((0,))
    ))

def test_only_sign():
    query_evaluator = query_processor.EvaluateQueries(fibers_labels, labels_fibers)
    query_evaluator.visit(ast.parse("A=~0"))
    assert((
        'A' in query_evaluator.queries_to_save and
        query_evaluator.evaluated_queries_fibers['A'] == fiber_in_label_0_uniquely and
        query_evaluator.evaluated_queries_labels['A'] == set((0,))
    ))

def test_only():
    query_evaluator = query_processor.EvaluateQueries(fibers_labels, labels_fibers)
    query_evaluator.visit(ast.parse("A=only(0)"))
    assert((
        'A' in query_evaluator.queries_to_save and
        query_evaluator.evaluated_queries_fibers['A'] == fiber_in_label_0_uniquely and
        query_evaluator.evaluated_queries_labels['A'] == set((0,))
    ))


def test_unsaved_query():
    query_evaluator = query_processor.EvaluateQueries(fibers_labels, labels_fibers)
    query_evaluator.visit(ast.parse("A|=0"))
    assert((
        'A' not in query_evaluator.queries_to_save and
        query_evaluator.evaluated_queries_fibers['A'] == labels_fibers[0] and
        query_evaluator.evaluated_queries_labels['A'] == set((0,))
    ))

def test_symbolic_assignment():
    query_evaluator = query_processor.EvaluateQueries(fibers_labels, labels_fibers)
    query_evaluator.visit(ast.parse("A=0; B=A"))
    assert((
        'B' in query_evaluator.queries_to_save and
        query_evaluator.evaluated_queries_fibers['B'] == labels_fibers[0] and
        query_evaluator.evaluated_queries_labels['B'] == set((0,))
    ))

def test_unarySub():
    query_evaluator = query_processor.EvaluateQueries(fibers_labels, labels_fibers)
    query_evaluator.visit(ast.parse("B=0; A=-B"))
    assert((
        'A' in query_evaluator.queries_to_save and
        query_evaluator.evaluated_queries_fibers['A'] == fibers_in_all_but_0 and
        query_evaluator.evaluated_queries_labels['A'] == set(labels_fibers.keys()).difference((0,))
    ))

def test_not():
    query_evaluator = query_processor.EvaluateQueries(fibers_labels, labels_fibers)
    query_evaluator.visit(ast.parse("A= not 0"))
    assert((
        'A' in query_evaluator.queries_to_save and
        query_evaluator.evaluated_queries_fibers['A'] == fibers_in_all_but_0 and
        query_evaluator.evaluated_queries_labels['A'] == set(labels_fibers.keys()).difference((0,))
    ))




