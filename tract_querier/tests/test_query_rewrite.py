from .. import query_processor

from nose.tools import assert_equal, assert_not_equal
from unittest import expectedFailure, skip

import ast

import parser
import token
import symbol


def match(pattern, data, vars=None):
    if vars is None:
        vars = {}
    if type(pattern) is list:
        vars[pattern[0]] = data
        return 1, vars
    if type(pattern) is not tuple:
        return (pattern == data), vars
    if len(data) != len(pattern):
        return 0, vars
    for pattern, data in map(None, pattern, data):
        same, vars = match(pattern, data, vars)
        if not same:
            break
    return same, vars


@skip
def test_rewrite_notin_precedence():
    code1 = "a and b not in c"
    code2 = "(a and b) not in c"
    code3 = "a and (b not in c)"
    code4 = "(b not in c) and a"

    rw = query_processor.RewriteChangeNotInPrescedence()

    tree1 = ast.parse(code1)
    tree2 = ast.parse(code2)
    tree3 = ast.parse(code3)

    tree1_rw = ast.parse(code1)
    tree2_rw = ast.parse(code2)
    tree3_rw = ast.parse(code3)

    rw.visit(tree1_rw)
    rw.visit(tree2_rw)
    rw.visit(tree3_rw)

    assert_not_equal(ast.dump(tree1), ast.dump(tree2))
    assert_equal(ast.dump(tree2), ast.dump(tree2_rw))
    assert_equal(ast.dump(tree1_rw), ast.dump(tree2))

    assert_equal(ast.dump(tree3), ast.dump(tree3_rw))

    assert_equal(ast.dump(tree1), ast.dump(tree3_rw))
