import ast
from os import path
from copy import deepcopy
from operator import lt, gt
from itertools import takewhile
import fnmatch

from .code_util import DocStringInheritor

__all__ = ['keywords', 'EvaluateQueries', 'eval_queries', 'queries_syntax_check', 'queries_preprocess']

keywords = [
    'and',
    'or',
    'not in',
    'not',
    'only',
    'endpoints_in',
    'both_endpoints_in',
    'anterior_of',
    'posterior_of',
    'medial_of',
    'lateral_of',
    'inferior_of',
    'superior_of',
]


class FiberQueryInfo(object):

    r"""
    Information about a processed query

    Attribute
    ---------
        tracts : set
            set of tract indices resulting from the query
        labels : set
            set of labels resulting by the query
        tracts_endpoints : (set, set)
            sets of labels of where the tract endpoints are
    """

    def __init__(self, tracts=None, labels=None, tracts_endpoints=None):
        if tracts is None:
            tracts = set()
        if labels is None:
            labels = set()
        if tracts_endpoints is None:
            tracts_endpoints = (set(), set())
        self.tracts = tracts
        self.labels = labels
        self.tracts_endpoints = tracts_endpoints

    def __getattribute__(self, name):
        if name in (
            'update', 'intersection_update', 'union', 'intersection',
            'difference', 'difference_update'
        ):
            return self.set_operation(name)
        else:
            return object.__getattribute__(self, name)

    def copy(self):
        return FiberQueryInfo(
            self.tracts.copy(), self.labels.copy(),
            (self.tracts_endpoints[0].copy(), self.tracts_endpoints[1].copy()),
            #            (self.labels_endpoints[0].copy(), self.labels_endpoints[1].copy()),
        )

    def set_operation(self, name):
        def operation(tract_query_info):
            tracts_op = getattr(self.tracts, name)
            if name == 'intersection':
                name_labels = 'union'
            elif name == 'intersection_update':
                name_labels = 'update'
            else:
                name_labels = name
            labels_op = getattr(self.labels, name_labels)

            new_tracts = tracts_op(tract_query_info.tracts)
            new_labels = labels_op(tract_query_info.labels)

            new_tracts_endpoints = (
                getattr(self.tracts_endpoints[0], name)(tract_query_info.tracts_endpoints[0]),
                getattr(self.tracts_endpoints[1], name)(tract_query_info.tracts_endpoints[1])
            )

#            new_labels_endpoints = (
#                getattr(self.labels_endpoints[0], name_labels)(tract_query_info.labels_endpoints[0]),
#                getattr(self.labels_endpoints[1], name_labels)(tract_query_info.labels_endpoints[1])
#            )

            if name.endswith('update'):
                return self
            else:
                return FiberQueryInfo(
                    new_tracts, new_labels,
                    new_tracts_endpoints,
                )

        return operation


class EndpointQueryInfo:

    def __init__(
        self,
        endpoint_tracts=None,
        endpoint_labels=None,
        endpoint_points=None,
    ):
        if endpoint_tracts is None:
            endpoint_tracts = (set(), set())
        if endpoint_labels is None:
            endpoint_labels = (set(), set())
        if endpoint_points is None:
            endpoint_points = (set(), set())
        self.endpoint_tracts = endpoint_tracts
        self.endpoint_labels = endpoint_labels
        self.endpoint_points = endpoint_points

    def __getattribute__(self, name):
        if name in (
            'update', 'intersection_update', 'union', 'intersection',
            'difference', 'difference_update'
        ):
            return self.set_operation(name)
        else:
            return object.__getattribute__(self, name)

    def set_operation(self, name):
        def operation(endpoint_query_info):

            tracts_op = (
                getattr(self.endpoint_tracts[0], name),
                getattr(self.endpoint_tracts[1], name)
            )
            labels_op = (
                getattr(self.endpoint_labels[0], name),
                getattr(self.endpoint_labels[1], name)
            )
            points_op = (
                getattr(self.endpoint_points[0], name),
                getattr(self.endpoint_points[1], name)
            )

            new_tracts = (
                tracts_op[0](endpoint_query_info.endpoint_tracts[0]),
                tracts_op[1](endpoint_query_info.endpoint_tracts[1])
            )

            new_labels = (
                labels_op[0](endpoint_query_info.endpoint_labels[0]),
                labels_op[1](endpoint_query_info.endpoint_labels[1])
            )

            new_points = (
                points_op[0](endpoint_query_info.endpoint_points[0]),
                points_op[1](endpoint_query_info.endpoint_points[1])
            )

            if name.endswith('update'):
                return self
            else:
                return EndpointQueryInfo(new_tracts, new_labels, new_points)
        return operation


class EvaluateQueries(ast.NodeVisitor):

    r"""
    This class implements the parser to process
    White Matter Query Language modules. By inheriting from
    :py:mod:`ast.NodeVisitor` it uses a syntax close to the
    python language.

    Every node expression visitor has the following signature

    Parameters
    ----------
    node : ast.Node

    Returns
    -------
    tracts : set
        numbers of the tracts that result of this
        query

    labels : set
        numbers of the labels that are traversed by
        the tracts resulting from this query

    """
    __metaclass__ = DocStringInheritor

    relative_terms = [
        'anterior_of',
        'posterior_of',
        'medial_of',
        'lateral_of',
        'inferior_of',
        'superior_of'
    ]

    def __init__(
        self,
        tractography_spatial_indexing,
    ):
        self.tractography_spatial_indexing = tractography_spatial_indexing

        self.evaluated_queries_info = {}
        self.queries_to_save = set()

        self.evaluating_endpoints = False

    def visit_Module(self, node):
        for line in node.body:
            self.visit(line)

    def visit_Compare(self, node):
        if any(not isinstance(op, ast.NotIn) for op in node.ops):
            raise TractQuerierSyntaxError(
                "Invalid syntax in query line %d" % node.lineno
            )

        query_info = self.visit(node.left).copy()
        for value in node.comparators:
            query_info_ = self.visit(value)
            query_info.difference_update(query_info_)

        return query_info

    def visit_BoolOp(self, node):
        query_info = self.visit(node.values[0])
        query_info = query_info.copy()

        if isinstance(node.op, ast.Or):
            for value in node.values[1:]:
                query_info_ = self.visit(value)
                query_info.update(query_info_)

        elif isinstance(node.op, ast.And):
            for value in node.values[1:]:
                query_info_ = self.visit(value)
                query_info.intersection_update(query_info_)

        else:
            return self.generic_visit(node)

        return query_info

    def visit_BinOp(self, node):
        info_left = self.visit(node.left)
        info_right = self.visit(node.right)
        if isinstance(node.op, ast.Add):
            return info_left.union(info_right)
        if isinstance(node.op, ast.Mult):
            return info_left.intersection(info_right)
        if isinstance(node.op, ast.Sub):
            return (
                info_left.difference(info_right)
            )
        else:
            return self.generic_visit(node)

    def visit_UnaryOp(self, node):
        query_info = self.visit(node.operand)
        if isinstance(node.op, ast.Invert):
            return FiberQueryInfo(
                set(
                    tract for tract in query_info.tracts
                    if self.tractography_spatial_indexing.crossing_tracts_labels[tract].issubset(query_info.labels)
                ),
                query_info.labels
            )
        elif isinstance(node.op, ast.UAdd):
            return query_info
        elif isinstance(node.op, ast.USub) or isinstance(node.op, ast.Not):
            all_labels = set(self.tractography_spatial_indexing.crossing_labels_tracts.keys())
            all_labels.difference_update(query_info.labels)
            all_tracts = set().union(*tuple(
                (self.tractography_spatial_indexing.crossing_labels_tracts[label] for label in all_labels)
            ))

            new_info = FiberQueryInfo(all_tracts, all_labels)
            return new_info
        else:
            raise TractQuerierSyntaxError(
                "Syntax error in query line %d" % node.lineno)

    def visit_Str(self, node):
        query_info = FiberQueryInfo()
        for name in fnmatch.filter(self.evaluated_queries_info.keys(), node.s):
            query_info.update(self.evaluated_queries_info[name])
        return query_info

    def visit_Call(self, node):
        # Single string argument function
        if (
            isinstance(node.func, ast.Name) and
            len(node.args) == 1 and
            len(node.args) == 1 and
            node.starargs is None and
            node.keywords == [] and
            node.kwargs is None
        ):
            if (node.func.id.lower() == 'only'):
                query_info = self.visit(node.args[0])

                only_tracts = set(
                    tract for tract in query_info.tracts
                    if self.tractography_spatial_indexing.crossing_tracts_labels[tract].issubset(query_info.labels)
                )
                only_endpoints = tuple((
                    set(
                        tract for tract in query_info.tracts_endpoints[i]
                        if self.tractography_spatial_indexing.ending_tracts_labels[i][tract] in query_info.labels
                    )
                    for i in (0, 1)
                ))
                return FiberQueryInfo(
                    only_tracts,
                    query_info.labels,
                    only_endpoints
                )
            elif (node.func.id.lower() == 'endpoints_in'):
                query_info = self.visit(node.args[0])
                new_tracts = query_info.tracts_endpoints[0].union(query_info.tracts_endpoints[1])

                # tracts = set().union(set(
                #    tract for tract in query_info.tracts
                #    if (
                #        self.tractography_spatial_indexing.ending_tracts_labels[i][tract] in query_info.labels
                #    )
                #))

                # labels = set().union(
                #    *tuple((self.tractography_spatial_indexing.crossing_tracts_labels[tract] for tract in tracts))
                #)
                return FiberQueryInfo(new_tracts, query_info.labels, query_info.tracts_endpoints)
            elif (node.func.id.lower() == 'both_endpoints_in'):
                query_info = self.visit(node.args[0])
                new_tracts = (
                    query_info.tracts_endpoints[0].intersection(query_info.tracts_endpoints[1])
                )
                return FiberQueryInfo(new_tracts, query_info.labels, query_info.tracts_endpoints)
            elif (node.func.id.lower() == 'save' and isinstance(node.args, ast.Str)):
                self.queries_to_save.add(node.args[0].s)
                return
            elif node.func.id.lower() in self.relative_terms:
                return self.process_relative_term(node)

        raise TractQuerierSyntaxError("Invalid query in line %d" % node.lineno)

    def process_relative_term(self, node):
        r"""
        Processes the relative terms

        * anterior_of
        * posterior_of
        * superior_of
        * inferior_of
        * medial_of
        * lateral_of

        Parameters
        ----------
        node :  :py:class:`ast.Node`
            Parsed tree


        Returns
        -------

        tracts, labels

        tracts :  set
            Numbers of the tracts that result of this
            query

        labels :  set
            Numbers of the labels that are traversed by
            the tracts resulting from this query
        """
        if len(self.tractography_spatial_indexing.label_bounding_boxes) == 0:
            return FiberQueryInfo()

        arg = node.args[0]
        if isinstance(arg, ast.Name):
            query_info = self.visit(arg)
        elif isinstance(arg, ast.Attribute):
            if arg.attr.lower() in ('left', 'right'):
                side = arg.attr.lower()
                query_info = self.visit(arg)
        else:
            raise TractQuerierSyntaxError(
                "Attribute not recognized for relative specification."
                "Line %d" % node.lineno
            )

        labels = query_info.labels

        labels_generator = (l for l in labels)
        bounding_box = self.tractography_spatial_indexing.label_bounding_boxes[labels_generator.next()]
        for label in labels_generator:
            bounding_box = bounding_box.union(self.tractography_spatial_indexing.label_bounding_boxes[label])

        function_name = node.func.id.lower()

        name = function_name.replace('_of', '')

        if (
            name in ('anterior', 'inferior') or
            name == 'medial' and side == 'left' or
            name == 'lateral' and side == 'right'
        ):
            operator = gt
        else:
            operator = lt

        if name == 'medial':
            if side == 'left':
                name = 'right'
            else:
                name = 'left'
        elif name == 'lateral':
            if side == 'left':
                name = 'left'
            else:
                name = 'right'

        tract_bounding_box_coordinate =\
            self.tractography_spatial_indexing.tract_bounding_boxes[name]

        tract_endpoints_pos = self.tractography_spatial_indexing.tract_endpoints_pos

        bounding_box_coordinate = getattr(bounding_box, name)

        if name in ('left', 'right'):
            column = 0
        elif name in ('anterior', 'posterior'):
            column = 1
        elif name in ('superior', 'inferior'):
            column = 2

        tracts = set(
            operator(tract_bounding_box_coordinate, bounding_box_coordinate).nonzero()[0]
        )

        endpoints = tuple((
            set(
                operator(
                    tract_endpoints_pos[:, i, column],
                    bounding_box_coordinate
                ).nonzero()[0]
            )
            for i in (0, 1)
        ))

        labels = set().union(*tuple((
            self.tractography_spatial_indexing.crossing_tracts_labels[tract]
            for tract in tracts
        )))

        return FiberQueryInfo(tracts, labels, endpoints)

    def visit_Assign(self, node):
        if len(node.targets) > 1:
            raise TractQuerierSyntaxError(
                "Invalid assignment in line %d" % node.lineno)

        queries_to_evaluate = self.process_assignment(node)

        for query_name, value_node in queries_to_evaluate.items():
            self.queries_to_save.add(query_name)
            self.evaluated_queries_info[query_name] = self.visit(value_node)

    def visit_AugAssign(self, node):
        if not isinstance(node.op, ast.BitOr):
            raise TractQuerierSyntaxError(
                "Invalid assignment in line %d" % node.lineno)

        queries_to_evaluate = self.process_assignment(node)

        for query_name, value_node in queries_to_evaluate.items():
            query_info = self.visit(value_node)
            self.evaluated_queries_info[query_name] = query_info

    def process_assignment(self, node):
        r"""
        Processes the assignment operations

        Parameters
        ----------
        node :  :py:class:`ast.Node`
            Parsed tree


        Returns
        -------

        queries_to_evaluate: dict
            A dictionary or pairs '<name of the query>'= <node to evaluate>

        """
        queries_to_evaluate = {}
        if 'target' in node._fields:
            target = node.target
        if 'targets' in node._fields:
            target = node.targets[0]

        if isinstance(target, ast.Name):
            queries_to_evaluate[target.id] = node.value
        elif (
            isinstance(target, ast.Attribute) and
            target.attr == 'side'
        ):
            node_left, node_right = self.rewrite_side_query(node)
            self.visit(node_left)
            self.visit(node_right)
        elif (
            isinstance(target, ast.Attribute) and
            isinstance(target.value, ast.Name)
        ):
            queries_to_evaluate[
                target.value.id.lower() + '.' + target.attr.lower()] = node.value
        else:
            raise TractQuerierSyntaxError(
                "Invalid assignment in line %d" % node.lineno)
        return queries_to_evaluate

    def rewrite_side_query(self, node):
        r"""
        Processes the side suffixes in a query

        Parameters
        ----------
        node :  :py:class:`ast.Node`
            Parsed tree


        Returns
        -------

        node_left, node_right: nodes
            two AST nodes, one for the query instantiated on the left hemisphere
            one for the query instantiated on the right hemisphere

        """
        node_left = deepcopy(node)
        node_right = deepcopy(node)

        for node_ in ast.walk(node_left):
            if isinstance(node_, ast.Attribute):
                if node_.attr == 'side':
                    node_.attr = 'left'
                elif node_.attr == 'opposite':
                    node_.attr = 'right'

        for node_ in ast.walk(node_right):
            if isinstance(node_, ast.Attribute):
                if node_.attr == 'side':
                    node_.attr = 'right'
                elif node_.attr == 'opposite':
                    node_.attr = 'left'

        return node_left, node_right

    def visit_Name(self, node):
        if node.id in self.evaluated_queries_info:
            return self.evaluated_queries_info[node.id]
        else:
            raise TractQuerierSyntaxError(
                "Invalid query name in line %d: %s" % (node.lineno, node.id))

    def visit_Attribute(self, node):
        if not isinstance(node.value, ast.Name):
            raise TractQuerierSyntaxError(
                "Invalid query in line %d: %s" % node.lineno)

        query_name = node.value.id + '.' + node.attr
        if query_name in self.evaluated_queries_info:
            return self.evaluated_queries_info[query_name]
        else:
            raise TractQuerierSyntaxError(
                "Invalid query name in line %d: %s" % (node.lineno, query_name))

    def visit_Num(self, node):
        if node.n in self.tractography_spatial_indexing.crossing_labels_tracts:
            tracts = self.tractography_spatial_indexing.crossing_labels_tracts[node.n]
        else:
            tracts = set()

        endpoints = (set(), set())
        for i in (0, 1):
            elt = self.tractography_spatial_indexing.ending_labels_tracts[i]
            if node.n in elt:
                endpoints[i].update(elt[node.n])

        labelset = set((node.n,))
        tract_info = FiberQueryInfo(
            tracts, labelset,
            endpoints
        )

        return tract_info

    def visit_Expr(self, node):
        if isinstance(node.value, ast.Name):
            if node.value.id in self.evaluated_queries_info.keys():
                self.queries_to_save.add(node.value.id)
            else:
                raise TractQuerierSyntaxError(
                    "Query %s not known line: %d" % (node.value.id, node.lineno))
        elif isinstance(node.value, ast.Module):
            self.visit(node.value)
        else:
            raise TractQuerierSyntaxError(
                "Invalid expression at line: %d" % (node.lineno))

    def generic_visit(self, node):
        raise TractQuerierSyntaxError(
            "Invalid Operation %s line: %d" % (type(node), node.lineno))

    def visit_For(self, node):
        id_to_replace = node.target.id.lower()

        iter_ = node.iter
        if isinstance(iter_, ast.Str):
            list_items = fnmatch.filter(
                self.evaluated_queries_info.keys(), iter_.s.lower())
        elif isinstance(iter_, ast.List):
            list_items = []
            for item in iter_.elts:
                if isinstance(item, ast.Name):
                    list_items.append(item.id.lower())
                else:
                    raise TractQuerierSyntaxError(
                        'Error in FOR statement in line %d, elements in the list must be query names' % node.lineno)

        original_body = ast.Module(body=node.body)

        for item in list_items:
            aux_body = deepcopy(original_body)
            for node_ in ast.walk(aux_body):
                if isinstance(node_, ast.Name) and node_.id.lower() == id_to_replace:
                    node_.id = item

            self.visit(aux_body)


class TractQuerierSyntaxError(ValueError):

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class RewriteChangeNotInPrescedence(ast.NodeTransformer):

    def visit_BoolOp(self, node):
        predicate = lambda value: not (
            isinstance(value, ast.Compare) and
            isinstance(value.ops[0], ast.NotIn)
        )

        values_which_are_not_in_op = [value for value in takewhile(
            predicate,
            node.values[1:]
        )]

        if (len(values_which_are_not_in_op) == len(node.values) - 1):
            return node

        old_CompareNode = node.values[len(values_which_are_not_in_op) + 1]
        new_CompareNodeLeft = ast.copy_location(
            ast.BoolOp(
                op=node.op,
                values=(
                    [node.values[0]] +
                    values_which_are_not_in_op +
                    [old_CompareNode.left]
                )
            ),
            node
        )

        new_CompareNode = ast.copy_location(
            ast.Compare(
                left=new_CompareNodeLeft,
                ops=old_CompareNode.ops,
                comparators=old_CompareNode.comparators
            ),
            node
        )

        rest_of_the_values = node.values[len(values_which_are_not_in_op) + 2:]

        if len(rest_of_the_values) == 0:
            return self.visit(new_CompareNode)
        else:
            return self.visit(ast.copy_location(
                ast.BoolOp(
                    op=node.op,
                    values=(
                        [new_CompareNode] +
                        rest_of_the_values
                    )
                ),
                node
            ))


class RewritePreprocess(ast.NodeTransformer):

    def __init__(self, *args, **kwargs):
        if 'include_folders' in kwargs:
            self.include_folders = kwargs['include_folders']
            kwargs['include_folders'] = None
            del kwargs['include_folders']
        else:
            self.include_folders = ['.']
        super(RewritePreprocess, self).__init__(*args, **kwargs)

    def visit_Attribute(self, node):
        return ast.copy_location(
            ast.Attribute(
                value=self.visit(node.value),
                attr=node.attr.lower()
            ),
            node
        )

    def visit_Name(self, node):
        return ast.copy_location(
            ast.Name(id=node.id.lower()),
            node
        )

    def visit_Str(self, node):
        return ast.copy_location(
            ast.Str(s=node.s.lower()),
            node
        )

    def visit_Import(self, node):
        try:
            module_names = []
            for module_name in node.names:
                file_name = module_name.name
                found = False
                for folder in self.include_folders:
                    file_ = path.join(folder, file_name)
                    if path.exists(file_) and path.isfile(file_):
                        module_names.append(file_)
                        found = True
                        break
                if not found:
                    raise TractQuerierSyntaxError(
                        'Imported file not found: %s' % file_name
                    )
            imported_modules = [
                ast.parse(file(module_name).read(), filename=module_name)
                for module_name in module_names
            ]
        except SyntaxError:
            import sys
            import traceback
            exc_type, exc_value, exc_traceback = sys.exc_info()
            formatted_lines = traceback.format_exc().splitlines()
            raise TractQuerierSyntaxError(
                'syntax error in line %s line %d: \n%s\n%s' %
                (
                    module_name,
                    exc_value[1][1],
                    formatted_lines[-3],
                    formatted_lines[-2]
                )
            )

        new_node = ast.Module(imported_modules)

        return ast.copy_location(
            self.visit(new_node),
            node
        )


def queries_preprocess(query_file, filename='<unknown>', include_folders=[]):

    try:
        query_file_module = ast.parse(query_file, filename='<unknown>')
    except SyntaxError:
        import sys
        import traceback
        exc_type, exc_value, exc_traceback = sys.exc_info()
        formatted_lines = traceback.format_exc().splitlines()
        raise TractQuerierSyntaxError(
            'syntax error in line %s line %d: \n%s\n%s' %
            (
                filename,
                exc_value[1][1],
                formatted_lines[-3],
                formatted_lines[-2]
            )
        )

    rewrite_preprocess = RewritePreprocess(include_folders=include_folders)
    rewrite_precedence_not_in = RewriteChangeNotInPrescedence()

    preprocessed_module = rewrite_precedence_not_in.visit(
        rewrite_preprocess.visit(query_file_module)
    )

    return preprocessed_module.body


def eval_queries(
    query_file_body,
    tractography_spatial_indexing
):
    eq = EvaluateQueries(tractography_spatial_indexing)

    if isinstance(query_file_body, list):
        eq.visit(ast.Module(query_file_body))
    else:
        eq.visit(query_file_body)

    return dict([(key, eq.evaluated_queries_info[key].tracts) for key in eq.queries_to_save])


def queries_syntax_check(query_file_body):
    class DummySpatialIndexing:

        def __init__(self):
            self.crossing_tracts_labels = {}
            self.crossing_labels_tracts = {}
            self.ending_tracts_labels = ({}, {})
            self.ending_labels_tracts = ({}, {})
            self.label_bounding_boxes = {}
            self.tract_bounding_boxes = {}

    eval_queries(query_file_body, DummySpatialIndexing())


def labels_for_tracts(crossing_tracts_labels):
    crossing_labels_tracts = {}
    for i, f in crossing_tracts_labels.items():
        for l in f:
            if l in crossing_labels_tracts:
                crossing_labels_tracts[l].add(i)
            else:
                crossing_labels_tracts[l] = set((i,))
    return crossing_labels_tracts
