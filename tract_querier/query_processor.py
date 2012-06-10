import ast
from itertools import izip
from copy import deepcopy
import re

class RewriteSide(ast.NodeTransformer):
    def __init__(self, left=True):
        if left:
            self.side = 'left'
            self.opposite = 'right'
        else:
            self.side = 'right'
            self.opposite = 'left'

    def visit_Attribute(self, node):
        if node.attr == 'side':
            return ast.copy_location(
                ast.Name(id=node.value.id + '_' + self.side),
                node
            )
        elif node.attr == 'opposite':
            return ast.copy_location(
                ast.Name(id=node.value.id + '_' + self.opposite),
                node
            )
        else:

            raise SyntaxError("Invalid subscript: " + ast.dump(node))

class RewritePreprocess(ast.NodeTransformer):
    rewrite_left = RewriteSide(left=True)
    rewrite_right = RewriteSide(left=False)

    def visit_Assign(self, node):
        if isinstance(node.targets[0], ast.Attribute):
            if node.targets[0].attr != 'side':
                raise SyntaxError("Wrong attribute")

            node_left_right = ast.Module([
                self.rewrite_left.visit(deepcopy(node)),
                self.rewrite_right.visit(deepcopy(node))
            ])

            return ast.copy_location(
                self.visit(node_left_right),
                node
            )
        else:
            return node


    def visit_AugAssign(self, node):
        if isinstance(node.target, ast.Attribute):
            if node.target.attr != 'side':
                raise SyntaxError("Wrong attribute")

            node_left_right = ast.Module([
                self.rewrite_left.visit(deepcopy(node)),
                self.rewrite_right.visit(deepcopy(node))
            ])

            return ast.copy_location(
                self.visit(node_left_right),
                node
            )
        else:
            return node

    def visit_Name(self, node):
        return ast.copy_location(
            ast.Name(id=node.id.lower()),
            node
        )

    def visit_Import(self, node):
        imported_modules = [
            ast.parse(file(module_name.name).read(), filename=module_name.name)
            for module_name in node.names
        ]

        new_node = ast.Module(imported_modules)

        return ast.copy_location(
            self.visit(new_node),
            node
        )


class ValidateQuery(ast.NodeVisitor):
    def __init__(self, labels, check_for_number_existence=False):
        self.labels = labels
        self.check_for_number_existence = check_for_number_existence

    def visit_BinOp(self, node):
        self.visit(node.left)
        self.visit(node.right)
        self.visit(node.op)
    def visit_UnaryOp(self, node):
        self.visit(node.operand)
        self.visit(node.op)
    def visit_Invert(self, node):
        pass
    def visit_Sum(self, node):
        pass
    def visit_Mult(self, node):
        pass
    def visit_Sub(self, node):
        pass
    def visit_Add(self, node):
        pass
    def visit_Name(self, node):
        pass
    def visit_UAdd(self, node):
        pass
    def visit_USub(self, node):
        pass
    def visit_Str(self, node):
        pass
    def visit_Num(self, node):
        if node.n not in self.labels:
            if self.check_for_number_existence:
                raise SyntaxError("Number not in labels line: %d" % node.lineno)
    def generic_visit(self, node):
        if not hasattr(node, 'lineno'):
            node.lineno = -1
        raise SyntaxError("Invalid Operation %s line: %d" % (type(node), node.lineno) )


class EvaluateQueries(ast.NodeVisitor):
    def __init__(self, fibers_labels, labels_fibers):
        self.fibers_labels = fibers_labels
        self.labels_fibers = labels_fibers
        self.evaluated_queries_fibers = {}
        self.evaluated_queries_labels = {}
        self.queries_to_save = set()

    def visit_Module(self, node):
        for line in node.body:
            self.visit(line)

    def visit_Compare(self, node):
        if any(not isinstance(op, ast.NotIn) for op in node.ops):
            raise SyntaxError("Invalid syntax in query line %d" % node.lineno)

        fibers, labels = self.visit(node.left)
        fibers = fibers.copy()
        labels = labels.copy()
        for value in node.comparators:
            fibers_, labels_ = self.visit(value)
            fibers.difference_update(fibers_)
            labels.difference_update(labels_)

        return fibers, labels

    def visit_BoolOp(self, node):
        fibers, labels = self.visit(node.values[0])
        fibers = fibers.copy()
        labels = labels.copy()

        if isinstance(node.op, ast.Or):
            for value in node.values[1:]:
                fibers_, labels_ = self.visit(value)
                fibers.update(fibers_)
                labels.update(labels_)

        elif isinstance(node.op, ast.And):
            for value in node.values[1:]:
                fibers_, labels_ = self.visit(value)
                fibers.intersection_update(fibers_)
                labels.update(labels_)

        else:
            return self.generic_visit(node)

        return fibers, labels

    def visit_BinOp(self, node):
        fibers_left, label_left = self.visit(node.left)
        fibers_right,label_right = self.visit(node.right)
        if isinstance(node.op, ast.Add):
            return fibers_left.union(fibers_right), label_left.union(label_right)
        if isinstance(node.op, ast.Mult):
            return fibers_left.intersection(fibers_right), label_left.union(label_right)
        if isinstance(node.op, ast.Sub):
            return fibers_left.difference(fibers_right), label_left.difference(label_right)
        else:
            return self.generic_visit(node)

    def visit_UnaryOp(self, node):
        fibers, labels = self.visit(node.operand)
        if isinstance(node.op, ast.Invert):
            return set(fiber for fiber in fibers if self.fibers_labels[fiber].issubset(labels)), labels
        elif isinstance(node.op, ast.UAdd):
            return fibers, labels
        elif isinstance(node.op, ast.USub) or isinstance(node.op, ast.Not):
            all_labels = set(self.labels_fibers.keys())
            all_labels.difference_update(labels)
            fibers = set().union(*tuple((self.labels_fibers[label] for label in all_labels)))
            return fibers, all_labels
        else:
            raise SyntaxError("Syntax error in query line %d" % node.lineno)

    def visit_Str(self, node):
        matching_fibers = set()
        matching_labels = set()
        for name in self.evaluated_queries_fibers.keys():
            if re.match(node.s, name):
                matching_fibers.update(self.evaluated_queries_fibers[name])
                matching_labels.update(self.evaluated_queries_labels[name])

        return matching_fibers, matching_labels

    def visit_Call(self, node):
        if (
            isinstance(node.func, ast.Name) and
            node.func.id.lower() == 'only' and
            len(node.args) == 1 and
            node.starargs is None and
            node.keywords == [] and
            node.kwargs is None
        ):
            fibers, labels = self.visit(node.args[0])
            return set(fiber for fiber in fibers if self.fibers_labels[fiber].issubset(labels)), labels
        else:
            raise SyntaxError("Invalid query in line %d" %node.lineno)

    def visit_Assign(self, node):
        if len(node.targets) > 1 or not isinstance(node.targets[0], ast.Name):
            raise SyntaxError("Invalid assignment in line %d" % node.lineno)
        fibers, labels = self.visit(node.value)
        self.queries_to_save.add(node.targets[0].id)
        self.evaluated_queries_fibers[node.targets[0].id] = fibers
        self.evaluated_queries_labels[node.targets[0].id] = labels

    def visit_AugAssign(self, node):
        if not isinstance(node.op, ast.BitOr) or not isinstance(node.target, ast.Name):
            raise SyntaxError("Invalid assignment in line %d" % node.lineno)
        fibers, labels = self.visit(node.value)
        self.evaluated_queries_fibers[node.target.id] = fibers
        self.evaluated_queries_labels[node.target.id] = labels

    def visit_Name(self, node):
        if node.id in self.evaluated_queries_fibers:
            return self.evaluated_queries_fibers[node.id], self.evaluated_queries_labels[node.id]
        else:
            raise SyntaxError("Invalid query name in line %d" % node.lineno)

    def visit_Num(self, node):
        if node.n in self.labels_fibers:
            fibers = self.labels_fibers[node.n]
        else:
            fibers = set()
        return fibers, set((node.n,))

    def generic_visit(self, node):
        raise SyntaxError("Invalid Operation %s line: %d" % (type(node), node.lineno) )




class ObtainQueries(ast.NodeVisitor):
    def __init__(self, labels, check_for_number_existence=False):
        self._queries = []
        self.validate_query = ValidateQuery(labels, check_for_number_existence=check_for_number_existence)

    def visit_Assign(self, node):
        target = node.targets[0]
        self.append_query(target, node.value, True)

    def visit_AugAssign(self, node):
        if not isinstance(node.op, ast.BitOr):
            raise SyntaxError("Wrong assignment operator line %d" % node.lineno)

        target = node.target
        self.append_query(target, node.value, False)

    def visit_Module(self, node):
        for node_ in node.body:
            self.visit(node_)

    def generic_visit(self, node):
        raise SyntaxError("Invalid operation")


    def append_query(self, target, value, compute):
        if not isinstance(target, ast.Name):
            raise SyntaxError("Wrong left side of assignment")

        try:
            self.validate_query.visit(value)
        except SyntaxError, e:
            print e
            raise SyntaxError("Problem with query %s" % target.id)

        self._queries.append((
            target.id,
            value,
            compute
        ))


    def queries(self):
        return self._queries


def queries_preprocess(query_file, filename='<unknown>'):

    query_file_module = ast.parse(query_file, filename='<unknown>')

    rewrite_preprocess = RewritePreprocess()

    preprocessed_module = rewrite_preprocess.visit(query_file_module)

    return preprocessed_module.body

def queries_syntax_check(query_file_body, labels, check_for_number_existence=False):
    oq = ObtainQueries(labels, check_for_number_existence=check_for_number_existence)
    oq.visit(ast.Module(query_file_body))
    return oq.queries()

def eval_queries(labels_fibers, fibers_labels, query_file_body):
    eq = EvaluateQueries(fibers_labels, labels_fibers)
    if isinstance(query_file_body, list):
        eq.visit(ast.Module(query_file_body))
    else:
        eq.visit(query_file_body)

    return dict([(key, eq.evaluated_queries_fibers[key]) for key in eq.queries_to_save])

def eval_queries_old(labels_fibers, fibers_labels, queries, evaluated_queries = {}, evaluated_queries_labels = {}):
    assert(set(evaluated_queries.keys()) == set(evaluated_queries_labels.keys()))
    for query_name, query, compute_query in queries:
        evaluated_query, evaluated_query_labels = eval_query(labels_fibers, fibers_labels, query, evaluated_queries, evaluated_queries_labels)
        evaluated_queries[query_name] = (evaluated_query, compute_query)
        evaluated_queries_labels[query_name] = evaluated_query_labels

    return evaluated_queries



def eval_query(labels_fibers, fibers_labels, query, evaluated_queries={}, evaluated_queries_labels={}):
    import re

    try:
        eval_query_ = lambda query: eval_query(labels_fibers, fibers_labels, query, evaluated_queries, evaluated_queries_labels)
        if isinstance(query, ast.Num):
            if query.n in labels_fibers:
                return labels_fibers[query.n], set([query.n])
            else:
                return set(), set()
        elif isinstance(query, ast.UnaryOp):
            query_fibers, query_labels = eval_query_(query.operand)
            if isinstance(query.op, ast.Invert):
                return set(fiber for fiber in query_fibers if fibers_labels[fiber].issubset(query_labels)), query_labels
            elif isinstance(query.op, ast.UAdd):
                return query_fibers, query_labels
            elif isinstance(query.op, ast.USub):
                all_fibers = set(fibers_labels.keys())
                all_labels = set(labels_fibers.keys())
                fibers = all_fibers.difference(query_fibers)
                labels = all_labels.difference(query_labels)
                return fibers, labels
            else:
                raise SyntaxError("Syntax error in query")
#            elif query.operand.n in labels_fibers:
#                return labels_fibers[query.operand.n], set([query.operand.n])
#            else:
#                return set(), set()
        elif isinstance(query, ast.BinOp):
            query_left, labels_left = eval_query_(query.left)
            query_right, labels_right = eval_query_(query.right)
            if isinstance(query.op, ast.Add):
                return query_left.union(query_right), labels_left.union(labels_right)
            elif isinstance(query.op, ast.Mult):
                return query_left.intersection(query_right), labels_left.intersection(labels_right)
            elif isinstance(query.op, ast.Sub):
                return query_left.difference(query_right), labels_left.difference(labels_right)
            else:
                raise SyntaxError("Syntax error in query")
        elif isinstance(query, ast.Name):
            return evaluated_queries[query.id][0], evaluated_queries_labels[query.id]
        elif isinstance(query, ast.Str):
            matching_queries = tuple((
                fibers[0] for name, fibers in evaluated_queries.items()
                if re.match(query.s, name)
            ))
            matching_labels = tuple((
                labels for name, labels in evaluated_queries_labels.items()
                if re.match(query.s, name)
            ))

            return set.union(*matching_queries), set.union(*matching_labels)
        else:
            raise SyntaxError("Syntax error in query")
    except IndexError:
        raise SyntaxError("Invalid label number")

def labels_for_fibers(fibers_labels):
    labels_fibers = {}
    for i, f in fibers_labels.items():
        for l in f:
            if l in labels_fibers:
                labels_fibers[l].add(i)
            else:
                labels_fibers[l] = set((i,))
    return labels_fibers
