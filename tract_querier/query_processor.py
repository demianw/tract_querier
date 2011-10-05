import ast
from copy import deepcopy

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
            ast.parse(file(module_name.name).read()) 
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
    def __init__(self, labels):
        self.labels = labels

    def visit_BinOp(self, node):
        self.visit(node.left)
        self.visit(node.right)
        self.visit(node.op)
    def visit_UnaryOp(self, node):
        self.visit(node.value)
        self.visit(node.op)
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
    def visit_Str(self, node):
        pass
    def visit_Num(self, node):
        if node.n not in self.labels:
            raise SyntaxError("Number not in labels line: %d" % node.lineno)
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


def queries_preprocess(query_file):
   
    query_file_module = ast.parse(query_file)

    rewrite_preprocess = RewritePreprocess()

    preprocessed_module = rewrite_preprocess.visit(query_file_module)

    return preprocessed_module.body

def queries_syntax_check(query_file_body, labels, check_for_number_existence=False):
    oq = ObtainQueries(labels, check_for_number_existence=check_for_number_existence)
    oq.visit(ast.Module(query_file_body))
    return oq.queries()

def eval_queries(labels_fibers, queries, evaluated_queries = {}):
    for query_name, query, compute_query in queries:
        evaluated_queries[query_name] = (eval_query(labels_fibers, query, evaluated_queries), compute_query)
    
    return evaluated_queries

    


def eval_query(labels_fibers, query, evaluated_queries={}):
    import re

    try:
        eval_query_ = lambda query: eval_query(labels_fibers, query, evaluated_queries)
        if isinstance(query, ast.Num):
            if query.n in labels_fibers:
                return labels_fibers[query.n]
            else:
                return set()
        elif isinstance(query, ast.UnaryOp):
            if query.operand.n in labels_fibers:
                return labels_fibers[query.operand.n]
            else:
                return set()
        elif isinstance(query, ast.BinOp):
            if isinstance(query.op, ast.Add):
                return eval_query_(query.left).union(eval_query_(query.right))
            elif isinstance(query.op, ast.Mult):
                return eval_query_(query.left).intersection(eval_query_(query.right))
            elif isinstance(query.op, ast.Sub):
                return eval_query_(query.left).difference(eval_query_(query.right))
            else:
                parser.error("Syntax error in query")
        elif isinstance(query, ast.Name):
            return evaluated_queries[query.id][0]
        elif isinstance(query, ast.Str):
            matching_queries = tuple((
                fibers[0] for name, fibers in evaluated_queries.items() 
                if re.match(query.s, name)
            ))
            return set.union( *matching_queries )
        else:
            parser.error("Syntax error in query")
    except IndexError:
        parser.error("Invalid label number")


def save_tractography_file(filename, tractography, fiber_numbers):
    from tractographyGP import vtkInterface

    original_fibers = tractography.getOriginalFibers()

    fibers_to_save = [original_fibers[i] for i in fiber_numbers]

    fibers_data_to_save = {}
    for key, data in tractography.getOriginalFibersData().items():
        fibers_data_to_save[key] = [data[f] for f in fiber_numbers]

    if 'tensors' in fibers_data_to_save:
        fibers_data_to_save['ActiveTensors'] = 'tensors'
    if 'vectors' in fibers_data_to_save:
        fibers_data_to_save['ActiveVectors'] = 'vectors'

    vtkInterface.writeLinesToVtkPolyData(
        filename,
        fibers_to_save,
        fibers_data_to_save
    )



