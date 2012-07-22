import ast
import cmd
import fnmatch
from query_processor import EvaluateQueries, queries_preprocess, TractQuerierSyntaxError, keywords


class SaveQueries(ast.NodeVisitor):
    def __init__(self, save_query_callback, querier):
        self.save_query_callback = save_query_callback
        self.querier = querier

    def visit_Assign(self, node):
        for target in node.targets:
            ast.dump(target)
            if isinstance(target, ast.Name):
                query_name = target.id.lower()
                self.save_query_callback(
                    query_name,
                    self.querier.evaluated_queries_fibers[query_name]
                )

    def visit_Name(self, node):
            query_name = node.id.lower()
            self.save_query_callback(
                query_name,
                self.querier.evaluated_queries_fibers[query_name]
            )

    def visit_Expr(self, node):
        self.visit(node.value)

    def visit_Module(self, node):
        for line in node.body:
            self.visit(line)


class TractQuerierCmd(cmd.Cmd):
    def __init__(self,
                 crossing_fibers_labels, crossing_labels_fibers,
                 ending_fibers_labels, ending_labels_fibers,
                 initial_body=None, tractography=None, save_query_callback=None
                ):
        cmd.Cmd.__init__(self, 'Tab')
        self.prompt = '[wmql] '

        self.tractography = tractography
        self.querier = EvaluateQueries(
            crossing_fibers_labels, crossing_labels_fibers,
            ending_fibers_labels, ending_labels_fibers
        )
        self.save_query_callback = save_query_callback
        self.save_query_visitor = SaveQueries(self.save_query_callback, self.querier)

        if initial_body is not None:
            if isinstance(initial_body, str):
                initial_body = queries_preprocess(initial_body, filename='Shell')

            if isinstance(initial_body, list):
                self.querier.visit(ast.Module(initial_body))
            else:
                self.querier.visit(initial_body)


    def do_dir(self, patterns):
        if patterns == '':
            patterns = '*'
        patterns = patterns.split(' ')
        k = self.querier.evaluated_queries_fibers.keys()
        keys = []
        if len(patterns) > 0:
            for p in patterns:
                keys_found = fnmatch.filter(k, p)
                keys_found.sort()
                keys += keys_found
        else:
            keys = k
        for k in keys:
            print k


    def default(self, line):
        try:
            body = queries_preprocess(line, filename='shell')
            body = ast.Module(body=body)
            self.querier.visit(body)
            self.save_query_visitor.visit(body)
        except SyntaxError, e:
            print e.value
        except TractQuerierSyntaxError, e:
            print e.value

        return False

    def completenames(self, text, *ignored):
        candidates = sum([
                [
                    query.replace('_left', '.side'),
                    query.replace('_left', '.opposite')
                ]
                for query in
                self.querier.evaluated_queries_fibers.keys()
                if query.endswith('_left')
            ],
            self.querier.evaluated_queries_fibers.keys()
        ) + keywords
        options = [candidate for candidate in candidates if candidate.startswith(text)]
        return options

    def completedefault(self, text, *ignored):
        return self.completenames(text, *ignored)

    def do_EOF(self, line):
        s = raw_input("\nSure you want to leave (y/n)? ")
        if s.lower() =='y':
            return True
        else:
            return False
