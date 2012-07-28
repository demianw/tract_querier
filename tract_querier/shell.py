import ast
import cmd
import fnmatch
from query_processor import EvaluateQueries, queries_preprocess, TractQuerierSyntaxError, keywords


class SaveQueries(ast.NodeVisitor):
    def __init__(self, save_query_callback, querier):
        self.save_query_callback = save_query_callback
        self.querier = querier

    def visit_AugAssign(self, node):
        pass


    def visit_Assign(self, node):
        for target in node.targets:
            if isinstance(target, ast.Name):
                query_name = target.id.lower()
                self.save_query_callback(
                    query_name,
                    self.querier.evaluated_queries_fibers[query_name]
                )
            elif (
                isinstance(target, ast.Attribute) and
                isinstance(target.value, ast.Name)
            ):
                query_name = (
                    target.value.id.lower() +
                    '.' +
                    target.attr.lower()
                )
                self.save_query_callback(
                    query_name,
                    self.querier.evaluated_queries_fibers[query_name]
                )



    def visit_Expr(self, node):
        value = node.value
        if isinstance(value, ast.Name):
            query_name = node.value.id.lower()
            self.save_query_callback(
                query_name,
                self.querier.evaluated_queries_fibers[query_name]
            )
        elif (
            isinstance(node, ast.Attribute) and
            isinstance(node.value, ast.Name)
        ):
            query_name = (
                node.value.id.lower() +
                '.' +
                node.attr.lower()
            )
            self.save_query_callback(
                query_name,
                self.querier.evaluated_queries_fibers[query_name]
            )
        else:
            self.visit(value)


    def visit_Module(self, node):
        for line in node.body:
            self.visit(line)


class TractQuerierCmd(cmd.Cmd):
    def __init__(self,
                 crossing_fibers_labels, crossing_labels_fibers,
                 ending_fibers_labels, ending_labels_fibers,
                 initial_body=None, tractography=None, save_query_callback=None,
                 include_folders=['.']
                ):
        cmd.Cmd.__init__(self, 'Tab')
        self.prompt = '[wmql] '
        self.include_folders = include_folders
        self.tractography = tractography
        self.querier = EvaluateQueries(
            crossing_fibers_labels, crossing_labels_fibers,
            ending_fibers_labels, ending_labels_fibers
        )
        self.save_query_callback = save_query_callback
        self.save_query_visitor = SaveQueries(self.save_query_callback, self.querier)

        if initial_body is not None:
            if isinstance(initial_body, str):
                initial_body = queries_preprocess(
                    initial_body,
                    filename='Shell', include_folders=self.include_folders
                )

            if isinstance(initial_body, list):
                self.querier.visit(ast.Module(initial_body))
            else:
                self.querier.visit(initial_body)


    def do_dir(self, patterns):
        if patterns == '':
            patterns = '*'
        patterns = patterns.split(' ')
        k = self.names()
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
            body = queries_preprocess(
                line,
                filename='shell', include_folders=self.include_folders
            )
            body = ast.Module(body=body)
            self.querier.visit(body)
            self.save_query_visitor.visit(body)
        except SyntaxError, e:
            print e.value
        except TractQuerierSyntaxError, e:
            print e.value

        return False

    def names(self):
        names = []
        for query in self.querier.evaluated_queries_fibers.keys():
            if query.endswith('_left'):
                names += [
                        query.replace('_left', '.left'),
                        query.replace('_left', '.right'),
                ]
            elif query.endswith('_right'):
                pass
            else:
                names.append(query)
        return names

    def completenames(self, text, *ignored):
        try:
            candidates = sum(
                ([
                    query.replace('.left', '.side'),
                    query.replace('.left', '.opposite'),
                ] for query in self.names() if query.endswith('.left')),
                self.names()
            )

            if '=' in text:
                candidates += keywords

            options = [
                candidate
                for candidate in candidates
                if candidate.startswith(text)
            ]

            return options
        except Exception, e:
            print repr(e)


    def completedefault(self, text, *ignored):
        return self.completenames(text, *ignored)

    def do_EOF(self, line):
        s = raw_input("\nSure you want to leave (y/n)? ")
        if s.lower() =='y':
            return True
        else:
            return False
