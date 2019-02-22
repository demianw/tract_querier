import ast
import cmd
import fnmatch
from .query_processor import (
    EvaluateQueries, queries_preprocess,
    TractQuerierSyntaxError, TractQuerierLabelNotFound,
    keywords
)


def safe_method(f):
    """exception handling decorator"""
    def newfunc(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except:
            import traceback
            import sys
            sys.stderr.write(
                "Uncaught exception, please contact the development team\n"
            )
            traceback.print_exc()
    return newfunc


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
                    self.querier.evaluated_queries_info[query_name].tracts
                )
            elif (
                isinstance(target, ast.Attribute) and
                isinstance(target.value, ast.Name)
            ):

                self.save_attribute_name(target)

    def visit_Expr(self, node):
        value = node.value
        if isinstance(value, ast.Name):
            query_name = node.value.id.lower()
            self.save_query_callback(
                query_name,
                self.querier.evaluated_queries_info[query_name].tracts
            )
        elif (
            isinstance(value, ast.Attribute) and
            isinstance(value.value, ast.Name)
        ):
            self.save_attribute_name(node.value)
        else:
            self.visit(value)

    def save_attribute_name(self, node):
        query_prefix = node.value.id.lower()
        query_suffix = node.attr.lower()
        if query_suffix == 'side':
            for suffix in ('left', 'right'):
                query_name = query_prefix + '.' + suffix
                self.save_query_callback(
                    query_name,
                    self.querier.evaluated_queries_info[
                        query_name
                    ].tracts
                )
        else:
            query_name = query_prefix + '.' + query_suffix
            self.save_query_callback(
                query_name,
                self.querier.evaluated_queries_info[
                    query_name
                ].tracts
            )

    def visit_Module(self, node):
        for line in node.body:
            self.visit(line)

    def visit_For(self, node):
        # If it is a for loop, only execute
        pass


class TractQuerierCmd(cmd.Cmd):

    def __init__(
            self,
            tractography_spatial_indexing,
            initial_body=None, tractography=None,
            save_query_callback=None,
            include_folders=['.']
    ):
        cmd.Cmd.__init__(self, 'Tab')
        self.prompt = '[wmql] '
        self.include_folders = include_folders
        self.tractography = tractography
        self.querier = EvaluateQueries(tractography_spatial_indexing)

        self.save_query_callback = save_query_callback
        self.save_query_visitor = SaveQueries(
            self.save_query_callback, self.querier
        )

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

    @safe_method
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
            print(k)

    @safe_method
    def do_save(self, line):
        try:
            body = queries_preprocess(
                line,
                filename='shell', include_folders=self.include_folders
            )
            self.save_query_visitor.visit(ast.Module(body=body))
        except SyntaxError as e:
            print(e.value)
        except TractQuerierSyntaxError as e:
            print(e.value)
        except TractQuerierLabelNotFound as e:
            print(e.value)
        except KeyError as e:
            print("Query name not recognized: %s" % e)

        return False

    def do_help(self, line):
        print('''WMQL Help

        Commands:
            dir <pattern>: list the available queries according to the pattern
            save <query name>: save the corresponding query

        Expressions:
            <query name> = <query>: execute a query and save its result
            <query name> |= <query>: execute a query without saving its result

        Exit pressing Ctrl+D
        ''')
        return

    def emptyline(self):
        return

    @safe_method
    def default(self, line):
        print(line)
        if len(line) == 0:
            return False

        try:
            body = queries_preprocess(
                line,
                filename='shell', include_folders=self.include_folders
            )
            body = ast.Module(body=body)
            self.querier.visit(body)
            self.save_query_visitor.visit(body)
        except SyntaxError as e:
            print(e.value)
        except TractQuerierSyntaxError as e:
            print(e.value)
        except TractQuerierLabelNotFound as e:
            print(e.value)

        return False

    @safe_method
    def names(self):
        names = []
        for query in self.querier.evaluated_queries_info.keys():
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

    @safe_method
    def completenames(self, text, *ignored):
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

    def completedefault(self, text, *ignored):
        return self.completenames(text, *ignored)

    def do_EOF(self, line):
        s = input("\nSure you want to leave (y/n)? ")
        if s.lower() == 'y':
            return True
        else:
            return False
