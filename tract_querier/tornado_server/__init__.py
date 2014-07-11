import os
import json
from StringIO import StringIO
import sysconfig

import tornado.ioloop
import tornado.web
import tornado.gen
import tornado.ioloop
import tornado.websocket

websocket_clients = []


class WSHandler(tornado.websocket.WebSocketHandler):

    def initialize(self, shell=None):
        self.shell = shell
        self.sio = StringIO()
        self.shell.stdout = self.sio
        self.json_encoder = json.JSONEncoder()
        self.json_decoder = json.JSONDecoder()

    def open(self):
        global websocket_clients
        websocket_clients.append(self)

    def on_message(self, message):
        action = self.json_decoder.decode(message)
        if action['receiver'] == 'terminal':
            if action['action'] == 'cmd':
                self.sio.seek(0)
                self.shell.onecmd(action['command'])
                self.sio.seek(0)
                result = self.sio.getvalue()
                term_output = {
                    'receiver': 'terminal',
                    'output': result
                }
                self.write_message(self.json_encoder.encode(term_output))
                self.sio.truncate(0)

        #self.write_message(message + "Response")

    def on_close(self):
        global websocket_clients
        websocket_clients.remove(self)


class JSONRPCHandler(tornado.web.RequestHandler):
    def initialize(self, shell=None):
        self.shell = shell
        self.sio = StringIO()
        self.shell.stdout = self.sio
        self.json_encoder = json.JSONEncoder()
        self.json_decoder = json.JSONDecoder()

    def post(self):
        args = self.get_argument('data', 'No data received')
        decoded_args = self.json_decoder.decode(self.request.body)
        if decoded_args['jsonrpc'] != '2.0':
            raise ValueError('Wrong JSON-RPC version, must be 2.0')
        try:
            if decoded_args['method'].startswith('system.'):
                method = getattr(self, decoded_args['method'].replace('system.', ''))
                result = method(self, *decoded_args['params'])
            else:
                wmql_string = '%s %s' % (
                    decoded_args['method'],
                    ''.join((s + ' ' for s in decoded_args['params']))
                )
                self.sio.seek(0)
                self.shell.onecmd(wmql_string)
                self.sio.seek(0)
                result = self.sio.getvalue()
                self.sio.truncate(0)

            output = {
                'jsonrpc': "2.0",
                'result': result,
                'error': '',
                'id': decoded_args['id']
            }
            self.write(output)
        except KeyError:
            raise ValueError("JSON-RPC protocol not well implemented " + args)


    def get(self):
        return self.post()

    def describe(self, *args):
        return '''
        White Matter Query Language Command Line
    '''

    def completion(self, *args):
        completions = self.shell.completedefault(args[1])
        return completions


class MainHandler(tornado.web.RequestHandler):
    def initialize(
            self,
            host=None, port=None, path=None,
            filename=None, colortable=None,
            suffix='',
            websocketsuffix='ws'):
        self.filename = filename
        self.colortable = colortable
        self.host = host
        self.port = port
        self.path = path
        self.suffix = suffix
        self.websocketsuffix = websocketsuffix

    def get(self):
        return self.render(
            os.path.join(self.path, 'index.html'),
            host=self.host, port=self.port,
            websocket_server='ws://%(host)s:%(port)04d/%(websocketsuffix)s' % {
                'host': self.host,
                'port': self.port,
                'websocketsuffix': self.websocketsuffix
            },
            filename=self.filename, colortable=self.colortable,
            suffix=self.suffix
        )


class AtlasHandler(tornado.web.StaticFileHandler):
    def initialize(
            self,
            filename=None, colortable=None, suffix=''
    ):
        super(AtlasHandler, self).initialize('/')
        self.filename = filename
        self.colortable = colortable
        self.suffix = suffix

    def get(self, args):
        if args == 'atlas_%s.nii.gz' % self.suffix:
            super(AtlasHandler, self).get(self.filename)
        elif args == 'colortable_%s.txt' % self.suffix:
            super(AtlasHandler, self).get(self.colortable)
        else:
            raise ValueError("Unidentified file")


class NoCacheStaticHandler(tornado.web.StaticFileHandler):
    """ Request static file handlers for development and debug only.
    It disables any caching for static file.
    """
    def set_extra_headers(self, path):
        self.set_header('Cache-Control', 'no-cache')
        #self.set_header('Expires', '0')
        #now = datetime.datetime.now()
        #expiration = datetime.datetime(now.year - 1, now.month, now.day)
        #self.set_header('Last-Modified', expiration)


def xtk_server(atlas=None, colortable=None, hostname='localhost', port=9999, files_path=None, suffix='', shell=None):
    print "Using atlas", atlas
    global application

    folder_prefix = sysconfig.get_path(name='data')

    static_folder = os.path.join(
        folder_prefix,
        'tract_querier', 'tornado_server'
    )

    if colortable is None:
        print "No color table specified, using FreeSurfer"
        colortable = os.path.join(
            folder_prefix, 'tract_querier',
            'tornado_server', 'FreeSurferColorLUT.txt'
        )
    else:
        colortable = os.path.abspath(colortable)

    adapt_shell_callbacks(shell, 'http://%s:%04d/tracts' % (hostname, port))

    application = tornado.web.Application([
        (r"/", MainHandler, {
            'host': hostname, 'port': port,
            'path': static_folder,
            'filename': atlas,
            'colortable': colortable,
            'suffix': suffix
        }),
        (
            r"/static/(.*)",
            tornado.web.StaticFileHandler,
            {"path": static_folder}
        ),
        (r'/ws', WSHandler, {
            'shell': shell,
        }),
        (r'/jsonrpc', JSONRPCHandler, {
            'shell': shell,
        }),
        (
            r'/atlas/(.*)',
            AtlasHandler, {
                'filename': atlas,
                'colortable': colortable,
                'suffix': suffix
            }

        ),
        (r'/files/(.*)', NoCacheStaticHandler, {"path": files_path})
    ])

    application.listen(port)
    tornado.ioloop.IOLoop.instance().start()


def adapt_shell_callbacks(shell, url):
    import sys
    shell_save_query_callback = shell.save_query_callback
    shell_del_query_callback = shell.del_query_callback

    def save_query_callback(query_name, query_result):
        if shell_save_query_callback is not None:
            old_stdout = sys.stdout
            sys.stdout = shell.stdout
            filename = shell_save_query_callback(query_name, query_result)
            sys.stdout.flush()
            sys.stdout = old_stdout
        else:
            filename = None

        if filename is not None:
            try:
                json_encoder = json.JSONEncoder()
                action = {
                    'receiver': 'tract',
                    'action': 'add',
                    'file': filename,
                    'name': query_name,
                }

                action_json = json_encoder.encode(action)
                for client in websocket_clients:
                    client.write_message(action_json)
            except Exception, e:
                print "Websocket error:", e
        return filename

    def del_query_callback(query_name):
        if shell_del_query_callback is not None:
            old_stdout = sys.stdout
            sys.stdout = shell.stdout
            shell_del_query_callback(query_name)
            sys.stdout.flush()
            sys.stdout = old_stdout
        try:
            json_encoder = json.JSONEncoder()
            action = {
                'receiver': 'tract',
                'action': 'remove',
                'name': query_name,
            }

            action_json = json_encoder.encode(action)
            for client in websocket_clients:
                client.write_message(action_json)

        except Exception, e:
            print "Websocket error:", e

    shell.save_query_callback = save_query_callback
    shell.del_query_callback = del_query_callback




