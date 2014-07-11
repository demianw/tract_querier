import os
import json
from multiprocessing import Process
from StringIO import StringIO
import sys
import sysconfig
import uuid
import urllib

import tornado.ioloop
import tornado.web
import tornado.gen
import tornado.ioloop
import tornado.websocket


class WSHandler(tornado.websocket.WebSocketHandler):

    def initialize(self, shell=None, websocket_clients=None):
        self.shell = shell
        self.sio = StringIO()
        self.shell.stdout = self.sio
        self.json_encoder = json.JSONEncoder()
        self.json_decoder = json.JSONDecoder()
        self.websocket_clients = websocket_clients

        if websocket_clients is None:
            raise ValueError("websocket_clients needs to be a list of websockets")

    def open(self):
        self.websocket_clients.append(self)

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

    def on_close(self):
        self.websocket_clients.remove(self)


class JSONRPCHandler(tornado.web.RequestHandler):
    def initialize(self, shell=None, file_handler=None, websocket_clients=None):
        self.shell = shell
        self.file_handler = file_handler
        self.sio = StringIO()
        self.shell.stdout = self.sio
        self.json_encoder = json.JSONEncoder()
        self.json_decoder = json.JSONDecoder()

        self.websocket_clients = websocket_clients

        if websocket_clients is None:
            raise ValueError("websocket_clients needs to be a list of websockets")

    def post(self):
        error = ''
        result = ''
        decoded_args = self.json_decoder.decode(self.request.body)
        if decoded_args['jsonrpc'] != '2.0':
            raise ValueError('Wrong JSON-RPC version, must be 2.0')
        try:
            if decoded_args['method'].startswith('system.'):
                method = getattr(self, decoded_args['method'].replace('system.', ''))
                result = method(self, *decoded_args['params'])
            else:
                if decoded_args['method'] in ('save', 'download'):
                    tracts_to_download = decoded_args['params']
                    if '*' in tracts_to_download:
                        tracts_to_download = self.file_handler.tract_name_file.keys()

                    result = 'Tracts downloading....'

                    for query_name in tracts_to_download:
                        if query_name not in self.file_handler.tract_name_file:
                            result = ''
                            error = {
                                'code': -1,
                                'message': 'Tract %s not found' % query_name,
                                'data': ''
                            }
                            break

                        action = {
                            'receiver': 'tract',
                            'action': 'download',
                            'name': query_name,
                        }

                        action_json = self.json_encoder.encode(action)
                        for client in self.websocket_clients:
                            client.write_message(action_json)

                elif decoded_args['method'] == 'list':
                    result = ''
                    for k in self.file_handler.tract_name_file:
                        result += k + '\n'
                else:
                    if decoded_args['method'] == 'show':
                        decoded_args['method'] = 'save'

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
                'error': error,
                'id': decoded_args['id']
            }
            self.write(output)
            self.finish()
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


class TractDownloadHandler(tornado.web.RequestHandler):
    def initialize(self, file_handler=None):
        self.file_handler = file_handler

    def get(self, args):
        tract = str(args)
        self.set_header('Content-Type', 'application/octet-stream')
        self.set_header('Content-Disposition', 'attachment; filename=%s.trk' % (tract))
        f = file(self.file_handler.tract_name_file[tract])
        self.write(f.read())
        self.finish()


class MainHandler(tornado.web.RequestHandler):
    def initialize(
            self,
            host=None, port=None, path=None,
            filename=None, colortable=None,
            suffix='',
            websocket_suffix='ws',
            tract_download_suffix='tractdownload'
    ):
        self.filename = filename
        self.colortable = colortable
        self.host = host
        self.port = port
        self.path = path
        self.suffix = suffix
        self.websocket_suffix = websocket_suffix
        self.tract_download_suffix = tract_download_suffix

    def get(self):
        return self.render(
            os.path.join(self.path, 'index.html'),
            host=self.host, port=self.port,
            websocket_server='ws://%(host)s:%(port)04d/%(websocket_suffix)s' % {
                'host': self.host,
                'port': self.port,
                'websocket_suffix': self.websocket_suffix
            },
            tract_download_server='http://%(host)s:%(port)04d/%(tract_download_suffix)s' % {
                'host': self.host,
                'port': self.port,
                'tract_download_suffix': self.tract_download_suffix
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


def xtk_server(atlas=None, colortable=None, hostname='localhost', port=9999, files_path=None, suffix='', shell=None):
    print "Using atlas", atlas

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

    websocket_clients = []
    wmql_file_management = WMQLFileManagement(shell, websocket_clients)

    websocket_suffix = 'ws'
    tract_download_suffix = 'tractdownload'

    application = tornado.web.Application([
        (r"/", MainHandler, {
            'host': hostname, 'port': port,
            'path': static_folder,
            'filename': atlas,
            'colortable': colortable,
            'suffix': suffix,
            'websocket_suffix': websocket_suffix,
            'tract_download_suffix': tract_download_suffix
        }),
        (
            r"/static/(.*)",
            tornado.web.StaticFileHandler,
            {"path": static_folder}
        ),
        (r'/%s' % websocket_suffix, WSHandler, {
            'shell': shell,
            'websocket_clients': websocket_clients
        }),
        (r'/jsonrpc', JSONRPCHandler, {
            'shell': shell,
            'file_handler': wmql_file_management,
            'websocket_clients': websocket_clients,
        }),
        (
            r'/atlas/(.*)',
            AtlasHandler, {
                'filename': atlas,
                'colortable': colortable,
                'suffix': suffix
            }

        ),
        (
            r'/%s/(.*)' % tract_download_suffix,
            TractDownloadHandler, {
                'file_handler': wmql_file_management
            }
        ),
        (r'/files/(.*)', NoCacheStaticHandler, {"path": files_path})
    ])
    application.listen(port)

    try:
        tornado.ioloop.IOLoop.instance().start()
    except KeyboardInterrupt:
        tornado.ioloop.IOLoop.instance().stop()
        wmql_file_management.clean_files()


class WMQLFileManagement(object):
    def __init__(self, shell, websocket_clients):
        self.json_encoder = json.JSONEncoder()
        self.tract_name_file = {}
        self.adapt_shell(shell)
        self.websocket_clients = websocket_clients

    def __del__(self):
        self.clean_files()

    def clean_files(self):
        for k, v in self.tract_name_file.iteritems():
            print "Removing", k, v
            os.remove(v)
        self.tract_name_file = {}

    def adapt_shell(self, shell):
        self.shell = shell
        self.old_shell_save_query_callback = shell.save_query_callback
        self.old_shell_del_query_callback = shell.del_query_callback

        self.shell.save_query_callback = self.save_query_callback
        self.shell.del_query_callback = self.del_query_callback

    def save_query_callback(self, query_name, query_result):
        if self.old_shell_save_query_callback is not None:
            old_stdout = sys.stdout
            sys.stdout = self.shell.stdout
            filename = self.old_shell_save_query_callback(query_name, query_result)
            sys.stdout.flush()
            sys.stdout = old_stdout
        else:
            filename = None

        if filename is not None:
            try:
                new_filename = str(uuid.uuid4()).replace('-', 'W') + '_' + filename
                os.rename(filename, new_filename)

                if query_name in self.tract_name_file:
                    os.remove(self.tract_name_file[query_name])
                self.tract_name_file[query_name] = new_filename

                action = {
                    'receiver': 'tract',
                    'action': 'add',
                    'file': new_filename,
                    'name': query_name,
                }

                action_json = self.json_encoder.encode(action)
                for client in self.websocket_clients:
                    client.write_message(action_json)
            except Exception, e:
                print "Websocket error:", e
        return filename

    def del_query_callback(self, query_name):
        if self.old_shell_del_query_callback is not None:
            old_stdout = sys.stdout
            sys.stdout = self.shell.stdout
            self.old_shell_del_query_callback(query_name)
            sys.stdout.flush()
            sys.stdout = old_stdout
        try:
            action = {
                'receiver': 'tract',
                'action': 'remove',
                'name': query_name,
            }

            action_json = self.json_encoder.encode(action)
            for client in self.websocket_clients:
                client.write_message(action_json)

            os.remove(self.tract_name_file[query_name])
            del self.tract_name_file[query_name]
        except Exception, e:
            print "Websocket error:", e
