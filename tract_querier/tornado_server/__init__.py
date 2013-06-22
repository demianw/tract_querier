import os
import sys
import json

import tornado.ioloop
import tornado.web
import tornado.gen
import tornado.ioloop
import tornado.websocket

websocket_clients = []


class WSHandler(tornado.websocket.WebSocketHandler):

    def open(self):
        global websocket_clients
        websocket_clients.append(self)

    def on_message(self, message):
        # self.write_message(u"You Said: " + message)
        global change
        change = message

        #self.write_message(message + "Response")

    def on_close(self):
        global websocket_clients
        print 'connection closed'
        websocket_clients.remove(self)


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


class TractHandler(tornado.web.RequestHandler):
    def initialize(self):
        self.json_encoder = json.JSONEncoder()

    def post(self):
        try:
            action = {
                'action': self.get_argument('action'),
                'name': self.get_argument('name')
            }

            if action['action'] == 'add':
                action['file'] = self.get_argument("file")

            action_json = self.json_encoder.encode(action)
            for client in websocket_clients:
                client.write_message(action_json)
        except Exception, e:
            print e


def xtk_server(atlas, colortable=None, port=9999, files_path=None, suffix=''):
    print "Using atlas", atlas
    global application

    static_folder = os.path.join(
        sys.prefix,
        'tract_querier', 'tornado_server'
    )

    if colortable is None:
        print "No color table specified, using FreeSurfer"
        colortable = os.path.join(
            sys.prefix, 'tract_querier',
            'tornado_server', 'FreeSurferColorLUT.txt'
        )
    else:
        colortable = os.path.abspath(colortable)

    application = tornado.web.Application([
        (r"/", MainHandler, {
            'host': 'localhost', 'port': 9999,
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
        (r'/ws', WSHandler),
        (r'/tracts', TractHandler),
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
