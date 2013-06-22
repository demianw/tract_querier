import os
import sys

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
            websocketsuffix='ws'):
        self.filename = filename
        self.colortable = colortable
        self.host = host
        self.port = port
        self.path = path
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
            filename=self.filename, colortable=self.colortable
        )


class AtlasHandler(tornado.web.StaticFileHandler):
    def initialize(
            self,
            filename=None, colortable=None,
    ):
        super(AtlasHandler, self).initialize('/')
        self.filename = filename
        self.colortable = colortable

    def get(self, args):
        print "GETTING", args
        if args == 'atlas.nii.gz':
            super(AtlasHandler, self).get(self.filename)
        elif args == 'colortable.txt':
            super(AtlasHandler, self).get(self.colortable)
        else:
            raise ValueError("Unidentified file")


class TractHandler(tornado.web.RequestHandler):
    def post(self):
        try:
            arg = self.get_argument("file")
            for client in websocket_clients:
                client.write_message(arg)
        except Exception, e:
            print e


def xtk_server(atlas, colortable=None, port=9999, files_path=None):
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
            'colortable': colortable
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
                'colortable': colortable
            }

        ),
        (r'/files/(.*)', tornado.web.StaticFileHandler, {"path": files_path})
    ])
    application.listen(port)
    tornado.ioloop.IOLoop.instance().start()
