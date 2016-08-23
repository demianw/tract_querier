import hashlib
import os
from os import path
import tempfile
from six.moves import urllib

import unittest


FILES = {
    'tract_file': (
        'http://midas.kitware.com/bitstream/view/17631',
        'IIT3mean_left_hemisphere_small.trk',
        '\xe7\xec\xfd+\xd2n\xff\x96\xae\xb4\xdf+\x194\xdf\x81'
    ),
    'atlas_file': (
        'http://midas.kitware.com/bitstream/view/17622',
        'IIT3mean_desikan_2009.nii.gz',
        'vx\x13\xbaE\x1dR\t\xcd\xc9EF\x17\xa66\xb7'
    ),
    'query_uf_file': (
        'http://midas.kitware.com/bitstream/view/17627',
        'wmql_2_uf.qry',
        '\\+R\x8c<B#\xea\xfc\x9aE\xbd\xb0(\xbdn'
    )
}


class TestDataSet(unittest.TestCase):

    @unittest.skip("temporarily disabled")
    def __init__(self):
        self.dirname = path.join(
            tempfile.gettempdir(),
            'tract_querier'
        )

        self.files = {}

        if not path.exists(self.dirname):
            os.mkdir(self.dirname)

        for k, v in FILES.items():
            dst_filename = path.join(self.dirname, v[1])

            if (
                not path.exists(dst_filename) or
                hashlib.md5(open(dst_filename).read()).digest() != v[2]
            ):
                dl_file = urllib.request.urlopen(v[0])
                dst_file = open(dst_filename, 'wb')
                dst_file.write(dl_file.read())
                dst_file.close()

            if (
                hashlib.md5(open(dst_filename).read()).digest() != v[2]
            ):
                raise IOError('File %s url %s was not properly downloaded' % (v[1], v[0]))

            self.files[k] = dst_filename




