import hashlib
import os
from os import path
import tempfile
import urllib.request


import unittest


FILES = {
    'tract_file': (
        'https://osf.io/ugyqa/files/osfstorage/6788df92de6b5942fae241be',
        'IIT3mean_left_hemisphere_small.trk',
        '\xe7\xec\xfd+\xd2n\xff\x96\xae\xb4\xdf+\x194\xdf\x81'
    ),
    'atlas_file': (
        'https://osf.io/ugyqa/files/osfstorage/6788df9090f7678e2caeae01',
        'IIT3mean_desikan_2009.nii.gz',
        'vx\x13\xbaE\x1dR\t\xcd\xc9EF\x17\xa66\xb7'
    ),
    'query_uf_file': (
        'https://osf.io/ugyqa/files/osfstorage/6788df9781590325003424e1',
        'wmql_2_uf.qry',
        '\\+R\x8c<B#\xea\xfc\x9aE\xbd\xb0(\xbdn'
    )
}


class TestDataSet(unittest.TestCase):
    def __init__(self):
        self.dirname = path.join(
            tempfile.gettempdir(),
            'tract_querier'
        )

        self.files = {}

        if not path.exists(self.dirname):
            os.mkdir(self.dirname)

        for k, (url, filename, md5) in FILES.items():
            dst_filename = path.join(self.dirname, filename)
            if (
                not path.exists(dst_filename) or
                hashlib.md5(open(dst_filename).read().encode('utf-8')).digest() != md5
            ):
                request = urllib.request.Request(url)
                try:
                    response = urllib.request.urlopen(request)
                except urllib.error.HTTPError as e:
                    if e.status not in (307, 308):
                        raise
                    redirected_url = urllib.parse.urljoin(url, e.headers['Location'])
                    response = urllib.request.urlopen(redirected_url)
                with open(dst_filename, 'wb') as out_file:
                    while True:
                        chunk = response.read(8192)  # Read 8 KB at a time
                        if not chunk:
                            break
                        out_file.write(chunk)
            if (
                hashlib.md5(open(dst_filename).read().encode('utf-8')).digest() != md5
            ):
                continue
                raise IOError('File %s url %s was not properly downloaded' % (filename, url))

            self.files[k] = dst_filename




