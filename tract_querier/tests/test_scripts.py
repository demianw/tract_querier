from nose.tools import assert_equal, assert_greater, assert_in, assert_is_not_none, assert_true

import os
from os import path
import re
import subprocess
import sys

from tract_querier.tests import datasets
from functools import reduce


PACKAGE_ROOT_DIR = path.dirname(path.dirname(path.dirname(__file__)))

TRACT_QUERIER_SCRIPT = path.join(
    PACKAGE_ROOT_DIR,
    'scripts', 'tract_querier'
)

TRACT_MATH_SCRIPT = path.join(
    PACKAGE_ROOT_DIR,
    'scripts', 'tract_math'
)

PYTHON = sys.executable

ENVIRON = os.environ.copy()
sys.path.insert(0, PACKAGE_ROOT_DIR)
ENVIRON['PYTHONPATH'] = reduce(lambda x, y: '%s:%s' % (x, y), sys.path)

TEST_DATA = datasets.TestDataSet()

def test_tract_querier_help():
    popen = subprocess.Popen(
        [PYTHON, TRACT_QUERIER_SCRIPT],
        shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=ENVIRON
    )
    popen.wait()
    stderr_text = ''.join(popen.stderr.readlines())
    assert_in('error: incorrect number of arguments', stderr_text)
    assert_greater(popen.returncode, 0)

def test_tract_math_help():
    popen = subprocess.Popen(
        [PYTHON, TRACT_MATH_SCRIPT],
        shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=ENVIRON
    )
    popen.wait()
    stderr_text = ''.join(popen.stderr.readlines())
    assert_in('error: too few arguments', stderr_text)
    assert_greater(popen.returncode, 0)

def test_tract_math_count():
    popen = subprocess.Popen(
        [PYTHON, TRACT_MATH_SCRIPT, TEST_DATA.files['tract_file'], 'count'],
        shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=ENVIRON
    )
    popen.wait()
    stdout_text = ''.join(popen.stdout.readlines())
    assert_is_not_none(re.search('[^0-9]6783[^0-9]', stdout_text))
    assert_equal(popen.returncode, 0)

def test_tract_querier_query():
    output_prefix = '%s/test' % TEST_DATA.dirname
    popen = subprocess.Popen(
        [PYTHON, TRACT_QUERIER_SCRIPT] +
        ('-a %(atlas_file)s -t %(tract_file)s -q %(query_uf_file)s' % TEST_DATA.files).split() +
        (' -o %s' % output_prefix).split(),
        shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=ENVIRON
    )
    popen.wait()
    stdout_text = ''.join(popen.stdout.readlines())
    assert_in('uncinate.left: 000102', stdout_text)
    assert_in('uncinate.right: 000000', stdout_text)
    assert_true(path.exists(output_prefix + '_uncinate.left.trk'))
    assert_equal(popen.returncode, 0)
    if path.exists(output_prefix + '_uncinate.left.trk'):
        os.remove(output_prefix + '_uncinate.left.trk')
