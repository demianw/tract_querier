import os
from os import path
import re
import subprocess
import sys
import unittest

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

@unittest.skip()
def test_tract_querier_help():
    popen = subprocess.Popen(
        [PYTHON, TRACT_QUERIER_SCRIPT],
        shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=ENVIRON
    )
    popen.wait()
    stderr_text = ''.join(popen.stderr.readlines())
    assert 'error: incorrect number of arguments' in stderr_text
    assert popen.returncode > 0

@unittest.skip()
def test_tract_math_help():
    popen = subprocess.Popen(
        [PYTHON, TRACT_MATH_SCRIPT],
        shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=ENVIRON
    )
    popen.wait()
    stderr_text = ''.join(popen.stderr.readlines())
    assert 'error: too few arguments' in stderr_text
    assert popen.returncode > 0

@unittest.skip()
def test_tract_math_count():
    popen = subprocess.Popen(
        [PYTHON, TRACT_MATH_SCRIPT, TEST_DATA.files['tract_file'], 'count'],
        shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=ENVIRON
    )
    popen.wait()
    stdout_text = ''.join(popen.stdout.readlines())
    assert re.search('[^0-9]6783[^0-9]', stdout_text) is not None
    assert popen.returncode == 0

@unittest.skip()
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
    assert 'uncinate.left: 000102' in stdout_text
    assert 'uncinate.right: 000000' in stdout_text
    assert path.exists(output_prefix + '_uncinate.left.trk')
    assert popen.returncode == 0
    if path.exists(output_prefix + '_uncinate.left.trk'):
        os.remove(output_prefix + '_uncinate.left.trk')
