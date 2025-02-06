#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re

from tract_querier.tests import datasets


def test_help_option(script_runner):
    ret = script_runner.run(["tract_math", "--help"])
    assert ret.success


def test_tract_math_count(script_runner):
    test_data = datasets.TestDataSet()
    tract_file = test_data.files["tract_file"]
    ret = script_runner.run(["tract_math", tract_file, "count"])
    assert re.search('[^0-9]6783[^0-9]', ret.stdout) is not None
    assert ret.success
