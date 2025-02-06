#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from os import path

from tract_querier.tests import datasets


def test_help_option(script_runner):
    ret = script_runner.run(["tract_querier", "--help"])
    assert ret.success


def test_tract_querier_query(script_runner):
    test_data = datasets.TestDataSet()
    atlas_file = test_data.files["atlas_file"]
    query_uf_file = test_data.files["query_uf_file"]
    tract_file = test_data.files["tract_file"]
    output_prefix = test_data.dirname + "/test"

    ret = script_runner.run(["tract_querier", "-a", atlas_file,"-t", tract_file, "-q", query_uf_file, "-o", output_prefix])
    tract_fname_end = "_uncinate.left.trk"
    assert "uncinate.left: 000102" in ret.stdout
    assert "uncinate.right: 000000" in ret.stdout
    assert path.exists(output_prefix + tract_fname_end)
    assert ret.success
    if path.exists(output_prefix + tract_fname_end):
        os.remove(output_prefix + tract_fname_end)
