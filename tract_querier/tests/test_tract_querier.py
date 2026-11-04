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
    query_commissural_file = test_data.files["query_commissural_file"]
    query_cst_file = test_data.files["query_cst_file"]
    query_uf_file = test_data.files["query_uf_file"]
    tract_file = test_data.files["tract_file"]
    output_prefix = test_data.dirname + "/test"

    ret = script_runner.run(["tract_querier", "-a", atlas_file,"-t", tract_file, "-q", query_commissural_file, "-o", output_prefix])
    tract_fname_end = "_all_commissural.trk"
    assert "all_commissural: 000003" in ret.stdout
    assert path.exists(output_prefix + tract_fname_end)
    assert ret.success
    if path.exists(output_prefix + tract_fname_end):
        os.remove(output_prefix + tract_fname_end)

    ret = script_runner.run(["tract_querier", "-a", atlas_file,"-t", tract_file, "-q", query_cst_file, "-o", output_prefix])
    tract_fname_end = "_cortico_spinal.left.trk"
    assert "cortico_spinal.left: 000385" in ret.stdout
    assert "cortico_spinal.right: 000000" in ret.stdout
    assert path.exists(output_prefix + tract_fname_end)
    assert ret.success
    if path.exists(output_prefix + tract_fname_end):
        os.remove(output_prefix + tract_fname_end)

    ret = script_runner.run(["tract_querier", "-a", atlas_file,"-t", tract_file, "-q", query_uf_file, "-o", output_prefix])
    tract_fname_end = "_uncinate.left.trk"
    assert "uncinate.left: 000102" in ret.stdout
    assert "uncinate.right: 000000" in ret.stdout
    assert path.exists(output_prefix + tract_fname_end)
    assert ret.success
    if path.exists(output_prefix + tract_fname_end):
        os.remove(output_prefix + tract_fname_end)


def test_tract_querier_missing_query_file(script_runner, tmp_path):
    for query_file in ("nonexistent_file.qry", str(tmp_path / "missing")):
        ret = script_runner.run([
            "tract_querier", "-a", "atlas.nii.gz", "-t", "tracts.trk",
            "-q", query_file, "-o", str(tmp_path / "out")
        ])
        assert not ret.success
        assert "Query file not found: " + query_file in ret.stderr
        assert "Invalid expression" not in ret.stderr


def test_tract_querier_query_file_syntax_error(script_runner, tmp_path):
    query_file = tmp_path / "broken.qry"
    query_file.write_text("a = 1\nb = (a or\n")
    ret = script_runner.run([
        "tract_querier", "-a", "atlas.nii.gz", "-t", "tracts.trk",
        "-q", str(query_file), "-o", str(tmp_path / "out")
    ])
    assert not ret.success
    assert "syntax error in file " + str(query_file) + " line 2" in ret.stderr


def test_tract_querier_inline_query(script_runner):
    test_data = datasets.TestDataSet()
    output_prefix = test_data.dirname + "/test_inline"
    query = (
        "import FreeSurfer.qry\n"
        "hemisphere.left |= '*.left'\n"
        "hemisphere.right |= '*.right'\n"
        "all_commissural = endpoints_in(hemisphere.left) and "
        "endpoints_in(hemisphere.right)\n"
    )
    ret = script_runner.run([
        "tract_querier", "-a", test_data.files["atlas_file"],
        "-t", test_data.files["tract_file"], "-q", query, "-o", output_prefix
    ])
    assert ret.success
    assert "all_commissural: 000003" in ret.stdout
    output_file = output_prefix + "_all_commissural.trk"
    assert path.exists(output_file)
    os.remove(output_file)
