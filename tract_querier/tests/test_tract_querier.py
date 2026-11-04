#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from os import path

import numpy as np

from scripts.cli_tract_querier import affine_transform_tract
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


def test_affine_transform_tract():
    affine = np.array([
        [0., -1., 0., 10.],
        [1., 0., 0., 20.],
        [0., 0., 2., 30.],
        [0., 0., 0., 1.],
    ])
    tract = np.array([[0., 0., 0.], [1., 2., 3.]])
    np.testing.assert_allclose(
        affine_transform_tract(affine, tract),
        [[10., 20., 30.], [8., 21., 36.]]
    )


def test_tract_querier_bounding_box_affine_transform_translation(
    script_runner, tmp_path
):
    # A pure translation moves the tracts and the atlas together, so the
    # selected tracts must be the same as without any transform.
    test_data = datasets.TestDataSet()
    ret = script_runner.run([
        "tract_querier",
        "-a", test_data.files["atlas_file"],
        "-t", test_data.files["tract_file"],
        "-q", test_data.files["query_cst_file"],
        "-o", str(tmp_path / "test"),
        "--bounding_box_affine_transform",
        "1,0,0,10,0,1,0,20,0,0,1,30,0,0,0,1",
    ])
    assert ret.success
    assert "cortico_spinal.left: 000385" in ret.stdout
    assert "cortico_spinal.right: 000000" in ret.stdout
