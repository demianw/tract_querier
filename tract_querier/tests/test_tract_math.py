#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import io
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


def test_tract_math_existing_output(script_runner, tmp_path):
    test_data = datasets.TestDataSet()
    tract_file = test_data.files["tract_file"]
    output = tmp_path / "count.csv"
    output.write_text("old\n")

    # Without --force and without an interactive stdin, do not prompt: fail
    ret = script_runner.run(
        ["tract_math", tract_file, "count", str(output)],
        stdin=io.StringIO("")
    )
    assert not ret.success
    assert "--force" in ret.stderr
    assert output.read_text() == "old\n"

    ret = script_runner.run(
        ["tract_math", tract_file, "count", str(output), "--force"],
        stdin=io.StringIO("")
    )
    assert ret.success
    with open(output, newline='') as f:
        rows = list(csv.reader(f))
    assert len(rows) == 2
    assert "6783" in rows[1]
