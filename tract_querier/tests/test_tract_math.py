#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re

import pytest

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


@pytest.mark.parametrize('parameters', [
    ['tract_volume'],
    ['count', 'extra', 'out.txt'],
])
def test_tract_math_wrong_number_of_parameters(script_runner, parameters):
    test_data = datasets.TestDataSet()
    tract_file = test_data.files["tract_file"]
    ret = script_runner.run(["tract_math", tract_file] + parameters)
    assert not ret.success
    assert 'Wrong number of' in ret.stderr
    assert 'Traceback' not in ret.stdout + ret.stderr


def test_tract_math_operation_error_is_not_a_usage_error(monkeypatch):
    import sys
    from scripts import cli_tract_math
    from tract_querier.tract_math import tract_math_operation

    @tract_math_operation(': always fails')
    def failing_operation(optional_flags, tractography):
        raise TypeError('error inside the operation')

    test_data = datasets.TestDataSet()
    monkeypatch.setitem(
        cli_tract_math.operations, 'failing_operation', failing_operation
    )
    monkeypatch.setattr(
        sys, 'argv',
        ['tract_math', test_data.files["tract_file"], 'failing_operation']
    )
    with pytest.raises(TypeError, match='error inside the operation'):
        cli_tract_math.main()


def test_tract_math_file_read_error_on_stderr(script_runner, tmp_path):
    test_data = datasets.TestDataSet()
    bad_file = tmp_path / 'tracts.txt'
    bad_file.write_text('')
    ret = script_runner.run(
        ["tract_math", test_data.files["tract_file"], str(bad_file), "count"]
    )
    assert 'Error reading file' in ret.stderr
    assert 'Error reading file' not in ret.stdout
