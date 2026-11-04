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


@pytest.mark.parametrize('operation,missing_module,parameters', [
    ('tract_map_image', 'scipy', ['image.nii.gz', 'value', 'out.trk']),
    ('tract_deform', 'scipy', ['image.nii.gz', 'out.trk']),
    ('tract_smooth', 'sklearn', ['1', 'out.trk']),
])
def test_tract_math_missing_optional_dependency(
    monkeypatch, operation, missing_module, parameters
):
    import sys
    import numpy
    from tract_querier.tractography import Tractography
    from tract_querier.tract_math import operations

    for module in list(sys.modules):
        if module == missing_module or module.startswith(missing_module + '.'):
            monkeypatch.setitem(sys.modules, module, None)
    monkeypatch.setitem(sys.modules, missing_module, None)

    tractography = Tractography([numpy.zeros((2, 3))])
    with pytest.raises(ImportError, match=r"\[tract_math\]"):
        operations[operation](tractography, *parameters)
